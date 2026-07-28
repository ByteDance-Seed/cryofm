from __future__ import annotations

import logging
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Mapping

import torch

from cryoseed.runtime.distributed import RuntimeContext

logger = logging.getLogger(__name__)

_DONE_STATUS = "done"
_ERROR_STATUS = "error"


@dataclass(frozen=True)
class ExternalReconstructJob:
    """Description of one external reconstruction job."""

    name: str
    work_dir: str
    request_path: str
    command_extra_args: tuple[str, ...] = ()
    required_outputs: tuple[str, ...] = ()
    env_overrides: Mapping[str, str] = field(default_factory=dict)

    def normalized(self) -> ExternalReconstructJob:
        """Return a copy with absolute filesystem paths."""
        return ExternalReconstructJob(
            name=str(self.name),
            work_dir=os.path.abspath(self.work_dir),
            request_path=os.path.abspath(self.request_path),
            command_extra_args=tuple(str(arg) for arg in self.command_extra_args),
            required_outputs=tuple(os.path.abspath(path) for path in self.required_outputs),
            env_overrides={str(key): str(value) for key, value in self.env_overrides.items()},
        )


@dataclass(frozen=True)
class ExternalReconstructResult:
    """Execution result for one external reconstruction job."""

    job: ExternalReconstructJob
    returncode: int


class ExternalReconstructManager:
    """Coordinate external reconstruction jobs across distributed ranks.

    This manager is intentionally a command runner plus distributed status
    synchronizer. A job is considered successful when the external command
    exits cleanly and the result remains readable by the main process.

    In particular, this layer does not require the external program to prove
    that it produced a semantically updated reconstruction. Returning ``0`` and
    leaving the prewritten result untouched is treated as an acceptable
    identity-style outcome.
    """

    def __init__(
        self,
        *,
        runtime: RuntimeContext,
        executable_env_var: str = "CRYOSEED_EXTERNAL_RECONSTRUCT_EXECUTABLE",
        store_port_env_var: str = "CRYOSEED_EXTERNAL_RECONSTRUCT_STORE_PORT",
        sync_timeout_env_var: str = "CRYOSEED_EXTERNAL_RECONSTRUCT_SYNC_TIMEOUT_SECONDS",
        status_namespace: str = "external_reconstruct",
    ) -> None:
        self.runtime = runtime
        self.executable_env_var = str(executable_env_var)
        self.store_port_env_var = str(store_port_env_var)
        self.sync_timeout_env_var = str(sync_timeout_env_var)
        self.status_namespace = str(status_namespace).strip("/") or "external_reconstruct"
        self._store = None

    def run(
        self,
        jobs: list[ExternalReconstructJob],
        *,
        run_id: str,
    ) -> list[ExternalReconstructResult]:
        """Run a batch of jobs and return one result per job.

        Success here means that rank0 launched the command(s), observed clean
        process exit, and propagated that command-level success to other ranks.
        It does not imply that the external tool must have produced a changed
        output volume; an identity outcome is still acceptable as long as the
        downstream result path remains readable.
        """
        normalized_jobs = [job.normalized() for job in jobs]
        self._validate_jobs(normalized_jobs)

        if self.runtime.rank == 0:
            return self._run_on_rank0(normalized_jobs, run_id=run_id)

        status, message = self._wait_for_status(run_id=run_id)
        if status != _DONE_STATUS:
            raise RuntimeError(message or "External reconstruction failed on rank0.")

        self._ensure_outputs_exist(normalized_jobs)
        return [ExternalReconstructResult(job=job, returncode=0) for job in normalized_jobs]

    def build_launch_env(self, extra_env: Mapping[str, str] | None = None) -> dict[str, str]:
        """Return a sanitized environment for launching external jobs."""
        env = os.environ.copy()
        exact_keys = {
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
            "MASTER_HOST",
        }
        prefix_keys = (
            "TORCHELASTIC_",
            "PMI_",
            "PMIX_",
            "OMPI_",
            "MPI_",
        )
        slurm_keys = {
            "SLURM_PROCID",
            "SLURM_LOCALID",
            "SLURM_NODEID",
            "SLURM_NTASKS",
            "SLURM_NPROCS",
            "SLURM_STEP_ID",
            "SLURM_STEPID",
            "SLURM_STEP_NUM_NODES",
            "SLURM_STEP_NUM_TASKS",
            "SLURM_STEP_NODELIST",
            "SLURM_STEP_LAUNCHER_PORT",
            "SLURM_TASK_PID",
        }
        for key in exact_keys | slurm_keys:
            env.pop(key, None)
        for key in list(env.keys()):
            if key.startswith(prefix_keys):
                env.pop(key, None)
        if extra_env:
            env.update({str(key): str(value) for key, value in extra_env.items()})
        return env

    def _run_on_rank0(
        self,
        jobs: list[ExternalReconstructJob],
        *,
        run_id: str,
    ) -> list[ExternalReconstructResult]:
        """Launch and wait for all jobs on rank0.

        This method validates command-level success only. The external program
        may legally behave like an identity transform and keep the prewritten
        result unchanged, so long as it exits successfully.
        """
        processes: list[tuple[ExternalReconstructJob, subprocess.Popen]] = []
        try:
            processes = [(job, self._launch(job)) for job in jobs]
            results: list[ExternalReconstructResult] = []
            failures: list[tuple[str, int]] = []

            for job, process in processes:
                returncode = int(process.wait())
                results.append(ExternalReconstructResult(job=job, returncode=returncode))
                if returncode != 0:
                    failures.append((job.name, returncode))

            if failures:
                summary = ", ".join(
                    f"{job_name} (exit_code={returncode})" for job_name, returncode in failures
                )
                raise RuntimeError(f"External reconstruction command failed for {summary}.")

            self._ensure_outputs_exist(jobs)
        except Exception as exc:
            self._publish_status(
                run_id=run_id,
                status=_ERROR_STATUS,
                message=f"External reconstruction failed on rank0: {exc}",
            )
            for _, process in processes:
                if process.poll() is None:
                    process.terminate()
            raise

        self._publish_status(run_id=run_id, status=_DONE_STATUS)
        return results

    def _launch(self, job: ExternalReconstructJob) -> subprocess.Popen:
        executable = os.environ.get(self.executable_env_var, "").strip()
        if not executable:
            raise RuntimeError(
                "External reconstruction is enabled, but environment variable "
                f"`{self.executable_env_var}` is not set."
            )

        command = shlex.split(executable)
        if len(command) == 0:
            raise RuntimeError(
                f"Environment variable `{self.executable_env_var}` must define a valid executable command."
            )

        command.extend(job.command_extra_args)
        logger.info("Launching external reconstruction job `%s`: %s", job.name, " ".join(command))
        return subprocess.Popen(
            command,
            cwd=job.work_dir,
            env=self.build_launch_env(job.env_overrides),
        )

    def _validate_jobs(self, jobs: list[ExternalReconstructJob]) -> None:
        if len(jobs) == 0:
            raise ValueError("At least one external reconstruction job is required.")

        names: set[str] = set()
        for job in jobs:
            if job.name in names:
                raise ValueError(f"Duplicate external reconstruction job name: {job.name!r}")
            names.add(job.name)

            if not os.path.isabs(job.work_dir):
                raise ValueError(f"job.work_dir must be an absolute path, got {job.work_dir!r}")
            if not os.path.isabs(job.request_path):
                raise ValueError(f"job.request_path must be an absolute path, got {job.request_path!r}")

            request_parent = Path(job.request_path).parent
            if request_parent != Path(job.work_dir):
                logger.debug(
                    "External reconstruction job `%s` uses request file outside work_dir: %s",
                    job.name,
                    job.request_path,
                )

    def _ensure_outputs_exist(self, jobs: list[ExternalReconstructJob]) -> None:
        missing: list[str] = []
        for job in jobs:
            if len(job.required_outputs) == 0:
                continue
            for path in job.required_outputs:
                if not os.path.exists(path):
                    missing.append(f"{job.name}: {path}")
        if missing:
            joined = ", ".join(missing)
            raise FileNotFoundError(f"External reconstruction outputs are missing: {joined}")

    def _store_host(self) -> str:
        return os.environ.get("MASTER_ADDR", "127.0.0.1")

    def _store_port(self) -> int:
        port_str = os.environ.get(self.store_port_env_var)
        if port_str:
            return int(port_str)
        master_port = int(os.environ.get("MASTER_PORT", "29500"))
        return master_port + 17

    def _store_timeout(self) -> timedelta:
        timeout_seconds = int(os.environ.get(self.sync_timeout_env_var, str(24 * 3600)))
        return timedelta(seconds=max(1, timeout_seconds))

    def _get_store(self):
        if not self.runtime.is_distributed:
            return None
        if self._store is None:
            self._store = torch.distributed.TCPStore(
                self._store_host(),
                self._store_port(),
                self.runtime.world_size,
                self.runtime.rank == 0,
                self._store_timeout(),
            )
        return self._store

    def _status_prefix(self, *, run_id: str) -> str:
        safe_run_id = str(run_id).strip("/")
        if not safe_run_id:
            raise ValueError("run_id must be a non-empty string.")
        return f"{self.status_namespace}/{safe_run_id}"

    def _publish_status(self, *, run_id: str, status: str, message: str = "") -> None:
        store = self._get_store()
        if store is None:
            return
        prefix = self._status_prefix(run_id=run_id)
        store.set(f"{prefix}/message", message)
        store.set(f"{prefix}/status", status)

    def _wait_for_status(self, *, run_id: str) -> tuple[str, str]:
        store = self._get_store()
        if store is None:
            return _DONE_STATUS, ""
        prefix = self._status_prefix(run_id=run_id)
        status = store.get(f"{prefix}/status").decode("utf-8")
        message = store.get(f"{prefix}/message").decode("utf-8")
        return status, message