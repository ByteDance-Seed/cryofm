from collections import OrderedDict
import os
import time
import logging
from dataclasses import dataclass
import torch

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from cryoseed.fft.fft_torch import fourier_to_primal_3d, primal_to_fourier_3d
from cryoseed.runtime.distributed import RuntimeContext, is_rank0
from cryoseed.cryoem.mask import lowpass_mask, load_mask_mrc, spherical_mask
from cryoseed.cryoem.rotation import quaternion_to_matrix
from cryoseed.config import MainConfig
from cryoseed.data import (
    DataBatch,
    ParticleDataset,
    build_dataloader,
    build_distributed_dataloader,
    data_collate_fn,
    save_mrc,
    single_thread_worker_init_fn,
)
from cryoseed.state import OptimState
from cryoseed.modules.volume import VoxelGrid
from cryoseed.modules.statistics import NoiseVariance
from cryoseed.modules.pose import Pose
from cryoseed.optim.pose import PoseSearcher
from cryoseed.optim import SGDSolver
from cryoseed.optim.scheduler import AbInitioScheduler
from cryoseed.ops.transforms import downsample3d
from cryoseed.utils.logging import setup_logging, log_block, log_config, log_state
from cryoseed.utils.reproducibility import set_seed


LOGGER = logging.getLogger(__name__)


# Engine-facing metrics and logs follow:
# quality -> assignment -> pose -> volume -> control.
# Keep summaries and progress displays in that order when the same block mixes
# loss/confidence, assignment stability, pose stability, and map-change terms.
@dataclass(frozen=True)
class ScheduleCheckSnapshot:
    pose_search_scope: str
    pose_search_strategy: str
    side_length: int
    healpix_order: int
    trans_grid_extent: float
    trans_grid_samples: int
    use_pose_translation_as_center: bool
    side_length_resolution: float
    avg_confidence: float
    avg_volume_class_confidence: float
    volume_class_change_rate: float
    ema_volume_class_change_rate: float | None
    ema_rot_update_rms: float | None
    ema_trans_update_rms: float | None


@dataclass(frozen=True)
class _ScheduleProbeBatch:
    image: torch.Tensor
    particle_index: torch.Tensor
    ctf: torch.Tensor | None = None


class AbInitioEngine(torch.nn.Module):

    def __init__(
        self,
        config: MainConfig,
        runtime: RuntimeContext,
        resume_checkpoint_path: str | None = None,
        auto_resume: bool = False,
    ):
        super().__init__()

        # config
        self.config = config.validate_for_command("abinitio")
        self.resume_checkpoint_path = resume_checkpoint_path
        self.auto_resume = bool(auto_resume)
        self.runtime = runtime
        self.init_particles_per_volume = int(
            self.config.abinitio.engine.init_particles_per_volume
        )
        if int(self.init_particles_per_volume) <= 0:
            raise ValueError(
                "init_particles_per_volume must be > 0, got "
                f"{self.init_particles_per_volume}"
            )

        # device
        device = runtime.device
        device_mesh = runtime.device_mesh
        self.device = device
        self.device_mesh = device_mesh

        # dataloader
        dataset = ParticleDataset(
            config.io.star_path,
            data_prefix=config.io.data_path,
            num_particles=config.data.num_particles,
            selection_seed=int(config.reproduce.seed),
            image_size=config.data.image_size,
            angpix=config.data.angpix,
            default_optic_params=config.data.default_optic_params,
            default_particle_params=config.data.default_particle_params,
        )
        dataset.populate_data_config(self.config.data)
        self.dataset = dataset

        if runtime.is_distributed:
            dataloader, sampler = build_distributed_dataloader(
                dataset,
                batch_size=config.data.batch_size,
                shuffle=True,
                num_workers=config.data.num_workers,
                device=device,
                seed=int(config.reproduce.seed),
                worker_init_fn=single_thread_worker_init_fn,
                multiprocessing_context="spawn",
                device_mesh=device_mesh,
            )
        else:
            dataloader = build_dataloader(
                dataset,
                batch_size=config.data.batch_size,
                shuffle=True,
                num_workers=config.data.num_workers,
                pin_memory=(device.type=="cuda"),
                worker_init_fn=single_thread_worker_init_fn,
                multiprocessing_context="spawn",
            )
            sampler = None

        self.dataloader = dataloader
        self.sampler = sampler
        self.solvent_mask = self.build_solvent_mask()

        # modules
        self.volume = VoxelGrid.from_config(config, device=device, device_mesh=device_mesh, requires_grad=True, requires_accum=False)
        self.noise = NoiseVariance.from_config(config, device=device)

        self.pose = Pose.from_config(config, device=device, device_mesh=device_mesh)

        # optimization
        self.state = OptimState.from_config(config, command="abinitio")
        self.pose_searcher = PoseSearcher(
            state=self.state,
            volume=self.volume,
            noise=self.noise,
            pose=self.pose,
            config=config,
            device=device,
            device_mesh=device_mesh,
        )
        self.solver = SGDSolver(
            state=self.state,
            pose_searcher=self.pose_searcher,
            learning_rate=float(config.abinitio.solver.learning_rate),
            learning_rate_decay=float(config.abinitio.solver.learning_rate_decay),
            momentum=float(config.abinitio.solver.momentum),
        )
        self.scheduler = AbInitioScheduler(self.state, device=self.device).from_config(config)
        self.current_loss: float | None = None
        self.volume_real: torch.Tensor | None = None
        self._init_summary: dict[str, object] | None = None
        self._last_schedule_check_volume: torch.Tensor | None = None
        self.probe: _ScheduleProbeBatch | None = None
        self._effective_schedule_check_interval_iters: int | None = None
        self._schedule_check_epoch_total_iters: int | None = None
        self._confidence_sum = 0.0
        self._confidence_count = 0
        self._volume_class_confidence_sum = 0.0
        self._volume_class_confidence_count = 0
        self._latest_avg_confidence = 0.0
        self._latest_avg_volume_class_confidence = 0.0
        self._ema_loss: float | None = None
        self._ema_loss_change: float | None = None

    def _load_optional_module_state(self, module, state_dict, name: str) -> None:
        if module is None:
            if state_dict is not None:
                raise ValueError(f"Checkpoint contains `{name}` state, but the current config disables it.")
            return

        if state_dict is None:
            raise ValueError(f"Checkpoint does not contain `{name}` state, but the current config requires it.")

        module.load_state_dict(state_dict)

    def _load_pose_state(self, pose: Pose, state_dict, name: str) -> None:
        if not isinstance(state_dict, dict):
            raise ValueError(f"Checkpoint is missing a valid `{name}` state.")

        required_keys = {"quat", "trans"}
        optional_keys = {
            "valid_count",
            "volume_index",
            "confidence",
            "volume_class_confidence",
        }
        state_keys = set(state_dict.keys())

        missing_keys = required_keys - state_keys
        unexpected_keys = state_keys - required_keys - optional_keys
        if missing_keys:
            raise ValueError(f"Checkpoint `{name}` state is missing keys: {sorted(missing_keys)}")
        if unexpected_keys:
            raise ValueError(f"Checkpoint `{name}` state has unexpected keys: {sorted(unexpected_keys)}")

        pose.load_state_dict(state_dict, strict=False)

    def _resolve_resume_checkpoint_path(self) -> tuple[str | None, bool]:
        if self.resume_checkpoint_path:
            checkpoint_path = os.path.abspath(self.resume_checkpoint_path)
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
            return checkpoint_path, True

        if not self.auto_resume:
            return None, False

        checkpoint_path = os.path.join(self.config.io.output_path, "checkpoints", "latest.pt")
        checkpoint_path = os.path.abspath(checkpoint_path)
        if os.path.isfile(checkpoint_path):
            return checkpoint_path, True
        return checkpoint_path, False

    def _sync_execution_flags_from_state(self) -> None:
        is_last_configured_epoch = (
            int(self.state.progress.epoch)
            >= int(self.config.abinitio.engine.num_epochs) - 1
        )
        self.state.abinitio.engine.is_final_epoch = (
            bool(self.state.abinitio.engine.is_final_epoch) or is_last_configured_epoch
        )
        self.state.schedule.full_backprojection = (
            bool(self.config.modules.volume.full_backprojection)
        )
        self.state.abinitio.engine.skip_external_reconstruct = bool(
            self.state.abinitio.engine.is_final_epoch
        )

    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        modules = ckpt.get("modules")
        if not isinstance(modules, dict):
            raise ValueError("Checkpoint is missing a valid `modules` section.")

        next_epoch = ckpt.get("next_epoch")
        next_schedule = ckpt.get("next_schedule")
        progress = ckpt.get("progress")
        if next_epoch is None:
            raise ValueError("Checkpoint is missing `next_epoch`.")
        if not isinstance(next_schedule, dict):
            raise ValueError("Checkpoint is missing a valid `next_schedule` section.")
        if not isinstance(progress, dict):
            raise ValueError("Checkpoint is missing a valid `progress` section.")
        if "volume" not in modules:
            raise ValueError("Checkpoint `modules` is missing `volume`.")
        if "pose" not in modules:
            raise ValueError("Checkpoint `modules` is missing `pose`.")
        if "noise" not in modules and self.noise is not None:
            raise ValueError("Checkpoint `modules` is missing `noise`.")

        self.volume.load_state_dict(modules["volume"])
        self._load_pose_state(self.pose, modules["pose"], "pose")
        self._load_optional_module_state(self.noise, modules.get("noise"), "noise")

        self.state.progress.epoch = int(next_epoch)
        self.state.progress.half = None
        self.state.progress.iter = 0
        self.state.abinitio.scheduler.num_checks_with_stable_side_length = int(
            progress.get("num_checks_with_stable_side_length", 0)
        )
        self.state.abinitio.scheduler.num_checks_with_stable_pose = int(
            progress.get("num_checks_with_stable_pose", 0)
        )
        self.state.abinitio.scheduler.num_checks_ready_to_stop = int(
            progress.get("num_checks_ready_to_stop", 0)
        )
        self.state.abinitio.scheduler.has_converged = bool(
            progress.get("has_converged", False)
        )

        required_schedule_keys = (
            "pose_search_scope",
            "pose_search_strategy",
            "healpix_order",
            "oversampling",
            "side_length",
            "trans_grid_extent",
            "pose_translation_center_mode",
            "use_pose_translation_as_center",
            "use_particle_mask",
            "particle_mask_extra_diameter_angstrom",
            "proj_cache_backend",
            "is_final_epoch",
        )
        for key in required_schedule_keys:
            if key not in next_schedule:
                raise ValueError(f"Checkpoint `next_schedule` is missing `{key}`.")
            if key == "is_final_epoch":
                self.state.abinitio.engine.is_final_epoch = bool(next_schedule[key])
            else:
                setattr(self.state.schedule, key, next_schedule[key])
        self.state.schedule.trans_grid_samples = int(
            next_schedule.get(
                "trans_grid_samples",
                self.state.schedule.trans_grid_samples,
            )
        )
        self.state.schedule.search_grad_mode = next_schedule.get(
            "search_grad_mode", "full"
        )
        self.state.abinitio.scheduler.initial_healpix_alignment_done = bool(
            next_schedule.get("initial_healpix_alignment_done", False)
        )
        self.state.abinitio.scheduler.healpix_terminal_reached = bool(
            next_schedule.get("healpix_terminal_reached", False)
        )
        self._sync_execution_flags_from_state()

        self._reset_confidence_metrics()
        self.state.abinitio.solver.activate_learning_rate_decay = False
        self.state.abinitio.metrics.volume_class_change_rate = 0.0
        self.state.abinitio.metrics.ema_volume_class_change_rate = None
        self._ema_loss = None
        self._ema_loss_change = None

        self.current_loss = None
        self.volume_real = None
        self._reset_schedule_check_state()
        self.probe = None
        self._effective_schedule_check_interval_iters = None
        self._schedule_check_epoch_total_iters = None
        self.pose_searcher.refresh()

    @torch.no_grad()
    def _resolve_init_lowpass(self) -> dict[str, float | int]:
        init_lowpass_angstrom = float(
            self.config.abinitio.engine.init_lowpass_angstrom
        )
        if init_lowpass_angstrom <= 0:
            raise ValueError(
                "abinitio.engine.init_lowpass_angstrom must be > 0, got "
                f"{self.config.abinitio.engine.init_lowpass_angstrom}"
            )

        init_radius = int(
            float(self.config.data.image_size) * float(self.config.data.angpix)
            / init_lowpass_angstrom
        )
        init_radius = max(1, min(int(self.config.data.image_size) // 2, init_radius))
        init_side_length = 2 * int(init_radius)
        return {
            "init_lowpass_angstrom": float(init_lowpass_angstrom),
            "init_radius": int(init_radius),
            "init_side_length": int(init_side_length),
        }

    @torch.no_grad()
    def build_solvent_mask(self) -> torch.Tensor | None:
        selector = str(self.config.abinitio.engine.solvent_mask).strip()
        mode = selector.lower()

        if os.path.isfile(selector):
            return load_mask_mrc(
                selector,
                side_length=int(self.config.data.image_size),
                angpix=float(self.config.data.angpix),
                device=self.device,
            )
        if mode == "none":
            return None
        if mode == "sphere":
            particle_diameter = self.config.data.particle_diameter
            if particle_diameter is None or float(particle_diameter) <= 0:
                raise ValueError(
                    "data.particle_diameter must be positive for a spherical solvent mask"
                )
            angpix = float(self.config.data.angpix)
            if angpix <= 0:
                raise ValueError(f"angpix must be positive, got {angpix}")
            return spherical_mask(
                int(self.config.data.image_size),
                int(self.config.data.image_size),
                int(self.config.data.image_size),
                radius=float(particle_diameter) / (2.0 * angpix),
                soft_edge_pixels=float(
                    self.config.abinitio.engine.solvent_mask_soft_edge_pixels
                ),
                device=self.device,
            ).to(dtype=torch.float32)
        if mode == "auto":
            raise NotImplementedError("automatic solvent masking is not implemented yet")
        raise FileNotFoundError(f"solvent mask does not exist: {selector}")

    @torch.no_grad()
    def project_volume_constraints(
        self,
        volume: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        volume_real = fourier_to_primal_3d(volume).real
        if self.solvent_mask is not None:
            constrained_real = (
                volume_real * self.solvent_mask.unsqueeze(0)
            ).clamp_min_(0.0)
        else:
            constrained_real = volume_real.clamp_min_(0.0)
        return primal_to_fourier_3d(constrained_real), constrained_real

    @torch.no_grad()
    def apply_volume_constraints(self) -> None:
        constrained_volume, constrained_real = self.project_volume_constraints(
            self.volume.volume.detach()
        )
        self.volume.copy_volume_(constrained_volume)
        self.volume_real = constrained_real

    def initialize(self):
        init_particles_per_volume = int(self.init_particles_per_volume)
        K = int(self.volume.num_volumes)
        num_particles = len(self.dataset)
        total_init_particles = K * init_particles_per_volume
        if num_particles < total_init_particles:
            raise ValueError(
                f"ab initio initialization requires at least {total_init_particles} particles "
                f"for K={K} and init_particles_per_volume={init_particles_per_volume}, got {num_particles}"
            )

        sample_gen = torch.Generator(device="cpu")
        sample_gen.manual_seed(int(self.config.reproduce.seed))
        sample_index = torch.randperm(num_particles, generator=sample_gen)[:total_init_particles]
        try:
            samples = [self.dataset[int(i)] for i in sample_index.tolist()]
            init_batch = data_collate_fn(samples).to(self.device, non_blocking=True)
        finally:
            # Avoid carrying parent-process mmap state into the first worker startup.
            self.dataset.close_cached_mrc_handles(worker_id=-1)
        if init_batch.ctf is None:
            raise ValueError("ab initio initialization requires per-image CTF")

        rot_gen = torch.Generator(device="cpu")
        rot_gen.manual_seed(int(self.config.reproduce.seed) + 1)
        quat = torch.randn((total_init_particles, 4), generator=rot_gen, dtype=torch.float32)
        quat = quat.to(self.device)
        quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1e-12)
        rotation = quaternion_to_matrix(quat)
        translation = torch.zeros((total_init_particles, 2), device=self.device, dtype=torch.float32)
        volume_index = torch.arange(K, device=self.device, dtype=torch.long).repeat_interleave(
            init_particles_per_volume
        )

        if self.state.schedule.pose_translation_center_mode == "auto":
            self.state.schedule.use_pose_translation_as_center = False
        self._sync_execution_flags_from_state()
        self.volume.requires_accum = True
        self.volume.zero_accum(set_to_none=False)
        self.volume.backproject(
            image=init_batch.image,
            ctf=init_batch.ctf,
            probability=None,
            image_index=None,
            volume_index=volume_index,
            rotation=rotation,
            translation=translation,
        )
        init_summary = self._resolve_init_lowpass()
        init_side_length = int(init_summary["init_side_length"])
        self.volume.update()
        init_volume = self.volume.volume.detach() * lowpass_mask(
            self.volume.volume.detach(),
            init_side_length,
            ndim=3,
        ).to(dtype=self.volume.volume.dtype)
        init_volume, self.volume_real = self.project_volume_constraints(init_volume)
        self.volume.copy_volume_(init_volume)
        init_volume_paths = self.save_init_volume()
        self.state.schedule.side_length = int(init_side_length)
        self._init_summary = {
            **init_summary,
            "num_init_particles": int(total_init_particles),
            "num_volumes": int(K),
            "init_resolution": float(
                self.scheduler._side_length_to_resolution(int(init_side_length))
            ),
            "init_volume_path": self._format_saved_map_paths(init_volume_paths),
        }
        self.volume.zero_accum(set_to_none=True)
        self.volume.requires_accum = False

        if self.noise is not None:
            if self.runtime.is_distributed:
                noise_dataloader, _ = build_distributed_dataloader(
                    self.dataset,
                    batch_size=self.config.data.batch_size,
                    shuffle=False,
                    num_workers=self.config.data.num_workers,
                    device=self.device,
                    seed=int(self.config.reproduce.seed),
                    drop_last=True,
                    worker_init_fn=single_thread_worker_init_fn,
                    multiprocessing_context="spawn",
                    device_mesh=self.device_mesh,
                )
            else:
                noise_dataloader = build_dataloader(
                    self.dataset,
                    batch_size=self.config.data.batch_size,
                    shuffle=False,
                    num_workers=self.config.data.num_workers,
                    pin_memory=(self.device.type == "cuda"),
                    drop_last=True,
                    worker_init_fn=single_thread_worker_init_fn,
                    multiprocessing_context="spawn",
                )
            self.noise.from_data([noise_dataloader])


    def preprocess(self, _batch: DataBatch):
        # norm correction
        del _batch
        return

    def _reset_schedule_check_state(self) -> None:
        self._last_schedule_check_volume = None

    def _reset_confidence_metrics(self) -> None:
        self._confidence_sum = 0.0
        self._confidence_count = 0
        self._volume_class_confidence_sum = 0.0
        self._volume_class_confidence_count = 0
        self.state.abinitio.metrics.avg_confidence = 0.0
        self.state.abinitio.metrics.avg_volume_class_confidence = 0.0

    def _update_confidence_metric_means(self) -> None:
        if self._confidence_count <= 0:
            self.state.abinitio.metrics.avg_confidence = 0.0
        else:
            self.state.abinitio.metrics.avg_confidence = (
                self._confidence_sum / float(self._confidence_count)
            )
        if self._volume_class_confidence_count <= 0:
            self.state.abinitio.metrics.avg_volume_class_confidence = 0.0
        else:
            self.state.abinitio.metrics.avg_volume_class_confidence = (
                self._volume_class_confidence_sum
                / float(self._volume_class_confidence_count)
            )

    def _update_latest_confidence_metrics(self) -> None:
        self._latest_avg_confidence = float(
            self.state.abinitio.metrics.avg_confidence
        )
        self._latest_avg_volume_class_confidence = float(
            self.state.abinitio.metrics.avg_volume_class_confidence
        )

    def _configured_schedule_check_interval_iters(self) -> int:
        schedule_check_interval_iters = int(
            self.config.abinitio.scheduler.schedule_check_interval_iters
        )
        if schedule_check_interval_iters <= 0:
            raise ValueError(
                "abinitio.scheduler.schedule_check_interval_iters must be > 0"
            )
        return schedule_check_interval_iters

    def _set_effective_schedule_check_interval_for_epoch(self) -> None:
        epoch_total_iters = int(len(self.dataloader))
        if epoch_total_iters <= 0:
            raise ValueError("dataloader must yield at least one batch per epoch")
        configured_interval = self._configured_schedule_check_interval_iters()
        self._schedule_check_epoch_total_iters = epoch_total_iters
        self._effective_schedule_check_interval_iters = min(
            configured_interval,
            max(1, epoch_total_iters // 2),
        )

    def _current_schedule_check_interval_iters(self) -> int:
        interval = self._effective_schedule_check_interval_iters
        if interval is None:
            raise RuntimeError(
                "effective schedule check interval is unavailable before epoch start"
            )
        return int(interval)

    def _is_schedule_check_iter(self) -> bool:
        return (
            self.state.progress.iter > 0
            and self.state.progress.iter
            % self._current_schedule_check_interval_iters()
            == 0
        )

    def _stash_schedule_probe(self, batch: DataBatch) -> None:
        self.probe = _ScheduleProbeBatch(
            image=batch.image.detach().cpu(),
            particle_index=batch.particle_index.detach().cpu(),
            ctf=(
                None
                if batch.ctf is None
                else batch.ctf.detach().cpu()
            ),
        )

    def _reduce_probe_pose_stats(
        self,
        valid_count: torch.Tensor,
        rot_update_rms_accum: torch.Tensor,
        trans_update_rms_accum: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = getattr(torch, "distributed", None)
        if dist is None or not dist.is_available() or not dist.is_initialized():
            return valid_count, rot_update_rms_accum, trans_update_rms_accum

        if self.device_mesh is not None:
            group = (
                self.device_mesh.get_group(0)
                if hasattr(self.device_mesh, "get_group")
                else self.device_mesh
            )
        else:
            group = dist.group.WORLD

        data_parallel_size = dist.get_world_size(group=group)
        if data_parallel_size <= 1:
            return valid_count, rot_update_rms_accum, trans_update_rms_accum

        dist.all_reduce(valid_count, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(rot_update_rms_accum, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(trans_update_rms_accum, op=dist.ReduceOp.SUM, group=group)
        return valid_count, rot_update_rms_accum, trans_update_rms_accum

    @torch.no_grad()
    def _compute_probe_pose_rms(self) -> tuple[float, float] | None:
        if self.probe is None:
            return None

        probe = self.probe
        self.pose.zero_accum()
        try:
            probe_image = probe.image.to(self.device, non_blocking=True)
            probe_particle_index = probe.particle_index.to(
                self.device,
                non_blocking=True,
            )
            probe_ctf = (
                None
                if probe.ctf is None
                else probe.ctf.to(self.device, non_blocking=True)
            )
            self.pose_searcher.search_no_grad(
                probe_image,
                particle_index=probe_particle_index,
                ctf=probe_ctf,
            )
            valid_count = self.pose.valid_count.sum().reshape(1)
            rot_update_rms_accum = self.pose.rot_update_rms_accum.clone()
            trans_update_rms_accum = self.pose.trans_update_rms_accum.clone()
            (
                valid_count,
                rot_update_rms_accum,
                trans_update_rms_accum,
            ) = self._reduce_probe_pose_stats(
                valid_count,
                rot_update_rms_accum,
                trans_update_rms_accum,
            )
            total_valid = int(valid_count.item())
            if total_valid <= 0:
                return None
            valid_count_float = valid_count.to(dtype=rot_update_rms_accum.dtype)
            rot_update_rms = torch.sqrt(
                rot_update_rms_accum / valid_count_float
            ).item()
            trans_update_rms = torch.sqrt(
                trans_update_rms_accum
                / valid_count.to(dtype=trans_update_rms_accum.dtype)
            ).item()
            return float(rot_update_rms), float(trans_update_rms)
        finally:
            self.pose.zero_accum()

    @torch.no_grad()
    def _current_schedule_check_volume(self) -> torch.Tensor:
        side_length = int(self.state.schedule.side_length)
        downsampled = downsample3d(self.volume.volume.detach(), side_length)
        center = side_length // 2
        mask = spherical_mask(
            side_length,
            side_length,
            side_length,
            center=(center, center, center),
            radius=side_length / 2,
            device=downsampled.device,
        )
        return downsampled.masked_fill(~mask, 0.0).clone()

    @torch.no_grad()
    def _update_schedule_check_metrics(self) -> None:
        current_volume = self._current_schedule_check_volume()
        self._last_schedule_check_volume = current_volume

    @torch.no_grad()
    def _snapshot_schedule_check_state(self) -> None:
        self._last_schedule_check_volume = self._current_schedule_check_volume()

    def _current_volume_real(self) -> torch.Tensor:
        volume_real = self.volume_real
        if volume_real is None:
            volume_real = self.volume.volume_real
        if volume_real is None:
            raise RuntimeError("working volume is unavailable")
        return volume_real

    def _save_volume_maps(
        self,
        *,
        output_dir: str,
        base_name: str,
    ) -> list[str] | None:
        if not is_rank0():
            return None
        os.makedirs(output_dir, exist_ok=True)
        volume_real = self._current_volume_real()
        if volume_real.ndim != 4:
            raise RuntimeError(
                f"expected volume_real to have shape (K,D,D,D), got {tuple(volume_real.shape)}"
            )

        root, ext = os.path.splitext(base_name)
        num_volumes = int(volume_real.shape[0])
        paths: list[str] = []
        for volume_idx in range(num_volumes):
            if num_volumes == 1:
                map_name = base_name
            else:
                map_name = f"{root}_class{volume_idx:03d}{ext}"
            map_path = os.path.join(output_dir, map_name)
            save_mrc(
                file_path=map_path,
                data=volume_real[volume_idx],
                voxel_size=self.config.data.angpix,
            )
            paths.append(map_path)
        return paths

    @staticmethod
    def _format_saved_map_paths(paths: list[str] | None) -> str | None:
        if paths is None:
            return None
        if len(paths) == 1:
            return paths[0]
        return ", ".join(paths)

    def _format_volume_occupancy(self) -> str | None:
        if self.pose is None:
            return None
        num_volumes = int(self.volume.num_volumes)
        if num_volumes <= 1:
            return None

        counts = torch.bincount(
            self.pose.volume_index.detach().to(dtype=torch.long),
            minlength=num_volumes,
        )
        total = int(counts.sum().item())
        if total <= 0:
            return None

        parts: list[str] = []
        for volume_idx, count in enumerate(counts.tolist()):
            frac = 100.0 * float(count) / float(total)
            parts.append(f"class{volume_idx:03d}={count} ({frac:.1f}%)")
        return " | ".join(parts)

    def _build_completion_log_lines(self, status: str) -> list[str]:
        lines = [
            status,
            (
                f"Resolution     : "
                f"{float(self.scheduler._side_length_to_resolution(int(self.state.schedule.side_length))):.2f} Angstrom"
            ),
        ]
        volume_occupancy = self._format_volume_occupancy()
        if volume_occupancy is not None:
            lines.append(f"Occupancy      : {volume_occupancy}")
        return lines

    def save_volume_snapshot(self) -> list[str] | None:
        epoch = int(self.state.progress.epoch)
        iteration = int(self.state.progress.iter)
        output_map_path = os.path.join(self.config.io.output_path, "maps", "snapshots")
        return self._save_volume_maps(
            output_dir=output_map_path,
            base_name=f"epoch_{epoch:03d}_iter_{iteration:06d}.mrc",
        )

    def save_init_volume(self) -> list[str] | None:
        output_map_path = os.path.join(self.config.io.output_path, "maps")
        return self._save_volume_maps(
            output_dir=output_map_path,
            base_name="init_volume.mrc",
        )

    def save_final_volume(self) -> list[str] | None:
        output_map_path = os.path.join(self.config.io.output_path, "maps")
        return self._save_volume_maps(
            output_dir=output_map_path,
            base_name="final_volume.mrc",
        )

    def _log_scheduler_event(
        self,
        logger: logging.Logger,
        *,
        epoch: int,
        before: ScheduleCheckSnapshot,
        after: ScheduleCheckSnapshot,
        before_is_final_epoch: bool,
        after_is_final_epoch: bool,
    ) -> None:
        if not is_rank0():
            return
        rotation_threshold = self.scheduler._rotation_stability_threshold(
            healpix_order=int(before.healpix_order)
        )
        side_length_rotation_threshold = (
            self.scheduler._rotation_stability_threshold_for_side_length(
                int(before.side_length)
            )
        )
        translation_threshold = (
            self.scheduler._translation_stability_threshold_for_side_length(
                int(before.side_length)
            )
        )
        healpix_rotation_stable = (
            before.ema_rot_update_rms is not None
            and float(before.ema_rot_update_rms) <= rotation_threshold
        )
        translation_stable = (
            before.ema_trans_update_rms is not None
            and float(before.ema_trans_update_rms) <= translation_threshold
        )
        side_length_rotation_stable = (
            before.ema_rot_update_rms is not None
            and float(before.ema_rot_update_rms)
            <= side_length_rotation_threshold
        )
        side_length_stable = side_length_rotation_stable and translation_stable
        pose_stable = side_length_stable
        before_resolution = float(before.side_length_resolution)
        rotation_threshold_deg = torch.rad2deg(
            torch.tensor(rotation_threshold)
        ).item()
        side_length_rotation_threshold_deg = torch.rad2deg(
            torch.tensor(side_length_rotation_threshold)
        ).item()
        ema_rot_update_rms_deg = (
            None
            if before.ema_rot_update_rms is None
            else torch.rad2deg(torch.tensor(before.ema_rot_update_rms)).item()
        )
        volume_occupancy = self._format_volume_occupancy()
        summary_lines = [
            f"Pose Search : L={before.side_length}, healpix={before.healpix_order}, trans_extent={before.trans_grid_extent:.2f}, trans_samples={before.trans_grid_samples}, trans_center={'pose' if before.use_pose_translation_as_center else 'zero'}",
            f"Stable      : side={side_length_stable}, side_rot={side_length_rotation_stable}, healpix_rot={healpix_rotation_stable}, trans={translation_stable}, pose={pose_stable}",
            f"Resolution  : {float(before_resolution):.2f} Angstrom",
            f"Confidence  : {100.0 * float(before.avg_confidence):.2f}%",
            f"Class Confidence : {100.0 * float(before.avg_volume_class_confidence):.2f}%",
            f"Class Change : {100.0 * float(before.ema_volume_class_change_rate if before.ema_volume_class_change_rate is not None else before.volume_class_change_rate):.2f}%",
            f"Pose RMS    : rot_side={'n/a' if ema_rot_update_rms_deg is None else f'{ema_rot_update_rms_deg:.2f}/{side_length_rotation_threshold_deg:.2f} deg'} | rot_healpix={'n/a' if ema_rot_update_rms_deg is None else f'{ema_rot_update_rms_deg:.2f}/{rotation_threshold_deg:.2f} deg'} | trans={'n/a' if before.ema_trans_update_rms is None else f'{float(before.ema_trans_update_rms):.2f}/{translation_threshold:.2f} px'}",
        ]
        if volume_occupancy is not None:
            summary_lines.append(f"Occupancy   : {volume_occupancy}")
        log_block(
            logger,
            title=f"Schedule Check Summary (Epoch {epoch} Iter {self.state.progress.iter})",
            lines=summary_lines,
        )

        next_lines = [
            f"Pose Search : L={after.side_length}, healpix={after.healpix_order}, trans_extent={after.trans_grid_extent:.2f}, trans_samples={after.trans_grid_samples}, trans_center={'pose' if after.use_pose_translation_as_center else 'zero'}",
        ]
        next_lines.extend(
            self._schedule_action_lines(
                before,
                after,
                before_is_final_epoch=before_is_final_epoch,
                after_is_final_epoch=after_is_final_epoch,
            )
        )
        if self.scheduler.local_entry_blocked():
            next_lines.append(
                "Action      : local entry blocked by class change"
            )
        next_lines.append(f"Final Epoch : {after_is_final_epoch}")
        log_block(
            logger,
            title=f"Next Schedule Configuration (Epoch {epoch} Iter {self.state.progress.iter})",
            lines=next_lines,
        )

    def _capture_schedule_check_snapshot(self) -> ScheduleCheckSnapshot:
        return ScheduleCheckSnapshot(
            pose_search_scope=str(self.state.schedule.pose_search_scope),
            pose_search_strategy=str(self.state.schedule.pose_search_strategy),
            side_length=int(self.state.schedule.side_length),
            healpix_order=int(self.state.schedule.healpix_order),
            trans_grid_extent=float(self.state.schedule.trans_grid_extent),
            trans_grid_samples=int(self.state.schedule.trans_grid_samples),
            use_pose_translation_as_center=bool(
                self.state.schedule.use_pose_translation_as_center
            ),
            side_length_resolution=float(
                self.scheduler._side_length_to_resolution(int(self.state.schedule.side_length))
            ),
            avg_confidence=float(self.state.abinitio.metrics.avg_confidence),
            avg_volume_class_confidence=float(self.state.abinitio.metrics.avg_volume_class_confidence),
            volume_class_change_rate=float(self.state.abinitio.metrics.volume_class_change_rate),
            ema_volume_class_change_rate=(
                None
                if self.state.abinitio.metrics.ema_volume_class_change_rate is None
                else float(self.state.abinitio.metrics.ema_volume_class_change_rate)
            ),
            ema_rot_update_rms=(
                None
                if self.state.abinitio.metrics.ema_rot_update_rms is None
                else float(self.state.abinitio.metrics.ema_rot_update_rms)
            ),
            ema_trans_update_rms=(
                None
                if self.state.abinitio.metrics.ema_trans_update_rms is None
                else float(self.state.abinitio.metrics.ema_trans_update_rms)
            ),
        )

    def _schedule_changed(
        self,
        before: ScheduleCheckSnapshot,
        after: ScheduleCheckSnapshot,
        *,
        before_is_final_epoch: bool,
        after_is_final_epoch: bool,
    ) -> bool:
        return (
            before.pose_search_scope != after.pose_search_scope
            or before.pose_search_strategy != after.pose_search_strategy
            or before.side_length != after.side_length
            or before.healpix_order != after.healpix_order
            or before.trans_grid_extent != after.trans_grid_extent
            or before.trans_grid_samples != after.trans_grid_samples
            or (
                before.use_pose_translation_as_center
                != after.use_pose_translation_as_center
            )
            or before_is_final_epoch != after_is_final_epoch
        )

    def _should_refresh_solver_for_schedule_change(
        self,
        before: ScheduleCheckSnapshot,
        after: ScheduleCheckSnapshot,
        *,
        before_is_final_epoch: bool,
        after_is_final_epoch: bool,
    ) -> bool:
        return (
            before.pose_search_scope != after.pose_search_scope
            or before.pose_search_strategy != after.pose_search_strategy
            or before.side_length != after.side_length
            or before.healpix_order != after.healpix_order
            or before.trans_grid_extent != after.trans_grid_extent
            or before.trans_grid_samples != after.trans_grid_samples
            or (
                before.use_pose_translation_as_center
                != after.use_pose_translation_as_center
            )
            or (not before_is_final_epoch and after_is_final_epoch)
        )

    def _should_save_volume_snapshot_for_schedule_change(
        self,
        before: ScheduleCheckSnapshot,
        after: ScheduleCheckSnapshot,
        *,
        before_is_final_epoch: bool,
        after_is_final_epoch: bool,
    ) -> bool:
        return (
            len(
                self._schedule_action_lines(
                    before,
                    after,
                    before_is_final_epoch=before_is_final_epoch,
                    after_is_final_epoch=after_is_final_epoch,
                )
            )
            > 0
        )

    def _schedule_action_lines(
        self,
        before: ScheduleCheckSnapshot,
        after: ScheduleCheckSnapshot,
        *,
        before_is_final_epoch: bool,
        after_is_final_epoch: bool,
    ) -> list[str]:
        actions: list[str] = []
        if before.side_length != after.side_length:
            actions.append("Action      : increased side_length")
        if before.healpix_order != after.healpix_order:
            actions.append("Action      : increased healpix_order")
        if not before_is_final_epoch and after_is_final_epoch:
            actions.append("Action      : marked current epoch as final")
        return actions

    @torch.no_grad()
    def _accumulate_confidence_metrics_from_pose(self) -> None:
        if self.state.schedule.pose_search_criterion != "posterior":
            return
        valid_count = int(self.pose.valid_count.sum().item())
        if valid_count <= 0:
            return
        self._confidence_sum += float(self.pose.avg_confidence.item()) * valid_count
        self._confidence_count += valid_count
        self._volume_class_confidence_sum += (
            float(self.pose.avg_volume_class_confidence.item()) * valid_count
        )
        self._volume_class_confidence_count += valid_count
        self._update_confidence_metric_means()

    @torch.no_grad()
    def _evaluate_batch_metrics(self) -> None:
        # Track EMA as a detached scalar only; it must never enter autograd.
        if self.current_loss is None:
            raise RuntimeError("current_loss is unavailable for evaluation")
        self._accumulate_confidence_metrics_from_pose()
        self._update_latest_confidence_metrics()
        self.state.abinitio.metrics.side_length_resolution = self.scheduler._side_length_to_resolution(
            int(self.state.schedule.side_length)
        )
        batch_loss = float(self.current_loss)
        loss_ema_decay = float(self.config.abinitio.engine.loss_ema_decay)
        if not (0.0 <= loss_ema_decay < 1.0):
            raise ValueError(
                "abinitio.engine.loss_ema_decay must be in [0, 1), got "
                f"{self.config.abinitio.engine.loss_ema_decay}"
            )
        prev_ema_loss = self._ema_loss
        if prev_ema_loss is None:
            ema_loss = float(batch_loss)
        else:
            ema_loss = (
                loss_ema_decay * float(prev_ema_loss)
                + (1.0 - loss_ema_decay) * float(batch_loss)
            )
        self._ema_loss = float(ema_loss)
        self._ema_loss_change = (
            None
            if prev_ema_loss is None
            else float(ema_loss) - float(prev_ema_loss)
        )
        volume_class_change_rate = float(self.pose.volume_class_change_rate.item())
        self.state.abinitio.metrics.volume_class_change_rate = volume_class_change_rate
        prev_ema_volume_class_change_rate = self.state.abinitio.metrics.ema_volume_class_change_rate
        pose_rms_ema_decay = float(self.config.abinitio.engine.pose_rms_ema_decay)
        if not (0.0 <= pose_rms_ema_decay < 1.0):
            raise ValueError(
                "abinitio.engine.pose_rms_ema_decay must be in [0, 1), got "
                f"{self.config.abinitio.engine.pose_rms_ema_decay}"
            )
        if prev_ema_volume_class_change_rate is None:
            self.state.abinitio.metrics.ema_volume_class_change_rate = volume_class_change_rate
        else:
            self.state.abinitio.metrics.ema_volume_class_change_rate = (
                pose_rms_ema_decay * float(prev_ema_volume_class_change_rate)
                + (1.0 - pose_rms_ema_decay) * volume_class_change_rate
            )

    @torch.no_grad()
    def _update_pose_rms_ema(
        self,
        rot_update_rms: float,
        trans_update_rms: float,
    ) -> None:
        pose_rms_ema_decay = float(self.config.abinitio.engine.pose_rms_ema_decay)
        if not (0.0 <= pose_rms_ema_decay < 1.0):
            raise ValueError(
                "abinitio.engine.pose_rms_ema_decay must be in [0, 1), got "
                f"{self.config.abinitio.engine.pose_rms_ema_decay}"
            )
        prev_ema_rot_update_rms = self.state.abinitio.metrics.ema_rot_update_rms
        prev_ema_trans_update_rms = self.state.abinitio.metrics.ema_trans_update_rms
        if prev_ema_rot_update_rms is None:
            self.state.abinitio.metrics.ema_rot_update_rms = float(rot_update_rms)
        else:
            self.state.abinitio.metrics.ema_rot_update_rms = (
                pose_rms_ema_decay * float(prev_ema_rot_update_rms)
                + (1.0 - pose_rms_ema_decay) * float(rot_update_rms)
            )
        if prev_ema_trans_update_rms is None:
            self.state.abinitio.metrics.ema_trans_update_rms = float(trans_update_rms)
        else:
            self.state.abinitio.metrics.ema_trans_update_rms = (
                pose_rms_ema_decay * float(prev_ema_trans_update_rms)
                + (1.0 - pose_rms_ema_decay) * float(trans_update_rms)
            )

    @torch.no_grad()
    def _evaluate_schedule_check_metrics(self) -> None:
        probe_pose_rms = self._compute_probe_pose_rms()
        if probe_pose_rms is not None:
            rot_update_rms, trans_update_rms = probe_pose_rms
            self.state.abinitio.metrics.rot_update_rms = float(rot_update_rms)
            self.state.abinitio.metrics.trans_update_rms = float(trans_update_rms)
            self._update_pose_rms_ema(rot_update_rms, trans_update_rms)
        self._update_schedule_check_metrics()

    @torch.no_grad()
    def evaluate(self) -> None:
        self._evaluate_batch_metrics()
        if self._is_schedule_check_iter():
            self._evaluate_schedule_check_metrics()

    def save_checkpoint(self):
        if not is_rank0():
            return None

        epoch = self.state.progress.epoch
        output_checkpoints_root = os.path.join(self.config.io.output_path, "checkpoints")
        output_checkpoint_path = os.path.join(self.config.io.output_path, "checkpoints", f"epoch_{epoch:03d}")
        os.makedirs(output_checkpoint_path, exist_ok=True)

        ckpt = {
            "modules": {
                "volume": self.volume.state_dict(),
                "pose": self.pose.state_dict(),
                "noise": self.noise.state_dict() if self.noise else None,
            },
            "progress": {
                "num_checks_with_stable_side_length": self.state.abinitio.scheduler.num_checks_with_stable_side_length,
                "num_checks_with_stable_pose": self.state.abinitio.scheduler.num_checks_with_stable_pose,
                "num_checks_ready_to_stop": self.state.abinitio.scheduler.num_checks_ready_to_stop,
                "has_converged": self.state.abinitio.scheduler.has_converged,
            },

            "next_epoch": epoch + 1,
            "next_schedule":{
                "pose_search_scope": self.state.schedule.pose_search_scope,
                "pose_search_strategy": self.state.schedule.pose_search_strategy,
                "healpix_order": self.state.schedule.healpix_order,
                "oversampling": self.state.schedule.oversampling,
                "side_length": self.state.schedule.side_length,
                "trans_grid_extent": self.state.schedule.trans_grid_extent,
                "trans_grid_samples": self.state.schedule.trans_grid_samples,
                "search_grad_mode": self.state.schedule.search_grad_mode,
                "pose_translation_center_mode": self.state.schedule.pose_translation_center_mode,
                "use_pose_translation_as_center": self.state.schedule.use_pose_translation_as_center,
                "use_particle_mask": self.state.schedule.use_particle_mask,
                "particle_mask_extra_diameter_angstrom": self.state.schedule.particle_mask_extra_diameter_angstrom,
                "proj_cache_backend": self.state.schedule.proj_cache_backend,
                "initial_healpix_alignment_done": self.state.abinitio.scheduler.initial_healpix_alignment_done,
                "healpix_terminal_reached": self.state.abinitio.scheduler.healpix_terminal_reached,
                "is_final_epoch": self.state.abinitio.engine.is_final_epoch,
            }
        }

        epoch_checkpoint_path = os.path.join(output_checkpoint_path, f"epoch_{epoch:03d}.pt")
        latest_checkpoint_path = os.path.join(output_checkpoints_root, "latest.pt")
        latest_checkpoint_tmp_path = f"{latest_checkpoint_path}.tmp"

        torch.save(ckpt, epoch_checkpoint_path)
        torch.save(ckpt, latest_checkpoint_tmp_path)
        os.replace(latest_checkpoint_tmp_path, latest_checkpoint_path)

        checkpoint_paths = {
            "epoch_checkpoint_path": epoch_checkpoint_path,
            "latest_checkpoint_path": latest_checkpoint_path,
        }

        return checkpoint_paths

    def run(self):
        # set seed
        set_seed(self.config.reproduce.seed, self.config.reproduce.deterministic)

        logger = setup_logging(
            self.config.logging.log_dir,
            filename_prefix=self.config.logging.log_prefix,
            level=self.config.logging.level,
        )
        logger.info("Program started")
        log_config(logger, self.config)

        try:
            if self.resume_checkpoint_path:
                logger.info("Run mode | explicit resume")
            elif self.auto_resume:
                logger.info("Run mode | auto-resume")
            else:
                logger.info("Run mode | fresh start")

            resume_checkpoint_path, should_resume = self._resolve_resume_checkpoint_path()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            init_wall_start = time.perf_counter()
            if should_resume:
                logger.info("Resume started | checkpoint=%s", resume_checkpoint_path)
                self.resume_from_checkpoint(resume_checkpoint_path)
                start_epoch = int(self.state.progress.epoch)
            elif self.auto_resume:
                logger.info(
                    "Auto-resume checkpoint not found | checkpoint=%s | starting fresh",
                    resume_checkpoint_path,
                )
                logger.info("Initialization started")
                self.initialize()
                start_epoch = 0
            else:
                logger.info("Initialization started")
                self.initialize()
                start_epoch = 0

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            init_wall = time.perf_counter() - init_wall_start
            if should_resume:
                logger.info("Resume finished | next_epoch=%d | time=%.3fs", start_epoch, init_wall)
            else:
                logger.info("Initialization finished | time=%.3fs", init_wall)
                if is_rank0() and self._init_summary is not None:
                    log_block(
                        logger,
                        title="Ab Initio Initialization Summary",
                        lines=[
                            f"Pose Search    : L={int(self._init_summary['init_side_length'])}",
                            f"Resolution     : {float(self._init_summary['init_resolution']):.2f} Angstrom",
                            f"Low-pass       : {float(self._init_summary['init_lowpass_angstrom']):.2f} Angstrom (radius={int(self._init_summary['init_radius'])}, L={int(self._init_summary['init_side_length'])})",
                            f"Initialization : {int(self._init_summary['num_init_particles'])} particles across {int(self._init_summary['num_volumes'])} volume(s)",
                            f"Init Volume    : {self._init_summary['init_volume_path']}",
                        ],
                    )

            if should_resume and bool(self.state.abinitio.scheduler.has_converged):
                logger.info(
                    "Resume checkpoint already represents a completed ab initio run | next_epoch=%d",
                    start_epoch,
                )
                self.volume_real = self.volume.volume_real
                if is_rank0():
                    log_block(
                        logger,
                        title="Ab Initio Reconstruction Completed",
                        lines=self._build_completion_log_lines(
                            "Checkpoint already captured the final completed epoch"
                        ),
                    )
                final_map_paths = self.save_final_volume()
                final_map_path = self._format_saved_map_paths(final_map_paths)
                if final_map_path is not None:
                    logger.info("Final volume saved | path=%s", final_map_path)
                return

            if start_epoch >= int(self.config.abinitio.engine.num_epochs):
                logger.warning(
                    "No epochs to run after resume: start_epoch=%d, num_epochs=%d",
                    start_epoch,
                    int(self.config.abinitio.engine.num_epochs),
                )
                return

            self.state.progress.half = None
            self.state.progress.iter = 0
            completed_via_convergence_final_epoch = False
            self.solver.refresh()
            # loop
            for epoch in range(start_epoch, self.config.abinitio.engine.num_epochs):
                self.state.progress.epoch = epoch
                self._sync_execution_flags_from_state()
                self._set_effective_schedule_check_interval_for_epoch()
                self._reset_confidence_metrics()

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                epoch_wall_start = time.perf_counter()

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                loop_wall_start = time.perf_counter()

                if self.sampler is not None:
                    self.sampler.set_epoch(epoch)
                epoch_total_iters = int(self._schedule_check_epoch_total_iters)
                configured_interval = self._configured_schedule_check_interval_iters()
                effective_interval = self._current_schedule_check_interval_iters()
                if effective_interval == configured_interval:
                    logger.info(
                        "Epoch %d started | epoch_iters=%d | schedule_check=%d",
                        epoch,
                        epoch_total_iters,
                        effective_interval,
                    )
                else:
                    logger.info(
                        "Epoch %d started | epoch_iters=%d | schedule_check=%d (cfg=%d)",
                        epoch,
                        epoch_total_iters,
                        effective_interval,
                        configured_interval,
                    )

                log_state(
                    logger,
                    self.state,
                    title=f"Epoch {epoch} State",
                    command="abinitio",
                )

                dl = self.dataloader
                if is_rank0() and tqdm is not None:
                    dl = tqdm(dl, desc=f"Epoch {epoch}", dynamic_ncols=True)
                    dl.set_postfix(
                        OrderedDict(
                            [
                                ("loss", "n/a"),
                                ("conf", "n/a"),
                                ("cls_chg", "n/a"),
                                ("rot_rms", "n/a"),
                                ("trans_rms", "n/a"),
                            ]
                        ),
                        refresh=False,
                    )

                self.current_loss = None
                stopped_mid_epoch_on_convergence = False
                for batch in dl:
                    self.solver.zero_accum()
                    batch = batch.to(self.device, non_blocking=True)
                    result = self.solver.infer(batch)
                    self.current_loss = float(result.loss.detach().item())
                    self.solver.accumulate(result)
                    self.solver.update()
                    self.apply_volume_constraints()
                    self.state.progress.iter += 1
                    self.evaluate()
                    is_schedule_check_iter = self._is_schedule_check_iter()
                    if (
                        not bool(self.state.abinitio.engine.is_final_epoch)
                        and is_schedule_check_iter
                    ):
                        before_is_final_epoch = bool(
                            self.state.abinitio.engine.is_final_epoch
                        )
                        snapshot_before_step = self._capture_schedule_check_snapshot()
                        self.scheduler.step()
                        after_is_final_epoch = bool(
                            self.state.abinitio.engine.is_final_epoch
                        )
                        snapshot_after_step = self._capture_schedule_check_snapshot()
                        self._reset_confidence_metrics()
                        schedule_changed = self._schedule_changed(
                            snapshot_before_step,
                            snapshot_after_step,
                            before_is_final_epoch=before_is_final_epoch,
                            after_is_final_epoch=after_is_final_epoch,
                        )
                        if self._should_refresh_solver_for_schedule_change(
                            snapshot_before_step,
                            snapshot_after_step,
                            before_is_final_epoch=before_is_final_epoch,
                            after_is_final_epoch=after_is_final_epoch,
                        ):
                            self.solver.refresh()
                        if schedule_changed:
                            if self._should_save_volume_snapshot_for_schedule_change(
                                snapshot_before_step,
                                snapshot_after_step,
                                before_is_final_epoch=before_is_final_epoch,
                                after_is_final_epoch=after_is_final_epoch,
                            ):
                                snapshot_paths = self.save_volume_snapshot()
                                snapshot_path = self._format_saved_map_paths(
                                    snapshot_paths
                                )
                                if snapshot_path is not None:
                                    logger.info(
                                        "Volume snapshot saved | epoch=%d | iter=%d | path=%s",
                                        epoch,
                                        int(self.state.progress.iter),
                                        snapshot_path,
                                    )
                            self._log_scheduler_event(
                                logger,
                                epoch=epoch,
                                before=snapshot_before_step,
                                after=snapshot_after_step,
                                before_is_final_epoch=before_is_final_epoch,
                                after_is_final_epoch=after_is_final_epoch,
                            )
                        elif self.scheduler.local_entry_blocked():
                            self._log_scheduler_event(
                                logger,
                                epoch=epoch,
                                before=snapshot_before_step,
                                after=snapshot_after_step,
                                before_is_final_epoch=before_is_final_epoch,
                                after_is_final_epoch=after_is_final_epoch,
                            )
                        self._snapshot_schedule_check_state()
                        if bool(self.state.abinitio.scheduler.has_converged):
                            if epoch == 0:
                                logger.info(
                                    "Ab initio converged at epoch=0 | iter=%d | "
                                    "finishing the first epoch before stopping",
                                    int(self.state.progress.iter),
                                )
                            else:
                                stopped_mid_epoch_on_convergence = True
                                logger.info(
                                    "Ab initio converged at epoch=%d | iter=%d | "
                                    "stopping immediately",
                                    epoch,
                                    int(self.state.progress.iter),
                                )
                                break
                    if is_schedule_check_iter:
                        self._stash_schedule_probe(batch)
                    if tqdm is not None and hasattr(dl, "set_postfix"):
                        pose_rot = self.state.abinitio.metrics.ema_rot_update_rms
                        pose_rot_deg = (
                            "n/a"
                            if pose_rot is None
                            else f"{torch.rad2deg(torch.tensor(pose_rot)).item():.2f}"
                        )
                        pose_trans = self.state.abinitio.metrics.ema_trans_update_rms
                        volume_class_change = self.state.abinitio.metrics.ema_volume_class_change_rate
                        if volume_class_change is None:
                            volume_class_change = self.state.abinitio.metrics.volume_class_change_rate
                        dl.set_postfix(
                            OrderedDict(
                                [
                                    ("loss", f"{float(self._ema_loss):.3e}"),
                                    (
                                        "conf",
                                        f"{100.0 * float(self._latest_avg_confidence):.2f}%",
                                    ),
                                    (
                                        "cls_chg",
                                        f"{100.0 * float(volume_class_change):.2f}%",
                                    ),
                                    ("rot_rms", pose_rot_deg),
                                    (
                                        "trans_rms",
                                        (
                                            "n/a"
                                            if pose_trans is None
                                            else f"{float(pose_trans):.2f}"
                                        ),
                                    ),
                                ]
                            ),
                            refresh=False,
                        )

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                loop_wall = time.perf_counter() - loop_wall_start
                logger.info(
                    "Epoch %d loop finished | time=%.3fs",
                    epoch,
                    loop_wall,
                )
                if self.state.progress.iter == 0:
                    raise RuntimeError("ab initio epoch produced no loss values")
                self.volume_real = self.volume.volume_real

                # epoch summary
                if is_rank0():
                    epoch_rot_side_threshold_deg = torch.rad2deg(
                        torch.tensor(
                            self.scheduler._rotation_stability_threshold_for_side_length(
                                int(self.state.schedule.side_length)
                            )
                        )
                    ).item()
                    epoch_rot_healpix_threshold_deg = torch.rad2deg(
                        torch.tensor(self.scheduler._rotation_stability_threshold())
                    ).item()
                    epoch_rot_rms_deg = (
                        "n/a"
                        if self.state.abinitio.metrics.ema_rot_update_rms is None
                        else f"{torch.rad2deg(torch.tensor(self.state.abinitio.metrics.ema_rot_update_rms)).item():.2f}"
                    )
                    epoch_trans_rms = (
                        "n/a"
                        if self.state.abinitio.metrics.ema_trans_update_rms is None
                        else f"{float(self.state.abinitio.metrics.ema_trans_update_rms):.2f}"
                    )
                    epoch_trans_threshold = (
                        f"{self.scheduler._translation_stability_threshold_for_side_length(int(self.state.schedule.side_length)):.2f}"
                    )
                    volume_occupancy = self._format_volume_occupancy()
                    log_block(
                        logger,
                        title=f"Epoch {epoch} Summary",
                        lines=[
                            f"Pose Search : L={self.state.schedule.side_length}, healpix={self.state.schedule.healpix_order}, trans_extent={self.state.schedule.trans_grid_extent:.2f}, criterion={self.state.schedule.pose_search_criterion}",
                            f"Resolution  : {float(self.state.abinitio.metrics.side_length_resolution):.2f} Angstrom",
                            f"EMA Loss    : {float(self._ema_loss):.6e}",
                            f"Confidence  : {100.0 * float(self._latest_avg_confidence):.2f}%",
                            f"Class Confidence : {100.0 * float(self._latest_avg_volume_class_confidence):.2f}%",
                            f"Class Change : {100.0 * float(self.state.abinitio.metrics.ema_volume_class_change_rate if self.state.abinitio.metrics.ema_volume_class_change_rate is not None else self.state.abinitio.metrics.volume_class_change_rate):.2f}%",
                            *(
                                []
                                if volume_occupancy is None
                                else [f"Occupancy   : {volume_occupancy}"]
                            ),
                            f"Pose RMS    : rot_side={epoch_rot_rms_deg}/{epoch_rot_side_threshold_deg:.2f} deg | rot_healpix={epoch_rot_rms_deg}/{epoch_rot_healpix_threshold_deg:.2f} deg | trans={epoch_trans_rms}/{epoch_trans_threshold} px",
                            f"Final Epoch : {self.state.abinitio.engine.is_final_epoch}",
                            f"Stop After  : {bool(self.state.abinitio.scheduler.has_converged)}",
                        ],
                    )

                # save results
                checkpoint_paths = self.save_checkpoint()
                if checkpoint_paths is not None:
                    logger.info(
                        "Checkpoint updated | epoch=%d | epoch_ckpt=%s | latest=%s",
                        epoch,
                        checkpoint_paths["epoch_checkpoint_path"],
                        checkpoint_paths["latest_checkpoint_path"],
                    )

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                epoch_wall = time.perf_counter() - epoch_wall_start
                logger.info(
                    "Epoch %d finished | total=%.3fs | loop=%.3fs",
                    epoch,
                    epoch_wall,
                    loop_wall,
                )

                if bool(self.state.abinitio.scheduler.has_converged):
                    completed_via_convergence_final_epoch = True
                    if is_rank0():
                        log_block(
                            logger,
                            title="Ab Initio Reconstruction Completed",
                            lines=self._build_completion_log_lines(
                                "Completed immediately after convergence check"
                                if stopped_mid_epoch_on_convergence
                                else "Completed after finishing the converged epoch"
                            ),
                        )
                    break

            if not completed_via_convergence_final_epoch and is_rank0():
                log_block(
                    logger,
                    title="Ab Initio Reconstruction Completed",
                    lines=self._build_completion_log_lines(
                        f"Completed all {int(self.config.abinitio.engine.num_epochs)} epochs"
                    ),
                )
            final_map_paths = self.save_final_volume()
            final_map_path = self._format_saved_map_paths(final_map_paths)
            if final_map_path is not None:
                logger.info("Final volume saved | path=%s", final_map_path)

        except Exception:
            logger.exception("Ab initio reconstruction failed")
            raise