from __future__ import annotations

import logging
import os

import mrcfile
import torch

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from cryoseed.config import MainConfig
from cryoseed.cryoem.mask import load_mask_mrc, lowpass_mask, spherical_mask
from cryoseed.data import (
    ParticleDataset,
    build_dataloader,
    build_distributed_dataloader,
    save_mrc,
    single_thread_worker_init_fn,
)
from cryoseed.fft.fft_torch import fourier_to_primal_3d, primal_to_fourier_3d
from cryoseed.modules.pose import Pose
from cryoseed.modules.statistics import NoiseVariance, PriorVariance
from cryoseed.modules.volume import VoxelGrid
from cryoseed.ops.radial import radial_average
from cryoseed.optim import EMSolver
from cryoseed.optim.pose import PoseSearcher
from cryoseed.optim.scheduler import HeteroRefineScheduler
from cryoseed.runtime.distributed import RuntimeContext, is_rank0
from cryoseed.state import OptimState, parse_pose_search_criterion
from cryoseed.utils.logging import log_block, log_config, log_state, setup_logging
from cryoseed.utils.reproducibility import set_seed


LOGGER = logging.getLogger(__name__)


class HeteroRefineEngine(torch.nn.Module):
    def __init__(
        self,
        config: MainConfig,
        runtime: RuntimeContext,
        resume_checkpoint_path: str | None = None,
        auto_resume: bool = False,
    ):
        super().__init__()
        self.config = config.validate_for_command("heterorefine")
        self.runtime = runtime
        self.device = runtime.device
        self.device_mesh = runtime.device_mesh
        self.resume_checkpoint_path = resume_checkpoint_path
        self.auto_resume = bool(auto_resume)

        if int(config.modules.volume.num_volumes) <= 1:
            raise ValueError(
                "heterorefine requires modules.volume.num_volumes > 1"
            )

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
            self.dataloader, self.sampler = build_distributed_dataloader(
                dataset,
                batch_size=config.data.batch_size,
                shuffle=True,
                num_workers=config.data.num_workers,
                device=self.device,
                seed=int(config.reproduce.seed),
                worker_init_fn=single_thread_worker_init_fn,
                multiprocessing_context="spawn",
                device_mesh=self.device_mesh,
            )
        else:
            self.dataloader = build_dataloader(
                dataset,
                batch_size=config.data.batch_size,
                shuffle=True,
                num_workers=config.data.num_workers,
                pin_memory=(self.device.type == "cuda"),
                worker_init_fn=single_thread_worker_init_fn,
                multiprocessing_context="spawn",
            )
            self.sampler = None

        self.volume = VoxelGrid.from_config(
            config, device=self.device, device_mesh=self.device_mesh
        )
        self.noise = NoiseVariance.from_config(
            config, device=self.device, device_mesh=self.device_mesh
        )
        self.prior = PriorVariance.from_config(config, device=self.device)
        if self.prior is None:
            raise ValueError("heterorefine requires modules.statistics.prior.enabled=true")
        self.pose = Pose.from_config(
            config, device=self.device, device_mesh=self.device_mesh
        )
        self.state = OptimState.from_config(config, command="heterorefine")
        self.pose_searcher = PoseSearcher(
            state=self.state,
            volume=self.volume,
            noise=self.noise,
            pose=self.pose,
            config=config,
            device=self.device,
            device_mesh=self.device_mesh,
        )
        self.solver = EMSolver(
            state=self.state, pose_searcher=self.pose_searcher, prior=self.prior
        )
        self.scheduler = HeteroRefineScheduler.from_config(self.state, config)
        self.solvent_mask = self._build_solvent_mask()
        self.volume_real: torch.Tensor | None = None

    def _build_solvent_mask(self) -> torch.Tensor | None:
        selector = str(self.config.heterorefine.engine.solvent_mask).strip()
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
            diameter = self.config.data.particle_diameter
            if diameter is None or float(diameter) <= 0:
                raise ValueError("data.particle_diameter is required for sphere mask")
            return spherical_mask(
                int(self.config.data.image_size),
                int(self.config.data.image_size),
                int(self.config.data.image_size),
                radius=float(diameter) / (2.0 * float(self.config.data.angpix)),
                soft_edge_pixels=float(
                    self.config.heterorefine.engine.solvent_mask_soft_edge_pixels
                ),
                device=self.device,
            ).to(dtype=torch.float32)
        if mode == "auto":
            raise NotImplementedError("automatic solvent masking is not implemented")
        raise FileNotFoundError(f"solvent mask does not exist: {selector}")

    @torch.no_grad()
    def _apply_volume_constraints(self) -> None:
        volume_real = fourier_to_primal_3d(self.volume.volume).real
        if self.solvent_mask is not None:
            volume_real = volume_real * self.solvent_mask.unsqueeze(0)
        self.volume_real = volume_real
        self.volume.copy_volume_(primal_to_fourier_3d(volume_real))

    @torch.no_grad()
    def initialize(self) -> None:
        with mrcfile.open(self.config.io.ref_volume_path, permissive=True) as mrc:
            reference = torch.as_tensor(mrc.data, device=self.device).to(torch.float32)
        D = int(self.config.data.image_size)
        if tuple(reference.shape) != (D, D, D):
            raise ValueError(
                f"reference volume must have shape {(D, D, D)}, got {tuple(reference.shape)}"
            )
        init_radius = max(
            1,
            min(
                D // 2,
                int(
                    float(D)
                    * float(self.config.data.angpix)
                    / float(self.config.heterorefine.engine.init_lowpass_angstrom)
                ),
            ),
        )
        init_side_length = 2 * init_radius
        reference_fourier = primal_to_fourier_3d(reference.unsqueeze(0))
        reference_fourier = reference_fourier * lowpass_mask(
            reference_fourier, init_side_length, ndim=3
        ).to(reference_fourier.dtype)
        initial = reference_fourier.expand(
            int(self.volume.num_volumes), -1, -1, -1
        ).clone()
        self.volume.load_volume(initial)
        self.state.schedule.side_length = init_side_length
        self._apply_volume_constraints()
        self._update_prior()
        if self.noise is not None:
            self.noise.from_data([self.dataloader])
        self.solver.refresh()

    def _is_ncc_epoch(self, epoch: int) -> bool:
        return bool(self.config.heterorefine.engine.first_epoch_ncc) and epoch == 0

    def _is_bootstrap_epoch(self, epoch: int) -> bool:
        return epoch == (1 if self.config.heterorefine.engine.first_epoch_ncc else 0)

    def _fixed_volume_index(
        self, epoch: int, particle_index: torch.LongTensor
    ) -> torch.LongTensor | None:
        if self._is_ncc_epoch(epoch):
            return torch.zeros_like(particle_index, dtype=torch.long)
        if self._is_bootstrap_epoch(epoch):
            return particle_index.to(dtype=torch.long).remainder(
                int(self.volume.num_volumes)
            )
        return None

    @torch.no_grad()
    def _copy_ncc_volume_to_all_classes(self) -> None:
        shared = self.volume.volume[0:1].expand_as(self.volume.volume).clone()
        self.volume.copy_volume_(shared)
        self._apply_volume_constraints()

    @torch.no_grad()
    def _update_prior(self) -> None:
        self.prior.init_lowpass_cutoff = int(self.state.schedule.side_length)
        self.prior.from_volume(self.volume.volume.detach())

    @torch.no_grad()
    def _evaluate_dvp(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight_3d = self.volume.accumulated_weight
        weight_1d = radial_average(
            weight_3d,
            int(self.config.data.image_size) // 2,
            ndim=3,
            use_cache=True,
        )
        dvp = self.prior.variance * weight_1d
        return dvp, weight_1d

    @torch.no_grad()
    def _update_metrics(self) -> None:
        metrics = self.state.heterorefine.metrics
        metrics.avg_confidence = float(self.pose.avg_confidence.item())
        metrics.avg_volume_class_confidence = float(
            self.pose.avg_volume_class_confidence.item()
        )
        metrics.volume_class_change_rate = float(
            self.pose.volume_class_change_rate.item()
        )
        metrics.rot_update_rms = float(self.pose.rot_update_rms.item())
        metrics.trans_update_rms = float(self.pose.trans_update_rms.item())
        counts = torch.bincount(
            self.pose.volume_index, minlength=int(self.volume.num_volumes)
        )
        metrics.volume_occupancy = counts.detach().cpu().tolist()

    def _save_maps(self, epoch: int, *, final: bool = False) -> list[str]:
        if not is_rank0():
            return []
        if self.volume_real is None:
            self.volume_real = self.volume.volume_real
        root = os.path.join(self.config.io.output_path, "maps")
        os.makedirs(root, exist_ok=True)
        prefix = "final" if final else f"epoch_{epoch:03d}"
        paths: list[str] = []
        for volume_index, volume_real in enumerate(self.volume_real):
            path = os.path.join(root, f"{prefix}_class{volume_index:03d}.mrc")
            save_mrc(
                file_path=path,
                data=volume_real,
                voxel_size=float(self.config.data.angpix),
            )
            paths.append(path)
        return paths

    def _checkpoint_path(self) -> str:
        return os.path.join(self.config.io.output_path, "checkpoints", "latest.pt")

    def save_checkpoint(self, epoch: int) -> None:
        if not is_rank0():
            return
        root = os.path.dirname(self._checkpoint_path())
        os.makedirs(root, exist_ok=True)
        checkpoint = {
            "next_epoch": epoch + 1,
            "state": self.state.to_dict(),
            "modules": {
                "volume": self.volume.state_dict(),
                "pose": self.pose.state_dict(),
                "noise": None if self.noise is None else self.noise.state_dict(),
                "prior": self.prior.state_dict(),
            },
        }
        epoch_path = os.path.join(root, f"epoch_{epoch:03d}.pt")
        latest_tmp = self._checkpoint_path() + ".tmp"
        torch.save(checkpoint, epoch_path)
        torch.save(checkpoint, latest_tmp)
        os.replace(latest_tmp, self._checkpoint_path())

    def _resolve_resume(self) -> str | None:
        if self.resume_checkpoint_path:
            return os.path.abspath(self.resume_checkpoint_path)
        if self.auto_resume and os.path.isfile(self._checkpoint_path()):
            return self._checkpoint_path()
        return None

    def resume_from_checkpoint(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.device)
        modules = checkpoint["modules"]
        self.volume.load_state_dict(modules["volume"])
        self.pose.load_state_dict(modules["pose"])
        if self.noise is not None:
            self.noise.load_state_dict(modules["noise"])
        self.prior.load_state_dict(modules["prior"])
        loaded = OptimState.from_dict(checkpoint["state"])
        self.state.progress = loaded.progress
        self.state.schedule = loaded.schedule
        self.state.heterorefine = loaded.heterorefine
        self.solver.refresh()
        return int(checkpoint["next_epoch"])

    def run(self) -> None:
        set_seed(self.config.reproduce.seed, self.config.reproduce.deterministic)
        logger = setup_logging(
            self.config.logging.log_dir,
            filename_prefix=self.config.logging.log_prefix,
            level=self.config.logging.level,
        )
        log_config(logger, self.config)
        resume_path = self._resolve_resume()
        if resume_path is None:
            self.initialize()
            start_epoch = 0
        else:
            start_epoch = self.resume_from_checkpoint(resume_path)

        for epoch in range(start_epoch, int(self.config.heterorefine.engine.num_epochs)):
            self.state.progress.epoch = epoch
            self.state.progress.iter = 0
            is_ncc = self._is_ncc_epoch(epoch)
            is_bootstrap = self._is_bootstrap_epoch(epoch)
            self.state.heterorefine.engine.is_bootstrap_epoch = is_bootstrap
            self.state.schedule.pose_search_criterion = parse_pose_search_criterion(
                "correlation" if is_ncc else "posterior"
            )
            log_state(
                logger,
                self.state,
                title=f"Epoch {epoch} State",
                command="heterorefine",
            )
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            self.solver.refresh()
            self.solver.zero()

            loader = self.dataloader
            if is_rank0() and tqdm is not None:
                loader = tqdm(loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
            for batch in loader:
                batch = batch.to(self.device, non_blocking=True)
                fixed = self._fixed_volume_index(epoch, batch.particle_index)
                result = self.solver.infer(batch, fixed_volume_index=fixed)
                self.solver.accumulate(result)
                self.state.progress.iter += 1

            self.solver.update()
            self._apply_volume_constraints()
            if is_ncc:
                self._copy_ncc_volume_to_all_classes()
            self._update_prior()
            dvp_scores, weight_spectrum = self._evaluate_dvp()
            self._update_metrics()
            if not is_ncc and not is_bootstrap:
                self.scheduler.step(dvp_scores, weight_spectrum)

            if is_rank0():
                metrics = self.state.heterorefine.metrics
                rot_deg = float(torch.rad2deg(torch.tensor(metrics.rot_update_rms)))
                log_block(
                    logger,
                    title=f"Epoch {epoch} Summary",
                    lines=[
                        f"Phase       : {'ncc' if is_ncc else 'bootstrap' if is_bootstrap else 'soft-em'}",
                        f"Pose Search : L={self.state.schedule.side_length}, healpix={self.state.schedule.healpix_order}, oversampling={self.state.schedule.oversampling}",
                        f"Occupancy   : {metrics.volume_occupancy}",
                        f"Class Change: {100.0 * metrics.volume_class_change_rate:.2f}%",
                        f"Pose RMS    : rot={rot_deg:.2f} deg, trans={metrics.trans_update_rms:.2f} px",
                        f"DVP Radius  : {metrics.dvp_crossing_radius}",
                    ],
                )
            self._save_maps(epoch)
            self.save_checkpoint(epoch)

        self.volume_real = self.volume.volume_real
        self._save_maps(max(0, int(self.config.heterorefine.engine.num_epochs) - 1), final=True)