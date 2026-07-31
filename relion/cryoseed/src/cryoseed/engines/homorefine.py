import os
import time
import logging
import torch
import mrcfile

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from cryoseed.runtime.distributed import RuntimeContext, is_rank0
from cryoseed.fft.fft_torch import primal_to_fourier_3d
from cryoseed.cryoem.mask import lowpass_mask, load_mask_mrc, spherical_mask
from cryoseed.metrics.fsc import calc_fsc, fsc_to_resolution, apply_fsc_weighting_3d
from cryoseed.metrics.fsc import calc_solvent_corrected_fsc, SolventCorrectedFSC
from cryoseed.ops.radial import radial_average
from cryoseed.config import MainConfig
from cryoseed.data import (
    DataBatch,
    ParticleDataset,
    build_distributed_half_dataloaders,
    build_half_dataloaders,
    save_mrc,
)
from cryoseed.engines.external import (
    ExternalReconstructLayout,
    build_external_reconstruct_job,
    build_external_reconstruct_layout,
    write_external_reconstruct_metadata,
)
from cryoseed.engines.external import ExternalReconstructManager
from cryoseed.state import OptimState
from cryoseed.modules.volume import VoxelGrid
from cryoseed.modules.statistics import NoiseVariance, PriorVariance
from cryoseed.modules.pose import Pose
from cryoseed.optim.pose import PoseSearcher
from cryoseed.optim import EMSolver
from cryoseed.optim.scheduler import HomoRefineScheduler
from cryoseed.metrics.fsc import plot_fsc, plot_fsc_multiple, save_fsc_npz, save_fsc_txt
from cryoseed.utils.logging import setup_logging, log_block, log_config, log_state
from cryoseed.utils.reproducibility import set_seed


LOGGER = logging.getLogger(__name__)
HOMOREFINE_FSC_THRESHOLD = 0.143

class HomoRefineEngine(torch.nn.Module):
    def __init__(
        self,
        config: MainConfig,
        runtime: RuntimeContext,
        resume_checkpoint_path: str | None = None,
        auto_resume: bool = False,
    ):
        super().__init__()

        # config
        self.config = config.validate_for_command("homorefine")
        self.resume_checkpoint_path = resume_checkpoint_path
        self.auto_resume = bool(auto_resume)
        self.runtime = runtime
        self.external_reconstruct_manager = ExternalReconstructManager(runtime=runtime)
        if int(self.config.modules.volume.num_volumes) != 1:
            raise ValueError(
                "Refinement requires a single volume (num_volumes must be 1); "
                f"got num_volumes={int(self.config.modules.volume.num_volumes)}"
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

        if runtime.is_distributed:
            (
                dl_half0,
                dl_half1,
                sampler_half0,
                sampler_half1,
                _,
                _,
            ) = build_distributed_half_dataloaders(
                dataset,
                batch_size=config.data.batch_size,
                shuffle=False,
                num_workers=config.data.num_workers,
                device=device,
                seed=int(config.reproduce.seed),
                device_mesh=device_mesh,
            )
        else:
            dl_half0, dl_half1, _, _ = build_half_dataloaders(
                dataset,
                batch_size=config.data.batch_size,
                shuffle=False,
                num_workers=config.data.num_workers,
                device=device,
                seed=int(config.reproduce.seed),
            )
            sampler_half0 = None
            sampler_half1 = None

        self.dataloader_half0 = dl_half0
        self.dataloader_half1 = dl_half1
        self.sampler_half0 = sampler_half0
        self.sampler_half1 = sampler_half1

        # modules
        self.volume_half0 = VoxelGrid.from_config(config, device=device, device_mesh=device_mesh)
        self.volume_half1 = VoxelGrid.from_config(config, device=device, device_mesh=device_mesh)
        self.noise_half0 = NoiseVariance.from_config(config, device=device, device_mesh=device_mesh)
        self.noise_half1 = NoiseVariance.from_config(config, device=device, device_mesh=device_mesh)
        self.prior = PriorVariance.from_config(config, device=device)

        self.pose_half0 = Pose.from_config(config, device=device, device_mesh=device_mesh)
        self.pose_half1 = Pose.from_config(config, device=device, device_mesh=device_mesh)
        self.solvent_mask = self.build_solvent_mask()

        # optimization
        self.state = OptimState.from_config(config, command="homorefine")
        self.pose_searcher_half0 = PoseSearcher(
            state=self.state,
            volume=self.volume_half0,
            noise=self.noise_half0,
            pose=self.pose_half0,
            config=config,
            device=device,
            device_mesh=device_mesh,
        )
        self.pose_searcher_half1 = PoseSearcher(
            state=self.state,
            volume=self.volume_half1,
            noise=self.noise_half1,
            pose=self.pose_half1,
            config=config,
            device=device,
            device_mesh=device_mesh,
        )
        self.solver_half0 = EMSolver(state=self.state, pose_searcher=self.pose_searcher_half0, prior=self.prior)
        self.solver_half1 = EMSolver(state=self.state, pose_searcher=self.pose_searcher_half1, prior=self.prior)
        self.scheduler = HomoRefineScheduler(self.state, device=self.device).from_config(config)

        # placeholder
        self.volume_real_half0 = None
        self.volume_real_half1 = None
        self.unmasked_volume_real_half0 = None
        self.unmasked_volume_real_half1 = None
        self._init_lowpass_side_length: int | None = None
        self.register_buffer("_init_lowpass_mask", None, persistent=False)
        self._confidence_sum = 0.0
        self._confidence_count = 0
        self._volume_class_confidence_sum = 0.0
        self._volume_class_confidence_count = 0

    def _load_optional_module_state(self, module, state_dict, name: str) -> None:
        if module is None:
            if state_dict is not None:
                raise ValueError(f"Checkpoint contains `{name}` state, but the current config disables it.")
            return

        if state_dict is None:
            raise ValueError(f"Checkpoint does not contain `{name}` state, but the current config requires it.")

        module.load_state_dict(state_dict)

    def _load_pose_state(self, pose: Pose, state_dict, name: str) -> bool:
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
        return "valid_count" in state_dict

    @torch.no_grad()
    def _sync_pose_halves_from_valid_count(
        self,
        *,
        has_valid_count_half0: bool,
        has_valid_count_half1: bool,
    ) -> None:
        def warn_pose_sync_skipped(reason: str) -> None:
            if not is_rank0():
                return
            LOGGER.warning(
                "Pose half synchronization skipped during resume: %s. "
                "As a result, local pose search after resume may be unstable or non-deterministic, "
                "and the original local pose search trajectory may not be reproducible.",
                reason,
            )

        if not (has_valid_count_half0 and has_valid_count_half1):
            missing = []
            if not has_valid_count_half0:
                missing.append("`pose_half0.valid_count` is missing from the checkpoint")
            if not has_valid_count_half1:
                missing.append("`pose_half1.valid_count` is missing from the checkpoint")
            warn_pose_sync_skipped(" and ".join(missing))
            return

        mask_half0 = self.pose_half0.valid_count > 0
        mask_half1 = self.pose_half1.valid_count > 0
        if not mask_half0.any():
            warn_pose_sync_skipped("`pose_half0.valid_count` is all zeros")
            return
        if not mask_half1.any():
            warn_pose_sync_skipped("`pose_half1.valid_count` is all zeros")
            return

        overlap = mask_half0 & mask_half1
        if overlap.any():
            warn_pose_sync_skipped("`pose_half0.valid_count` and `pose_half1.valid_count` overlap")
            return

        idx_half0 = torch.nonzero(mask_half0, as_tuple=False).squeeze(1)
        idx_half1 = torch.nonzero(mask_half1, as_tuple=False).squeeze(1)

        self.pose_half1.quat.index_copy_(0, idx_half0, self.pose_half0.quat.index_select(0, idx_half0))
        self.pose_half1.trans.index_copy_(0, idx_half0, self.pose_half0.trans.index_select(0, idx_half0))
        self.pose_half0.quat.index_copy_(0, idx_half1, self.pose_half1.quat.index_select(0, idx_half1))
        self.pose_half0.trans.index_copy_(0, idx_half1, self.pose_half1.trans.index_select(0, idx_half1))

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
            >= int(self.config.homorefine.engine.num_epochs) - 1
        )
        self.state.homorefine.engine.is_final_epoch = (
            bool(self.state.homorefine.engine.is_final_epoch) or is_last_configured_epoch
        )
        self.state.schedule.full_backprojection = (
            bool(self.config.modules.volume.full_backprojection)
            or bool(self.state.homorefine.engine.is_final_epoch)
        )
        self.state.homorefine.engine.skip_external_reconstruct = bool(
            self.state.homorefine.engine.is_final_epoch
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

        self.volume_half0.load_state_dict(modules["volume_half0"])
        self.volume_half1.load_state_dict(modules["volume_half1"])
        has_valid_count_half0 = self._load_pose_state(self.pose_half0, modules["pose_half0"], "pose_half0")
        has_valid_count_half1 = self._load_pose_state(self.pose_half1, modules["pose_half1"], "pose_half1")
        self._sync_pose_halves_from_valid_count(
            has_valid_count_half0=has_valid_count_half0,
            has_valid_count_half1=has_valid_count_half1,
        )
        self._load_optional_module_state(self.noise_half0, modules.get("noise_half0"), "noise_half0")
        self._load_optional_module_state(self.noise_half1, modules.get("noise_half1"), "noise_half1")
        self._load_optional_module_state(self.prior, modules.get("prior"), "prior")
        self.state.progress.epoch = int(next_epoch)
        self.state.progress.half = 0
        self.state.progress.iter = 0
        self.state.homorefine.scheduler.num_epochs_without_resolution_gain = int(
            progress["num_epochs_without_resolution_gain"]
        )
        self.state.homorefine.scheduler.num_epochs_with_small_trans_update = int(
            progress["num_epochs_with_small_trans_update"]
        )
        self.state.homorefine.scheduler.has_converged = bool(progress["has_converged"])

        required_schedule_keys = (
            "pose_search_scope",
            "pose_search_strategy",
            "healpix_order",
            "oversampling",
            "side_length",
            "trans_grid_extent",
            "trans_grid_samples",
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
                self.state.homorefine.engine.is_final_epoch = bool(next_schedule[key])
            else:
                setattr(self.state.schedule, key, next_schedule[key])
        self._sync_execution_flags_from_state()

        self._reset_confidence_metrics()

        self.pose_searcher_half0.refresh()
        self.pose_searcher_half1.refresh()

    def _reset_confidence_metrics(self) -> None:
        self._confidence_sum = 0.0
        self._confidence_count = 0
        self._volume_class_confidence_sum = 0.0
        self._volume_class_confidence_count = 0
        self.state.homorefine.metrics.avg_confidence = 0.0
        self.state.homorefine.metrics.avg_volume_class_confidence = 0.0

    def _update_confidence_metric_means(self) -> None:
        if self._confidence_count <= 0:
            self.state.homorefine.metrics.avg_confidence = 0.0
        else:
            self.state.homorefine.metrics.avg_confidence = (
                self._confidence_sum / float(self._confidence_count)
            )
        if self._volume_class_confidence_count <= 0:
            self.state.homorefine.metrics.avg_volume_class_confidence = 0.0
        else:
            self.state.homorefine.metrics.avg_volume_class_confidence = (
                self._volume_class_confidence_sum
                / float(self._volume_class_confidence_count)
            )

    def _ensure_init_lowpass_cache(self, volume: torch.Tensor) -> tuple[int, torch.Tensor]:
        init_lowpass_angstrom = float(
            self.config.homorefine.engine.init_lowpass_angstrom
        )
        if init_lowpass_angstrom <= 0:
            raise ValueError(
                "homorefine.engine.init_lowpass_angstrom must be > 0, got "
                f"{self.config.homorefine.engine.init_lowpass_angstrom}"
            )

        side_length = 2 * int(
            self.config.data.image_size * self.config.data.angpix / init_lowpass_angstrom
        )
        side_length = max(1, min(int(self.config.data.image_size), int(side_length)))

        mask = self._init_lowpass_mask
        if (
            self._init_lowpass_side_length != side_length
            or mask is None
            or mask.shape != volume.shape
            or mask.device != volume.device
        ):
            mask = lowpass_mask(volume, side_length, ndim=3).to(dtype=torch.bool)
            self._init_lowpass_mask = mask
            self._init_lowpass_side_length = side_length

        return side_length, mask

    def initialize(self):
        # State
        init_volume_real = None
        init_volume = None
        init_side_length = None
        init_mask = None

        with mrcfile.open(self.config.io.ref_volume_path, permissive=True) as mrc:
            init_volume_real = torch.tensor(mrc.data, device=self.device)

        init_volume_real = init_volume_real.unsqueeze(0)
        init_volume = primal_to_fourier_3d(init_volume_real)
        init_side_length, init_mask = self._ensure_init_lowpass_cache(init_volume)
        self.state.schedule.side_length = init_side_length
        self._sync_execution_flags_from_state()

        # Volume
        init_volume *= init_mask

        self.volume_half0.load_volume(init_volume)
        self.volume_half1.load_volume(init_volume)

        # Noise
        if self.noise_half0 is not None:
            self.noise_half0.from_data([self.dataloader_half0, self.dataloader_half1])

        if self.noise_half1 is not None:
            if self.noise_half0 is not None:
                self.noise_half1.load_state_dict(self.noise_half0.state_dict())
            else:
                self.noise_half1.from_data([self.dataloader_half0, self.dataloader_half1])

        # Prior
        if self.prior is not None:
            self.prior.from_volume(init_volume)

    def snapshot_unmasked_halfmaps(self) -> None:
        if self.volume_real_half0 is None or self.volume_real_half1 is None:
            raise RuntimeError("half-maps are unavailable for unmasked snapshot")
        self.unmasked_volume_real_half0 = self.volume_real_half0.detach().clone()
        self.unmasked_volume_real_half1 = self.volume_real_half1.detach().clone()

    def build_solvent_mask(self) -> torch.Tensor | None:
        selector = str(self.config.homorefine.engine.solvent_mask).strip()
        mode = selector.lower()

        if os.path.isfile(selector):
            return load_mask_mrc(
                selector,
                side_length=int(self.config.data.image_size),
                angpix=float(self.config.data.angpix),
                device=self.device,
            )
        if mode == "none":
            if self.config.homorefine.engine.solvent_fsc_correction:
                raise ValueError(
                    "homorefine.engine.solvent_fsc_correction requires a solvent mask"
                )
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
                    self.config.homorefine.engine.solvent_mask_soft_edge_pixels
                ),
                device=self.device,
            ).to(dtype=torch.float32)
        if mode == "auto":
            raise NotImplementedError("automatic solvent masking is not implemented yet")
        raise FileNotFoundError(f"solvent mask does not exist: {selector}")

    def preprocess(self, batch: DataBatch):
        # norm correction
        # real-space mask
        del batch
        return

    @torch.no_grad()
    def _average_half_low_frequencies(self) -> None:
        """Share low-frequency Fourier coefficients between the two half maps."""
        volume_half0 = self.volume_half0.volume.detach().clone()
        volume_half1 = self.volume_half1.volume.detach().clone()
        _, shared_mask = self._ensure_init_lowpass_cache(volume_half0)
        shared_volume = 0.5 * (volume_half0 + volume_half1)

        volume_half0 = torch.where(shared_mask, shared_volume, volume_half0)
        volume_half1 = torch.where(shared_mask, shared_volume, volume_half1)

        self.volume_half0.load_volume(volume_half0)
        self.volume_half1.load_volume(volume_half1)

    @torch.no_grad()
    def apply_solvent_mask(self) -> torch.Tensor | None:
        if self.solvent_mask is None:
            self.volume_real_half0 = self.unmasked_volume_real_half0
            self.volume_real_half1 = self.unmasked_volume_real_half1
            return None
        if (
            self.unmasked_volume_real_half0 is None
            or self.unmasked_volume_real_half1 is None
        ):
            raise RuntimeError("unmasked half-maps must be available before masking")

        mask = self.solvent_mask.unsqueeze(0)
        masked_real_volumes: list[torch.Tensor] = []
        for volume, unmasked_volume_real in (
            (self.volume_half0, self.unmasked_volume_real_half0),
            (self.volume_half1, self.unmasked_volume_real_half1),
        ):
            if unmasked_volume_real is None:
                raise RuntimeError("reference volume is unavailable for solvent masking")
            # Apply the real-space mask, then copy the updated Fourier reference in place.
            masked_real = unmasked_volume_real * mask
            volume.copy_volume_(primal_to_fourier_3d(masked_real))
            masked_real_volumes.append(masked_real)
        self.volume_real_half0, self.volume_real_half1 = masked_real_volumes
        return self.solvent_mask

    def _calc_halfmap_fsc(self, volume_real_half0: torch.Tensor, volume_real_half1: torch.Tensor):
        fsc_scores, fsc_freqs = calc_fsc(
            volume_real_half0.squeeze(0),
            volume_real_half1.squeeze(0),
        )
        fsc_resol = fsc_to_resolution(
            fsc_scores,
            fsc_freqs,
            HOMOREFINE_FSC_THRESHOLD,
            self.config.data.angpix,
        )
        return fsc_scores, fsc_freqs, fsc_resol

    def _truncate_fsc_to_current_side_length(self, fsc_scores: torch.Tensor | None) -> torch.Tensor | None:
        if fsc_scores is None:
            return None
        fsc = torch.as_tensor(fsc_scores, device=self.device).reshape(-1).clone()
        valid_shells = max(0, int(self.state.schedule.side_length) // 2)
        if valid_shells < int(fsc.numel()):
            fsc[valid_shells:] = 0
        return fsc

    def _truncate_solvent_corrected_fsc_to_current_side_length(
        self,
        correction: SolventCorrectedFSC,
    ) -> SolventCorrectedFSC:
        return SolventCorrectedFSC(
            fsc_freqs=correction.fsc_freqs,
            corrected=self._truncate_fsc_to_current_side_length(correction.corrected),
            unmasked=self._truncate_fsc_to_current_side_length(correction.unmasked),
            masked=self._truncate_fsc_to_current_side_length(correction.masked),
            randomized_masked=self._truncate_fsc_to_current_side_length(
                correction.randomized_masked
            ),
            phase_randomization_frequency=correction.phase_randomization_frequency,
        )

    @torch.no_grad()
    def _accumulate_confidence_metrics_from_pose(self, pose: Pose) -> None:
        if self.state.schedule.pose_search_criterion != "posterior":
            return
        valid_count = int(pose.valid_count.sum().item())
        if valid_count <= 0:
            return
        self._confidence_sum += float(pose.avg_confidence.item()) * valid_count
        self._confidence_count += valid_count
        self._volume_class_confidence_sum += (
            float(pose.avg_volume_class_confidence.item()) * valid_count
        )
        self._volume_class_confidence_count += valid_count
        self._update_confidence_metric_means()

    def evaluate(self):
        volume_real_half0 = self.volume_real_half0
        volume_real_half1 = self.volume_real_half1
        if volume_real_half0 is None or volume_real_half1 is None:
            raise RuntimeError("half-maps are unavailable for FSC evaluation")

        correction = None
        if self.config.homorefine.engine.solvent_fsc_correction:
            if (
                self.solvent_mask is None
                or self.unmasked_volume_real_half0 is None
                or self.unmasked_volume_real_half1 is None
            ):
                raise RuntimeError("solvent FSC correction requires a solvent mask")
            correction = calc_solvent_corrected_fsc(
                self.unmasked_volume_real_half0.squeeze(0),
                self.unmasked_volume_real_half1.squeeze(0),
                self.solvent_mask,
                seed=int(self.config.reproduce.seed) + 2 * int(self.state.progress.epoch),
            )
            correction = self._truncate_solvent_corrected_fsc_to_current_side_length(correction)
            fsc_scores = correction.corrected
            fsc_freqs = correction.fsc_freqs
        else:
            fsc_scores, fsc_freqs, _ = self._calc_halfmap_fsc(
                volume_real_half0,
                volume_real_half1,
            )
            fsc_scores = self._truncate_fsc_to_current_side_length(fsc_scores)
        fsc_resol = fsc_to_resolution(
            fsc_scores,
            fsc_freqs,
            HOMOREFINE_FSC_THRESHOLD,
            self.config.data.angpix,
        )
        prev_fsc_resolution = self.state.homorefine.metrics.fsc_resolution
        if prev_fsc_resolution is None:
            fsc_resolution_change = None
        else:
            fsc_resolution_change = float(fsc_resol) - float(prev_fsc_resolution)

        self.state.homorefine.metrics.fsc_scores = torch.as_tensor(
            fsc_scores,
            dtype=torch.float32,
            device=self.device,
        )
        self.state.homorefine.metrics.fsc_resolution = float(fsc_resol)
        self.state.homorefine.metrics.fsc_resolution_change = fsc_resolution_change
        self.state.homorefine.metrics.rot_update_rms = 0.5 * (
            float(self.pose_half0.rot_update_rms.item())
            + float(self.pose_half1.rot_update_rms.item())
        )
        self.state.homorefine.metrics.trans_update_rms = 0.5 * (
            float(self.pose_half0.trans_update_rms.item())
            + float(self.pose_half1.trans_update_rms.item())
        )
        self._accumulate_confidence_metrics_from_pose(self.pose_half0)
        self._accumulate_confidence_metrics_from_pose(self.pose_half1)

        epoch = self.state.progress.epoch
        output_fsc_path = os.path.join(self.config.io.output_path, "fsc", f"epoch_{epoch:03d}")
        os.makedirs(output_fsc_path, exist_ok=True)
        fsc_plot_path = os.path.join(output_fsc_path, "fsc.png")
        if correction is None:
            plot_fsc(
                fsc_scores,
                fsc_freqs,
                threshold=HOMOREFINE_FSC_THRESHOLD,
                angpix=self.config.data.angpix,
                save_path=fsc_plot_path,
                color="tab:blue",
            )
        else:
            fsc_curves = [correction.unmasked, correction.masked]
            fsc_labels = ["unmasked", "masked"]
            fsc_colors = ["tab:blue", "tab:orange"]
            if correction.randomized_masked is not None:
                fsc_curves.append(correction.randomized_masked)
                fsc_labels.append("randomized")
                fsc_colors.append("tab:gray")
            fsc_curves.append(correction.corrected)
            fsc_labels.append("corrected")
            fsc_colors.append("tab:purple")
            plot_fsc_multiple(
                fsc_curves,
                fsc_freqs,
                labels=fsc_labels,
                threshold=HOMOREFINE_FSC_THRESHOLD,
                angpix=self.config.data.angpix,
                save_path=fsc_plot_path,
                colors=fsc_colors,
            )
        save_fsc_npz(
            os.path.join(output_fsc_path, "fsc.npz"),
            fsc_freqs,
            fsc_scores,
            epoch=epoch,
            resolution=fsc_resol,
            fsc_unmasked=None if correction is None else correction.unmasked,
            fsc_masked=None if correction is None else correction.masked,
            fsc_randomized_masked=None if correction is None else correction.randomized_masked,
            fsc_corrected=None if correction is None else correction.corrected,
            phase_randomization_frequency=None
            if correction is None
            else correction.phase_randomization_frequency,
        )
        save_fsc_txt(
            os.path.join(output_fsc_path, "fsc.txt"),
            fsc_freqs,
            fsc_scores,
            epoch=epoch,
            resolution=fsc_resol,
            fsc_unmasked=None if correction is None else correction.unmasked,
            fsc_masked=None if correction is None else correction.masked,
            fsc_randomized_masked=None if correction is None else correction.randomized_masked,
            fsc_corrected=None if correction is None else correction.corrected,
            phase_randomization_frequency=None
            if correction is None
            else correction.phase_randomization_frequency,
        )
 
    def update_prior(self):
        if self.prior is None:
            return

        weight = (self.volume_half0.accumulated_weight + self.volume_half1.accumulated_weight) * 0.5
        # squeeze singleton "K" dimension: (1,D,D,D)->(D,D,D)
        if weight.ndim == 4 and int(weight.shape[0]) == 1:
            weight = weight[0]

        weight_1d = radial_average(weight, self.config.data.image_size // 2, ndim=3, use_cache=True)
        # squeeze singleton "K" dimension: (1,R)->(R,)
        if weight_1d.ndim == 2 and int(weight_1d.shape[0]) == 1:
            weight_1d = weight_1d[0]
        if weight_1d.ndim != 1:
            weight_1d = weight_1d.reshape(-1, weight_1d.shape[-1]).mean(dim=0)

        fsc_scores = torch.as_tensor(
            self.state.homorefine.metrics.fsc_scores,
            device=weight_1d.device,
            dtype=torch.float32,
        )
        self.prior.update(fsc_scores, weight_1d)

    def _external_reconstruct_root(self) -> str:
        """Return the top-level directory used by the external reconstruction bridge."""
        return os.path.abspath(os.path.join(self.config.io.output_path, "external_reconstruct"))

    def _external_reconstruct_paths(self, half_index: int) -> ExternalReconstructLayout:
        """Build the file layout for one external reconstruction call."""
        return build_external_reconstruct_layout(
            output_root=self._external_reconstruct_root(),
            epoch=int(self.state.progress.epoch),
            half_index=half_index,
        )

    def _external_prior_variance(self) -> torch.Tensor:
        """Return the prior variance spectrum exported to the external tool."""
        if self.prior is not None:
            return self.prior.variance.detach().clone()
        return torch.full(
            (self.config.data.image_size // 2 + 1,),
            float(self.config.modules.statistics.prior.init_variance),
            dtype=torch.float32,
            device=self.device,
        )

    def _external_reconstruct_fsc(self, fsc_scores, num_shells: int) -> torch.Tensor:
        """Match a half-map FSC vector to the number of exported spectral shells."""
        fsc = torch.as_tensor(fsc_scores, dtype=torch.float32, device=self.device).reshape(-1)
        if int(fsc.numel()) == num_shells - 1:
            # Prepend the DC shell when the measured FSC starts from the first non-zero frequency.
            fsc = torch.cat((fsc.new_tensor([1.0]), fsc), dim=0)
        elif int(fsc.numel()) < num_shells:
            pad = torch.ones(num_shells - int(fsc.numel()), dtype=fsc.dtype, device=fsc.device)
            fsc = torch.cat((pad, fsc), dim=0)
        elif int(fsc.numel()) > num_shells:
            fsc = fsc[:num_shells]
        return fsc

    def _load_external_reconstruct_result(self, result_path: str) -> torch.Tensor:
        """Load the reconstructed real-space volume returned by the external tool."""
        with mrcfile.open(result_path, permissive=True) as mrc:
            volume_real = torch.tensor(mrc.data, dtype=torch.float32, device=self.device)
        if volume_real.ndim != 3:
            raise ValueError(
                f"External reconstruction result must be a 3D MRC volume, got shape={tuple(volume_real.shape)}"
            )
        return volume_real.unsqueeze(0)

    def _prepare_external_reconstruct_half(
        self,
        *,
        half_index: int,
        volume: VoxelGrid,
        volume_real: torch.Tensor,
        fsc: torch.Tensor,
    ) -> ExternalReconstructLayout:
        """Export one half-map state to disk for external reconstruction."""
        def as_single_volume(tensor: torch.Tensor, name: str) -> torch.Tensor:
            if tensor.ndim == 4 and int(tensor.shape[0]) == 1:
                tensor = tensor[0]
            if tensor.ndim != 3:
                raise ValueError(f"{name} must be a single 3D volume, got shape={tuple(tensor.shape)}")
            return tensor

        paths = self._external_reconstruct_paths(half_index)
        prior_variance = self._external_prior_variance()
        data = volume.accumulated_data
        weight = volume.accumulated_weight

        if is_rank0():
            os.makedirs(paths.work_dir, exist_ok=True)
            save_mrc(
                file_path=paths.data_real,
                data=as_single_volume(data.real.detach(), f"half{half_index} accumulated_data real"),
                voxel_size=self.config.data.angpix,
            )
            save_mrc(
                file_path=paths.data_imag,
                data=as_single_volume(data.imag.detach(), f"half{half_index} accumulated_data imag"),
                voxel_size=self.config.data.angpix,
            )
            save_mrc(
                file_path=paths.weight,
                data=as_single_volume(weight.detach(), f"half{half_index} accumulated_weight"),
                voxel_size=self.config.data.angpix,
            )
            save_mrc(
                file_path=paths.result,
                data=as_single_volume(volume_real.detach(), f"half{half_index} volume_real"),
                voxel_size=self.config.data.angpix,
            )

            write_external_reconstruct_metadata(
                star_path=paths.star,
                data_real_path=paths.data_real,
                data_imag_path=paths.data_imag,
                weight_path=paths.weight,
                result_path=paths.result,
                result_star_path=paths.result_star,
                original_image_size=int(self.config.data.image_size),
                current_image_size=int(self.state.schedule.side_length),
                pixel_size=float(self.config.data.angpix),
                particle_diameter=float(self.config.data.particle_diameter),
                prior_variance=prior_variance,
                fsc=fsc,
            )
        return paths

    def _apply_external_reconstruct_result(
        self,
        *,
        volume: VoxelGrid,
        result_path: str,
    ) -> torch.Tensor:
        """Reload the external result by copying the Fourier volume in place."""
        volume_real = self._load_external_reconstruct_result(result_path)
        volume.copy_volume_(primal_to_fourier_3d(volume_real))
        return volume_real

    def external_reconstruct(self) -> None:
        """Run the optional external reconstruction bridge for both half maps."""
        if not bool(self.config.homorefine.engine.external_reconstruct):
            return
        if (
            self.unmasked_volume_real_half0 is None
            or self.unmasked_volume_real_half1 is None
        ):
            raise RuntimeError("unmasked half-maps are unavailable for external reconstruction")

        external_fsc_scores, _, _ = self._calc_halfmap_fsc(
            self.unmasked_volume_real_half0,
            self.unmasked_volume_real_half1,
        )
        external_prior_variance = self._external_prior_variance()
        external_fsc = self._external_reconstruct_fsc(
            external_fsc_scores,
            int(external_prior_variance.numel()),
        )

        half0_paths = self._prepare_external_reconstruct_half(
            half_index=0,
            volume=self.volume_half0,
            volume_real=self.unmasked_volume_real_half0,
            fsc=external_fsc,
        )
        half1_paths = self._prepare_external_reconstruct_half(
            half_index=1,
            volume=self.volume_half1,
            volume_real=self.unmasked_volume_real_half1,
            fsc=external_fsc,
        )

        jobs = [
            build_external_reconstruct_job(name="half0", layout=half0_paths),
            build_external_reconstruct_job(name="half1", layout=half1_paths),
        ]
        self.external_reconstruct_manager.run(
            jobs,
            run_id=f"epoch_{int(self.state.progress.epoch):03d}",
        )

        self.volume_real_half0 = self._apply_external_reconstruct_result(
            volume=self.volume_half0,
            result_path=half0_paths.result,
        )
        self.volume_real_half1 = self._apply_external_reconstruct_result(
            volume=self.volume_half1,
            result_path=half1_paths.result,
        )
        self.snapshot_unmasked_halfmaps()

    def save_result(self):
        if not is_rank0():
            return None

        epoch = self.state.progress.epoch
        if self.volume_real_half0 is None or self.volume_real_half1 is None:
            raise RuntimeError("working half-maps are unavailable for saving")
        if (
            self.unmasked_volume_real_half0 is None
            or self.unmasked_volume_real_half1 is None
        ):
            raise RuntimeError("unmasked half-maps are unavailable for saving")

        output_checkpoints_root = os.path.join(self.config.io.output_path, "checkpoints")
        output_checkpoint_path = os.path.join(self.config.io.output_path, "checkpoints", f"epoch_{epoch:03d}")
        output_map_path = os.path.join(self.config.io.output_path, "maps", f"epoch_{epoch:03d}")
        os.makedirs(output_checkpoint_path, exist_ok=True)
        os.makedirs(output_map_path, exist_ok=True)

        ckpt = {
            "modules": {
                "volume_half0": self.volume_half0.state_dict(),
                "volume_half1": self.volume_half1.state_dict(),
                "pose_half0": self.pose_half0.state_dict(),
                "pose_half1": self.pose_half1.state_dict(),
                "noise_half0": self.noise_half0.state_dict() if self.noise_half0 else None,
                "noise_half1": self.noise_half1.state_dict() if self.noise_half1 else None,
                "prior": self.prior.state_dict() if self.prior else None,
            },

            "pose_searcher_half0": self.pose_searcher_half0.state_dict(),
            "pose_searcher_half1": self.pose_searcher_half1.state_dict(),
            "progress": {
                "num_epochs_without_resolution_gain": self.state.homorefine.scheduler.num_epochs_without_resolution_gain,
                "num_epochs_with_small_trans_update": self.state.homorefine.scheduler.num_epochs_with_small_trans_update,
                "has_converged": self.state.homorefine.scheduler.has_converged,
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
                "pose_translation_center_mode": self.state.schedule.pose_translation_center_mode,
                "use_pose_translation_as_center": self.state.schedule.use_pose_translation_as_center,
                "use_particle_mask": self.state.schedule.use_particle_mask,
                "particle_mask_extra_diameter_angstrom": self.state.schedule.particle_mask_extra_diameter_angstrom,
                "proj_cache_backend": self.state.schedule.proj_cache_backend,
                "is_final_epoch": self.state.homorefine.engine.is_final_epoch,
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

        save_mrc(file_path=os.path.join(output_map_path, f"epoch_{epoch:03d}_half0_masked.mrc"),
                data=self.volume_real_half0.squeeze(0),
                voxel_size=self.config.data.angpix)

        save_mrc(file_path=os.path.join(output_map_path, f"epoch_{epoch:03d}_half1_masked.mrc"),
                data=self.volume_real_half1.squeeze(0),
                voxel_size=self.config.data.angpix)

        save_mrc(
            file_path=os.path.join(output_map_path, f"epoch_{epoch:03d}_half0_unmasked.mrc"),
            data=self.unmasked_volume_real_half0.squeeze(0),
            voxel_size=self.config.data.angpix,
        )

        save_mrc(
            file_path=os.path.join(output_map_path, f"epoch_{epoch:03d}_half1_unmasked.mrc"),
            data=self.unmasked_volume_real_half1.squeeze(0),
            voxel_size=self.config.data.angpix,
        )

        volume_real_avg = (
            self.unmasked_volume_real_half0 + self.unmasked_volume_real_half1
        ) * 0.5
        volume_real_weighting = (self.volume_real_half0 + self.volume_real_half1) * 0.5
        volume_real_weighted = apply_fsc_weighting_3d(
            volume_real_weighting.squeeze(0),
            self.state.homorefine.metrics.fsc_scores,
        ).unsqueeze(0)

        save_mrc(file_path=os.path.join(output_map_path, f"epoch_{epoch:03d}_volume_unmasked.mrc"),
                data=volume_real_avg.squeeze(0),
                voxel_size=self.config.data.angpix)

        save_mrc(file_path=os.path.join(output_map_path, f"epoch_{epoch:03d}_volume_masked_weighted.mrc"),
                data=volume_real_weighted.squeeze(0),
                voxel_size=self.config.data.angpix)

        if self.solvent_mask is not None:
            save_mrc(
                file_path=os.path.join(output_map_path, f"epoch_{epoch:03d}_solvent_mask.mrc"),
                data=self.solvent_mask.to(dtype=torch.float32),
                voxel_size=self.config.data.angpix,
            )

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

            if start_epoch >= int(self.config.homorefine.engine.num_epochs):
                logger.warning(
                    "No epochs to run after resume: start_epoch=%d, num_epochs=%d",
                    start_epoch,
                    int(self.config.homorefine.engine.num_epochs),
                )
                return

            completed_via_convergence_final_epoch = False

            # loop
            for epoch in range(start_epoch, self.config.homorefine.engine.num_epochs):
                self.state.progress.epoch = epoch
                self._reset_confidence_metrics()
                current_is_final_epoch = bool(self.state.homorefine.engine.is_final_epoch)
                current_is_convergence_final_epoch = (
                    current_is_final_epoch
                    and self.state.homorefine.scheduler.has_converged
                )

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                epoch_wall_start = time.perf_counter()

                logger.info("Epoch %d started", epoch)

                self.solver_half0.zero_accum()
                self.solver_half1.zero_accum()
                self._average_half_low_frequencies()

                # half 0
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                half0_wall_start = time.perf_counter()

                logger.info("Epoch %d Half 0 started", epoch)
                self.state.progress.half = 0
                self.state.progress.iter = 0
                log_state(
                    logger,
                    self.state,
                    title=f"Epoch {epoch} Half 0 State",
                    command="homorefine",
                )
                if self.sampler_half0 is not None:
                    self.sampler_half0.set_epoch(epoch)
                self.solver_half0.refresh()

                dl0 = self.dataloader_half0
                if is_rank0() and tqdm is not None:
                    dl0 = tqdm(dl0, desc=f"Epoch {epoch} Half 0", dynamic_ncols=True)

                for batch in dl0:
                    batch = batch.to(self.device, non_blocking=True)
                    result = self.solver_half0.infer(batch)
                    self.solver_half0.accumulate(result)
                    self.state.progress.iter += 1

                self.solver_half0.update()

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                half0_wall = time.perf_counter() - half0_wall_start
                logger.info(
                    "Epoch %d Half 0 finished | time=%.3fs",
                    epoch,
                    half0_wall,
                )

                # half 1
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                half1_wall_start = time.perf_counter()

                logger.info("Epoch %d Half 1 started", epoch)
                self.state.progress.half = 1
                self.state.progress.iter = 0
                log_state(
                    logger,
                    self.state,
                    title=f"Epoch {epoch} Half 1 State",
                    command="homorefine",
                )
                if self.sampler_half1 is not None:
                    self.sampler_half1.set_epoch(epoch)
                self.solver_half1.refresh()

                dl1 = self.dataloader_half1
                if is_rank0() and tqdm is not None:
                    dl1 = tqdm(dl1, desc=f"Epoch {epoch} Half 1", dynamic_ncols=True)

                for batch in dl1:
                    batch = batch.to(self.device, non_blocking=True)
                    result = self.solver_half1.infer(batch)
                    self.solver_half1.accumulate(result)
                    self.state.progress.iter += 1

                self.solver_half1.update()

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                half1_wall = time.perf_counter() - half1_wall_start
                logger.info(
                    "Epoch %d Half 1 finished | time=%.3fs",
                    epoch,
                    half1_wall,
                )

                # Current-epoch state conventions:
                # - unmasked_volume_real_half* stores the external-reconstructed,
                #   pre-mask real-space state.
                # - volume_real_half* stores the final working real-space state
                #   after optional solvent masking.
                self.volume_real_half0 = self.volume_half0.volume_real
                self.volume_real_half1 = self.volume_half1.volume_real
                self.snapshot_unmasked_halfmaps()

                # optional external reconstruction
                if self.state.homorefine.engine.skip_external_reconstruct:
                    logger.info(
                        "Epoch %d skipping external reconstruction during final epoch",
                        epoch,
                    )
                else:
                    self.external_reconstruct()

                # optional solvent masking
                self.apply_solvent_mask()

                # half-map fsc
                self.evaluate()

                # update prior
                self.update_prior()

                # epoch summary
                if is_rank0():
                    rot_rms_deg = torch.rad2deg(
                        torch.tensor(self.state.homorefine.metrics.rot_update_rms)
                    ).item()
                    log_block(
                        logger,
                        title=f"Epoch {epoch} Summary",
                        lines=[
                            f"Pose Search : L={self.state.schedule.side_length}, healpix={self.state.schedule.healpix_order}, oversampling={self.state.schedule.oversampling}, trans_extent={self.state.schedule.trans_grid_extent:.2f}, criterion={self.state.schedule.pose_search_criterion}",
                            f"Particle Mask: enabled={self.state.schedule.use_particle_mask}, extra_diameter={self.state.schedule.particle_mask_extra_diameter_angstrom:.2f} A",
                            f"Backproject : full_bp={self.state.schedule.full_backprojection}",
                            f"Resolution  : {float(self.state.homorefine.metrics.fsc_resolution):.2f} Angstrom",
                            f"Confidence  : {100.0 * float(self.state.homorefine.metrics.avg_confidence):.2f}%",
                            f"Rot RMS     : {rot_rms_deg:.2f} deg",
                            f"Trans RMS   : {self.state.homorefine.metrics.trans_update_rms:.2f} px",
                        ],
                    )

                self.scheduler.step()
                self._reset_confidence_metrics()

                if (
                    self.state.homorefine.scheduler.has_converged
                    and is_rank0()
                    and not current_is_final_epoch
                ):
                    log_block(
                        logger,
                        title=f"Converged At Epoch {epoch}",
                        lines=[
                            "convergence conditions have been met for "
                            f"{self.config.homorefine.scheduler.convergence_patience} consecutive epochs",
                            "FSC has shown no meaningful gain for "
                            f"{self.state.homorefine.scheduler.num_epochs_without_resolution_gain} consecutive epochs "
                            f"(resolution={float(self.state.homorefine.metrics.fsc_resolution):.4f} A)",
                            "translation update RMS has stayed below threshold for "
                            f"{self.state.homorefine.scheduler.num_epochs_with_small_trans_update} consecutive epochs "
                            f"(trans_rms={self.state.homorefine.metrics.trans_update_rms:.2f} px, "
                            f"threshold={self.config.homorefine.scheduler.trans_update_rms_threshold:.2f} px)",
                        ],
                    )

                # next epoch plan
                if (
                    is_rank0()
                    and (epoch + 1 < int(self.config.homorefine.engine.num_epochs))
                    and not current_is_final_epoch
                ):
                    log_block(
                        logger,
                        title=f"Next Epoch Configuration (Epoch {epoch+1})",
                        lines=[
                            f"Pose Search : L={self.state.schedule.side_length}, healpix={self.state.schedule.healpix_order}, oversampling={self.state.schedule.oversampling}, trans_extent={self.state.schedule.trans_grid_extent:.2f}, criterion={self.state.schedule.pose_search_criterion}",
                            f"Scope       : {self.state.schedule.pose_search_scope}",
                            f"Strategy    : {self.state.schedule.pose_search_strategy}",
                            f"Cache       : {self.state.schedule.proj_cache_backend}",
                            f"Particle Mask: enabled={self.state.schedule.use_particle_mask}, extra_diameter={self.state.schedule.particle_mask_extra_diameter_angstrom:.2f} A",
                            f"Backproject : full_bp={self.state.schedule.full_backprojection}",
                            f"Final Epoch : {self.state.homorefine.engine.is_final_epoch}",
                        ],
                    )

                # save results
                checkpoint_paths = self.save_result()
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
                    "Epoch %d finished | total=%.3fs | half0=%.3fs | half1=%.3fs",
                    epoch,
                    epoch_wall,
                    half0_wall,
                    half1_wall,
                )

                if current_is_final_epoch:
                    completed_via_convergence_final_epoch = current_is_convergence_final_epoch
                    if current_is_convergence_final_epoch and is_rank0():
                        log_block(
                            logger,
                            title="Refinement Completed",
                            lines=[
                                "Completed after convergence-triggered final epoch",
                                f"Final resolution : {float(self.state.homorefine.metrics.fsc_resolution):.4f} Angstrom",
                            ],
                        )
                    break

                if (
                    self.state.homorefine.scheduler.has_converged
                    and self.state.homorefine.engine.is_final_epoch
                ):
                    logger.info(
                        "Refinement entering final epoch after epoch %d | reason=%s",
                        epoch,
                        "convergence conditions satisfied: FSC has shown no meaningful gain for "
                        f"{self.state.homorefine.scheduler.num_epochs_without_resolution_gain} consecutive epochs "
                        f"(resolution={float(self.state.homorefine.metrics.fsc_resolution):.4f} A), "
                        "translation update RMS has stayed below threshold for "
                        f"{self.state.homorefine.scheduler.num_epochs_with_small_trans_update} consecutive epochs "
                        f"(trans_rms={self.state.homorefine.metrics.trans_update_rms:.2f} px, "
                        f"threshold={self.config.homorefine.scheduler.trans_update_rms_threshold:.2f} px)",
                    )

            if not completed_via_convergence_final_epoch and is_rank0():
                log_block(
                    logger,
                    title="Refinement Completed",
                    lines=[
                        f"Completed all {int(self.config.homorefine.engine.num_epochs)} epochs",
                        f"Final resolution : {float(self.state.homorefine.metrics.fsc_resolution):.4f} Angstrom",
                    ],
                )

        except Exception:
            logger.exception("Refinement failed")
            raise