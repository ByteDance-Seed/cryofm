import os
import gc
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
from cryoseed.cryoem.mask import lowpass_mask
from cryoseed.metrics.fsc import calc_fsc, fsc_to_resolution
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
from cryoseed.optim import FrequencyMarchingScheduler
from cryoseed.metrics.fsc import plot_fsc, save_fsc_npz, save_fsc_txt
from cryoseed.utils.logging import setup_logging, log_block, log_config, log_state
from cryoseed.utils.reproducibility import set_seed


LOGGER = logging.getLogger(__name__)

def free_cuda(*objs):
    for o in objs:
        try:
            del o
        except:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
        self.config = config
        self.resume_checkpoint_path = resume_checkpoint_path
        self.auto_resume = bool(auto_resume)
        self.runtime = runtime
        self.external_reconstruct_manager = ExternalReconstructManager(runtime=runtime)
        if int(config.reconstruction.num_volumes) != 1:
            raise ValueError(
                "Refinement requires a single volume (num_volumes must be 1); "
                f"got num_volumes={int(config.reconstruction.num_volumes)}"
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
                idx_half0,
                idx_half1,
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
            dl_half0, dl_half1, idx_half0, idx_half1 = build_half_dataloaders(
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

        # optimization
        self.state = OptimState.from_config(config)
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
        self.scheduler = FrequencyMarchingScheduler(self.state, device=self.device).from_config(config)

        # placeholder
        self.volume_real_half0 = None
        self.volume_real_half1 = None

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
        optional_keys = {"valid_count"}
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

    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        modules = ckpt.get("modules")
        if not isinstance(modules, dict):
            raise ValueError("Checkpoint is missing a valid `modules` section.")

        next_epoch = ckpt.get("next_epoch")
        next_schedule = ckpt.get("next_schedule")
        if next_epoch is None:
            raise ValueError("Checkpoint is missing `next_epoch`.")
        if not isinstance(next_schedule, dict):
            raise ValueError("Checkpoint is missing a valid `next_schedule` section.")

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

        required_schedule_keys = (
            "pose_search_scope",
            "pose_search_strategy",
            "healpix_order",
            "oversampling",
            "side_length",
            "trans_grid_extent",
            "proj_cache_backend",
        )
        for key in required_schedule_keys:
            if key not in next_schedule:
                raise ValueError(f"Checkpoint `next_schedule` is missing `{key}`.")
            setattr(self.state.schedule, key, next_schedule[key])
        if "full_backprojection" in next_schedule:
            self.state.schedule.full_backprojection = bool(next_schedule["full_backprojection"])

        self.state.metrics.confidence_sum = 0.0
        self.state.metrics.confidence_count = 0

        self.pose_searcher_half0.refresh()
        self.pose_searcher_half1.refresh()

    def initialize(self):
        # State
        init_side_length = 2 * int(self.config.data.image_size * self.config.data.angpix / self.config.refinement.init_lowpass_angstrom)
        self.state.schedule.side_length = init_side_length
        self.state.schedule.full_backprojection = (
            bool(self.config.reconstruction.full_backprojection)
            or int(self.config.refinement.num_epochs) <= 1
        )

        # Volume
        with mrcfile.open(self.config.io.ref_volume_path, permissive=True) as mrc:
            init_volume_real = torch.tensor(mrc.data,device=self.device)

        init_volume_real = init_volume_real.unsqueeze(0)
        init_volume = primal_to_fourier_3d(init_volume_real)

        mask = lowpass_mask(init_volume, init_side_length, ndim=3)
        init_volume *= mask

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

    def preprocess(self, batch: DataBatch):
        # solvent mask
        # norm correction
        # real-space mask
        return

    def evaluate(self):
        vol0 = self.volume_real_half0.squeeze(0).detach().cpu().numpy()
        vol1 = self.volume_real_half1.squeeze(0).detach().cpu().numpy()

        fsc_scores_np, fsc_freqs_np = calc_fsc(vol0, vol1)
        fsc_resol = fsc_to_resolution(
            fsc_scores_np,
            fsc_freqs_np,
            self.config.refinement.fsc_threshold,
            self.config.data.angpix,
        )
        prev_fsc_resolution = self.state.metrics.fsc_resolution
        if prev_fsc_resolution is None:
            fsc_resolution_change = None
        else:
            fsc_resolution_change = float(fsc_resol) - float(prev_fsc_resolution)

        self.state.metrics.fsc_scores = torch.as_tensor(
            fsc_scores_np,
            dtype=torch.float32,
            device=self.device,
        )
        self.state.metrics.fsc_resolution = float(fsc_resol)
        self.state.metrics.fsc_resolution_change = fsc_resolution_change
        self.state.metrics.trans_update_rms = 0.5 * (
            float(self.pose_half0.trans_update_rms.item())
            + float(self.pose_half1.trans_update_rms.item())
        )

        epoch = self.state.progress.epoch
        output_fsc_path = os.path.join(self.config.io.output_path, "fsc", f"epoch_{epoch:03d}")
        os.makedirs(output_fsc_path, exist_ok=True)
        plot_fsc(
            fsc_scores_np,
            fsc_freqs_np,
            threshold=self.config.refinement.fsc_threshold,
            voxel_size=self.config.data.angpix,
            save_path=os.path.join(output_fsc_path, "fsc.png"),
        )
        save_fsc_npz(
            os.path.join(output_fsc_path, "fsc.npz"),
            fsc_freqs_np,
            fsc_scores_np,
            iter_=epoch,
            resolution=fsc_resol,
        )
        save_fsc_txt(
            os.path.join(output_fsc_path, "fsc.txt"),
            fsc_freqs_np,
            fsc_scores_np,
            iter_=epoch,
            resolution=fsc_resol,
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
            self.state.metrics.fsc_scores,
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
            float(self.config.statistics.init_variance),
            dtype=torch.float32,
            device=self.device,
        )

    def _external_reconstruct_fsc(self, num_shells: int) -> torch.Tensor:
        """Match the saved FSC vector to the number of exported spectral shells."""
        fsc = torch.as_tensor(self.state.metrics.fsc_scores, dtype=torch.float32, device=self.device).reshape(-1)
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
                fsc=self._external_reconstruct_fsc(int(prior_variance.numel())),
            )
        return paths

    def _apply_external_reconstruct_result(
        self,
        *,
        volume: VoxelGrid,
        result_path: str,
    ) -> torch.Tensor:
        """Reload the external result while preserving accumulation buffers."""
        volume_real = self._load_external_reconstruct_result(result_path)
        accum_numer = volume.accum_numer
        accum_denom = volume.accum_denom
        volume.load_volume(primal_to_fourier_3d(volume_real))
        volume.accum_numer = accum_numer
        volume.accum_denom = accum_denom
        return volume_real

    def external_reconstruct(self) -> None:
        """Run the optional external reconstruction bridge for both half maps."""
        if not bool(self.config.reconstruction.external_reconstruct):
            return

        half0_paths = self._prepare_external_reconstruct_half(
            half_index=0,
            volume=self.volume_half0,
            volume_real=self.volume_real_half0,
        )
        half1_paths = self._prepare_external_reconstruct_half(
            half_index=1,
            volume=self.volume_half1,
            volume_real=self.volume_real_half1,
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

    def save_result(self):
        if not is_rank0():
            return None

        epoch = self.state.progress.epoch
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

            "next_epoch": epoch + 1,
            "next_schedule":{
                "pose_search_scope": self.state.schedule.pose_search_scope,
                "pose_search_strategy": self.state.schedule.pose_search_strategy,
                "healpix_order": self.state.schedule.healpix_order,
                "oversampling": self.state.schedule.oversampling,
                "side_length": self.state.schedule.side_length,
                "trans_grid_extent": self.state.schedule.trans_grid_extent,
                "proj_cache_backend": self.state.schedule.proj_cache_backend,
                "full_backprojection": self.state.schedule.full_backprojection,
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

        save_mrc(file_path=os.path.join(output_map_path, f"epoch_{epoch:03d}_half0.mrc"),
                data=self.volume_real_half0.squeeze(0),
                voxel_size=self.config.data.angpix)

        save_mrc(file_path=os.path.join(output_map_path, f"epoch_{epoch:03d}_half1.mrc"),
                data=self.volume_real_half1.squeeze(0),
                voxel_size=self.config.data.angpix)

        volume_real = (self.volume_real_half0 + self.volume_real_half1) * 0.5

        save_mrc(file_path=os.path.join(output_map_path, f"epoch_{epoch:03d}_volume.mrc"),
                data=volume_real.squeeze(0),
                voxel_size=self.config.data.angpix)

        return checkpoint_paths

    def run(self):
        # set seed
        set_seed(self.config.reproduce.seed, self.config.reproduce.deterministic)

        logger = setup_logging(self.config.logging.log_dir, filename_prefix=self.config.logging.log_prefix)
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

            if start_epoch >= int(self.config.refinement.num_epochs):
                logger.warning(
                    "No epochs to run after resume: start_epoch=%d, num_epochs=%d",
                    start_epoch,
                    int(self.config.refinement.num_epochs),
                )
                return

            # loop
            for epoch in range(start_epoch, self.config.refinement.num_epochs):
                self.state.progress.epoch = epoch

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                epoch_wall_start = time.perf_counter()

                logger.info("Epoch %d started", epoch)

                self.solver_half0.zero_accum()
                self.solver_half1.zero_accum()

                # half 0
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                half0_wall_start = time.perf_counter()

                logger.info("Epoch %d Half 0 started", epoch)
                self.state.progress.half = 0
                log_state(logger, self.state, title=f"Epoch {epoch} Half 0 State")
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
                log_state(logger, self.state, title=f"Epoch {epoch} Half 1 State")
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

                self.solver_half1.update()

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                half1_wall = time.perf_counter() - half1_wall_start
                logger.info(
                    "Epoch %d Half 1 finished | time=%.3fs",
                    epoch,
                    half1_wall,
                )

                # real-domain volume
                self.volume_real_half0 = self.volume_half0.volume_real
                self.volume_real_half1 = self.volume_half1.volume_real

                # half-map fsc
                self.evaluate()

                # external reconstruction
                self.external_reconstruct()

                # update prior
                self.update_prior()

                # epoch summary
                if is_rank0():
                    log_block(
                        logger,
                        title=f"Epoch {epoch} Summary",
                        lines=[
                            f"Pose Search : L={self.state.schedule.side_length}, healpix={self.state.schedule.healpix_order}, oversampling={self.state.schedule.oversampling}, trans_extent={self.state.schedule.trans_grid_extent:.2f}",
                            f"Backproject : full_bp={self.state.schedule.full_backprojection}",
                            f"Resolution  : {float(self.state.metrics.fsc_resolution):.2f} Angstrom",
                            f"Trans RMS   : {self.state.metrics.trans_update_rms:.2e}",
                            f"Confidence  : {self.state.metrics.avg_confidence:.2e}",
                        ],
                    )

                self.scheduler.step()

                if self.state.progress.has_converged and is_rank0():
                    log_block(
                        logger,
                        title=f"Converged At Epoch {epoch}",
                        lines=[
                            "fsc_resolution has shown no meaningful gain for "
                            f"{self.state.progress.num_epochs_without_resolution_gain} consecutive epochs "
                            f"(resolution={float(self.state.metrics.fsc_resolution):.4f} A)",
                        ],
                    )

                # next epoch plan
                if is_rank0() and not self.state.progress.has_converged:
                    log_block(
                        logger,
                        title=f"Next Epoch Configuration (Epoch {epoch+1})",
                        lines=[
                            f"Pose Search : L={self.state.schedule.side_length}, healpix={self.state.schedule.healpix_order}, oversampling={self.state.schedule.oversampling}, trans_extent={self.state.schedule.trans_grid_extent:.2f}",
                            f"Backproject : full_bp={self.state.schedule.full_backprojection}",
                            f"Scope       : {self.state.schedule.pose_search_scope}",
                            f"Strategy    : {self.state.schedule.pose_search_strategy}",
                            f"Cache       : {self.state.schedule.proj_cache_backend}",
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

                if self.state.progress.has_converged:
                    logger.info(
                        "Refinement stopped early at epoch %d | reason=%s",
                        epoch,
                        "fsc_resolution has shown no meaningful gain for "
                        f"{self.state.progress.num_epochs_without_resolution_gain} consecutive epochs "
                        f"(resolution={float(self.state.metrics.fsc_resolution):.4f} A)",
                    )
                    break

            if not self.state.progress.has_converged and is_rank0():
                log_block(
                    logger,
                    title="Refinement Completed",
                    lines=[
                        f"Completed all {int(self.config.refinement.num_epochs)} epochs",
                        f"Final resolution : {float(self.state.metrics.fsc_resolution):.4f} Angstrom",
                    ],
                )

        except Exception:
            logger.exception("Refinement failed")
            raise