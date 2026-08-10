import math
from dataclasses import dataclass

import torch

from cryoseed.data import DataBatch
from cryoseed.ops.transforms import backproject
from cryoseed.state import OptimState

from cryoseed.modules.volume import Volume, VoxelGrid
from cryoseed.modules.statistics import NoiseVariance, PriorVariance

from .pose import PoseSearcher
from .solver import Solver, SolverInferResult

@dataclass(slots=True)
class SGDInferResult(SolverInferResult):
    loss: torch.Tensor
    prob: torch.Tensor
    prob2img_idx: torch.Tensor
    prob2vol_idx: torch.Tensor
    rotmat: torch.Tensor
    trans: torch.Tensor
    radial_residual_power: torch.Tensor | None = None


class _Curvature:
    def __init__(self, volume: VoxelGrid):
        self.grid_size = int(volume.grid_size)
        self.device_mesh = volume.device_mesh
        self.backproject_chunk = int(volume.backproject_chunk)
        self.curvature = torch.zeros_like(volume.volume.real, dtype=torch.float32)
        self.accum_numer = torch.zeros_like(volume.volume.real, dtype=torch.float32)
        self.accum_denom = torch.zeros(
            1,
            dtype=torch.float32,
            device=volume.volume.device,
        )

    @torch.no_grad()
    def zero_accum(self) -> None:
        self.curvature.zero_()
        self.accum_numer.zero_()
        self.accum_denom.zero_()

    @torch.no_grad()
    def accumulate(
        self,
        *,
        ctf: torch.Tensor | None,
        probability: torch.Tensor,
        image_index: torch.Tensor | None,
        volume_index: torch.Tensor | None,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        noise_spectrum: torch.Tensor | None,
        radius: float | None,
    ) -> None:
        device = rotation.device
        if ctf is not None:
            ctf = ctf.to(device=device)
            B = int(ctf.shape[0])
            L = int(ctf.shape[-1])
        else:
            if image_index is None:
                B = int(rotation.shape[0])
            else:
                B = int(image_index.max().item()) + 1
            L = self.grid_size

        if noise_spectrum is None:
            noise_spectrum = torch.ones((L, L), device=device, dtype=torch.float32)
        else:
            noise_spectrum = noise_spectrum.to(device=device, dtype=torch.float32)

        if radius is None:
            radius = float(L // 2)

        dummy_image = torch.zeros((B, L, L), device=device, dtype=torch.complex64)
        local_accum_numer = torch.zeros_like(self.accum_numer)

        dist = getattr(torch, "distributed", None)
        if (
            dist is not None
            and dist.is_available()
            and dist.is_initialized()
            and self.device_mesh is not None
        ):
            group = self.device_mesh.get_group(1)
            calculation_parallel_size = dist.get_world_size(group)
            calculation_process_rank = dist.get_rank(group)
        else:
            group = None
            calculation_parallel_size = 1
            calculation_process_rank = 0

        num_pose = int(rotation.shape[0])
        slice_size = math.ceil(num_pose / calculation_parallel_size)
        slice_start = slice_size * calculation_process_rank
        slice_end = min(slice_size * (calculation_process_rank + 1), num_pose)

        for chunk_start in range(slice_start, slice_end, self.backproject_chunk):
            chunk_end = min(chunk_start + self.backproject_chunk, slice_end)
            backproject(
                image=dummy_image,
                ctf=ctf,
                noise_spectrum=noise_spectrum,
                probability=probability[chunk_start:chunk_end],
                image_index=(
                    image_index[chunk_start:chunk_end] if image_index is not None else None
                ),
                volume_index=(
                    volume_index[chunk_start:chunk_end] if volume_index is not None else None
                ),
                rotation=rotation[chunk_start:chunk_end],
                translation=translation[chunk_start:chunk_end],
                radius=radius,
                volume_numerator=None,
                volume_denominator=local_accum_numer,
                return_denom=True,
            )

        if (
            dist is not None
            and dist.is_available()
            and dist.is_initialized()
            and calculation_parallel_size > 1
        ):
            dist.all_reduce(local_accum_numer, op=dist.ReduceOp.SUM, group=group)

        self.accum_numer += local_accum_numer
        self.accum_denom += float(B)

    @torch.no_grad()
    def update(self) -> None:
        curvature = self.curvature
        numer = self.accum_numer
        denom = self.accum_denom

        dist = getattr(torch, "distributed", None)
        if dist is not None and dist.is_available() and dist.is_initialized():
            if self.device_mesh is not None:
                group = (
                    self.device_mesh.get_group(0)
                    if hasattr(self.device_mesh, "get_group")
                    else self.device_mesh
                )
            else:
                group = dist.group.WORLD

            data_parallel_size = dist.get_world_size(group=group)
            if data_parallel_size > 1:
                numer = numer.clone()
                denom = denom.clone()
                dist.all_reduce(numer, op=dist.ReduceOp.SUM, group=group)
                dist.all_reduce(denom, op=dist.ReduceOp.SUM, group=group)

        curvature.copy_(numer / denom.clamp_min(1.0))
        self.curvature.copy_(curvature)

    def amax(self, *args, **kwargs) -> torch.Tensor:
        return self.curvature.amax(*args, **kwargs)

class SGDSolver(Solver):
    def __init__(
        self,
        *,
        state: OptimState,
        pose_searcher: PoseSearcher,
        prior=None,
    ):
        self.state: OptimState = state
        self.volume: Volume = pose_searcher.volume
        self.noise: NoiseVariance | None = pose_searcher.noise
        self.prior: PriorVariance | None = prior
        self.pose = pose_searcher.pose
        self.pose_searcher: PoseSearcher = pose_searcher

        if not isinstance(self.volume, VoxelGrid):
            raise TypeError(
                "SGDSolver curvature scaling currently requires VoxelGrid, "
                f"got {type(self.volume).__name__}"
            )
        self.curvature = _Curvature(self.volume)

        if not any(param.requires_grad for param in self.volume.parameters()):
            raise ValueError(
                "SGDSolver requires a differentiable volume with at least one "
                "parameter where requires_grad=True"
            )
        if self.noise is not None and not bool(getattr(self.noise, "requires_accum", False)):
            raise ValueError(
                "SGDSolver requires noise.requires_accum=True when noise estimation is enabled"
            )

        if self.volume.optimizer is None:
            raise ValueError("SGDSolver requires a configured volume optimizer")

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Return the optimizer owned by the volume."""
        optimizer = self.volume.optimizer
        if optimizer is None:
            raise RuntimeError("SGDSolver requires a configured volume optimizer")
        return optimizer

    def refresh(self):
        self.pose_searcher.refresh()
        self._validate_schedule_constraints()
        self.volume.reset_lr()

    def _validate_schedule_constraints(self) -> None:
        schedule = self.state.schedule
        if schedule.pose_search_criterion != "posterior":
            raise ValueError(
                "SGDSolver currently supports only pose_search_criterion='posterior', "
                f"got {schedule.pose_search_criterion!r}"
            )
        if int(schedule.oversampling) != 0:
            raise ValueError(
                "SGDSolver currently supports only oversampling=0, "
                f"got {int(schedule.oversampling)}"
            )

    def infer(self, batch: DataBatch) -> SGDInferResult:
        self._validate_schedule_constraints()
        image = batch.image.to(self.pose_searcher.device)
        particle_index = batch.particle_index.to(
            device=self.pose_searcher.device, dtype=torch.long
        )
        ctf = batch.ctf

        (
            loss,
            prob,
            prob2img_idx,
            prob2vol_idx,
            rotmat,
            trans,
            radial_residual_power,
        ) = self.pose_searcher.search_grad(
            image,
            particle_index=particle_index,
            ctf=ctf,
        )
        return SGDInferResult(
            image=image,
            ctf=ctf,
            loss=loss,
            prob=prob,
            prob2img_idx=prob2img_idx,
            prob2vol_idx=prob2vol_idx,
            rotmat=rotmat,
            trans=trans,
            radial_residual_power=radial_residual_power,
        )

    def accumulate(self, result: SGDInferResult) -> None:
        self.volume.backward(result.loss)

        if self.noise is not None:
            if result.radial_residual_power is None:
                raise RuntimeError(
                    "radial_residual_power must be present when noise estimation is enabled"
                )
            self.noise.accumulate(
                ctf=result.ctf,
                probability=result.prob,
                image_index=result.prob2img_idx,
                radial_residual_power=result.radial_residual_power,
                num_images=int(result.image.shape[0]),
                side_length=int(self.state.schedule.side_length),
            )
            noise_spectrum = self.noise.variance_spectrum(ndim=2)
        else:
            noise_spectrum = None

        radius = None
        if not bool(self.state.schedule.full_backprojection):
            radius = float(int(self.state.schedule.side_length) // 2)

        self.curvature.accumulate(
            ctf=result.ctf,
            probability=result.prob,
            image_index=result.prob2img_idx,
            volume_index=result.prob2vol_idx,
            rotation=result.rotmat,
            translation=result.trans,
            noise_spectrum=noise_spectrum,
            radius=radius,
        )

    def update(self) -> None:
        self.curvature.update()

        self.volume.sync_grad_()
        grad = self.volume.volume.grad
        if grad is not None:
            step_size = 1.0 / self.curvature.amax(dim=(1, 2, 3)).clamp_min(1e-6)
            grad.mul_(step_size[:, None, None, None])

        self.volume.update()
        if bool(self.state.abinitio.solver.activate_learning_rate_decay):
            self.volume.update_lr()

        if self.noise is not None:
            self.noise.update()

        self.pose.update()

    def zero(self) -> None:
        """Clear gradients and accumulated statistics for the next update."""
        self.curvature.zero_accum()
        self.volume.zero_grad(set_to_none=True)
        self.pose.zero_accum()
        if self.noise is not None:
            self.noise.zero_accum()