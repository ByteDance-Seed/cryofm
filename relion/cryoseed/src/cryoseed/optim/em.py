import torch

from dataclasses import dataclass

from .solver import Solver, SolverInferResult
from cryoseed.data import DataBatch
from cryoseed.modules.volume import Volume
from cryoseed.modules.statistics import NoiseVariance, PriorVariance
from cryoseed.modules.pose import Pose
from .pose import PoseSearcher
from cryoseed.state import OptimState

@dataclass(slots=True)
class EMInferResult(SolverInferResult):
    prob: torch.Tensor
    prob2img_idx: torch.Tensor
    prob2vol_idx: torch.Tensor
    rotmat: torch.Tensor
    trans: torch.Tensor
    radial_residual_power: torch.Tensor | None = None

class EMSolver(Solver):
    def __init__(self, *, state: OptimState, pose_searcher: PoseSearcher, prior: PriorVariance | None = None):
        self.state: OptimState = state
        self.volume: Volume = pose_searcher.volume
        self.noise: NoiseVariance | None = pose_searcher.noise
        self.prior: PriorVariance | None = prior
        self.pose: Pose = pose_searcher.pose
        self.pose_searcher: PoseSearcher = pose_searcher

        if not bool(getattr(self.volume, "requires_accum", False)):
            raise ValueError(
                "EMSolver requires volume.requires_accum=True for backprojection/update"
            )
        if self.noise is not None and not bool(getattr(self.noise, "requires_accum", False)):
            raise ValueError(
                "EMSolver requires noise.requires_accum=True when noise estimation is enabled"
            )

    def refresh(self):
        self.pose_searcher.refresh()

    def expectation(
        self,
        image,
        *,
        particle_index,
        ctf=None,
        fixed_volume_index: torch.LongTensor | None = None,
    ):
        prob, prob2img_idx, prob2vol_idx, rotmat, trans, radial_residual_power = self.pose_searcher.search(
            image,
            particle_index=particle_index,
            ctf=ctf,
            mode="no_grad",
            fixed_volume_index=fixed_volume_index,
        )
        return prob, prob2img_idx, prob2vol_idx, rotmat, trans, radial_residual_power

    def maximization(
        self,
        image,
        ctf=None,
        *,
        prob,
        prob2img_idx,
        prob2vol_idx,
        rotmat,
        trans,
        radial_residual_power=None,
    ):
        posterior_search = (
            self.state.schedule.pose_search_criterion == "posterior"
        )
        radius = None
        if not posterior_search or not self.state.schedule.full_backprojection:
            radius = int(self.state.schedule.side_length) // 2
        if self.noise is not None:
            noise_spectrum = self.noise.variance_spectrum(ndim=2)
        else:
            noise_spectrum = None
        self.volume.backproject(
            image,
            ctf,
            prob,
            prob2img_idx,
            prob2vol_idx,
            rotmat,
            trans,
            noise_spectrum=noise_spectrum,
            radius=radius,
        )
        if self.noise is not None and posterior_search:
            if radial_residual_power is not None and not self.state.schedule.full_backprojection:
                self.noise.accumulate(
                    probability=prob,
                    image_index=prob2img_idx,
                    radial_residual_power=radial_residual_power,
                    num_images=int(image.shape[0]),
                )
            else:
                noise_side_length = None
                if not self.state.schedule.full_backprojection:
                    noise_side_length = int(self.state.schedule.side_length)
                self.noise.accumulate(
                    image,
                    ctf,
                    self.volume,
                    prob,
                    prob2img_idx,
                    prob2vol_idx,
                    rotmat,
                    trans,
                    side_length=noise_side_length,
                )

    def infer(
        self,
        batch: DataBatch,
        *,
        fixed_volume_index: torch.LongTensor | None = None,
    )-> EMInferResult:
        image = batch.image
        particle_index = batch.particle_index
        ctf = batch.ctf

        prob, prob2img_idx, prob2vol_idx, rotmat, trans, radial_residual_power = self.expectation(
            image,
            particle_index=particle_index,
            ctf=ctf,
            fixed_volume_index=fixed_volume_index,
        )
        result = EMInferResult(
            image=image,
            ctf=ctf,
            prob=prob,
            prob2img_idx=prob2img_idx,
            prob2vol_idx=prob2vol_idx,
            rotmat=rotmat,
            trans=trans,
            radial_residual_power=radial_residual_power,
        )
        return result

    def accumulate(self, result: EMInferResult):
        self.maximization(
            image=result.image,
            ctf=result.ctf,
            prob=result.prob,
            prob2img_idx=result.prob2img_idx,
            prob2vol_idx=result.prob2vol_idx,
            rotmat=result.rotmat,
            trans=result.trans,
            radial_residual_power=result.radial_residual_power,
        )

    def update(self):
        if self.prior is not None:
            prior_precision_spectrum = self.prior.precision_spectrum(ndim=3)
        else:
            prior_precision_spectrum = None

        self.volume.update(prior_precision_spectrum)

        if (
            self.noise is not None
            and self.state.schedule.pose_search_criterion == "posterior"
        ):
            self.noise.update()

        self.pose.update()
        self.pose_searcher.clear_memory_cache()

    def zero_accum(self):
        self.volume.zero_accum()
        if self.noise is not None:
            self.noise.zero_accum()
        self.pose.zero_accum()