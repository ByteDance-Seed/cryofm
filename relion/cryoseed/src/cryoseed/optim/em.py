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

class EMSolver(Solver):
    def __init__(self, *, state: OptimState, pose_searcher: PoseSearcher, prior: PriorVariance | None = None):
        self.state: OptimState = state
        self.volume: Volume = pose_searcher.volume
        self.noise: NoiseVariance | None = pose_searcher.noise
        self.prior: PriorVariance | None = prior
        self.pose: Pose = pose_searcher.pose
        self.pose_searcher: PoseSearcher = pose_searcher

    def refresh(self):
        self.pose_searcher.refresh()

    def expectation(self, image, *, particle_index, ctf=None):
        prob, prob2img_idx, prob2vol_idx, rotmat, trans = self.pose_searcher.search(image, particle_index=particle_index, ctf=ctf)
        return prob, prob2img_idx, prob2vol_idx, rotmat, trans

    def maximization(self, image, ctf=None, *, prob, prob2img_idx, prob2vol_idx, rotmat, trans):
        radius = None
        if not self.pose_searcher.config.reconstruction.full_backprojection:
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
        if self.noise is not None:
            self.noise.accumulate(image, ctf, self.volume, prob, prob2img_idx, prob2vol_idx, rotmat, trans)

    def accumulate_metrics(self, result: EMInferResult):
        prob, prob2img_idx = result.prob, result.prob2img_idx
        _, img_counts = torch.unique_consecutive(
            prob2img_idx, return_counts=True
        )

        best_src_idx = torch.cumsum(img_counts, dim=0) - img_counts
        max_prob_per_img = prob[best_src_idx]
        self.state.metrics.confidence_sum += max_prob_per_img.sum().item()
        self.state.metrics.confidence_count += len(img_counts)

    def infer(self, batch: DataBatch)-> EMInferResult:
        image = batch.image
        particle_index = batch.particle_index
        ctf = batch.ctf

        prob, prob2img_idx, prob2vol_idx, rotmat, trans = self.expectation(image, particle_index=particle_index, ctf=ctf)
        result = EMInferResult(image=image, ctf=ctf, prob=prob, prob2img_idx=prob2img_idx, prob2vol_idx=prob2vol_idx, rotmat=rotmat, trans=trans)
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
        )
        self.accumulate_metrics(result)

    def update(self):
        if self.prior is not None:
            prior_precision_spectrum = self.prior.precision_spectrum(ndim=3)
        else:
            prior_precision_spectrum = None

        self.volume.update(prior_precision_spectrum)

        if self.noise is not None:
            self.noise.update()

        self.pose.update()
        self.pose_searcher.clear_memory_cache()

    def zero_accum(self):
        self.volume.zero_accum()
        if self.noise is not None:
            self.noise.zero_accum()
        self.pose.zero_accum()