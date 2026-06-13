import torch

from cryoseed.config import MainConfig
from cryoseed.utils.torch_utils import _norm_device
from cryoseed.modules.pose import Pose
from cryoseed.modules.statistics.noise import NoiseVariance
from cryoseed.modules.volume import Volume
from cryoseed.state import OptimState

from .healpix_searcher import HEALPixPoseSearcher
from .euler_searcher import EulerPoseSearcher


class PoseSearcher(torch.nn.Module):
    def __init__(
        self,
        state: OptimState,
        volume: Volume,
        pose: Pose,
        *,
        config: MainConfig,
        noise: NoiseVariance | None = None,
        device: torch.device | str | None = None,
        device_mesh=None,
    ):
        super().__init__()

        self.config = config
        self.state = state
        self.volume = volume
        self.pose = pose
        self.noise = noise

        dev = _norm_device(device)
        self.register_buffer(
            "_device_anchor",
            torch.empty(0, device=dev),
            persistent=False,
        )
        self.device_mesh = device_mesh

        self.pose_search_strategy: str | None = None
        self.pose_searcher: torch.nn.Module | None = None

        self.refresh()

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    def _build_pose_searcher(self, strategy: str) -> torch.nn.Module:
        searcher_cls = {
            "healpix": HEALPixPoseSearcher,
            "euler": EulerPoseSearcher,
        }.get(strategy)

        if searcher_cls is None:
            raise ValueError(f"Unknown pose search strategy: {strategy}")

        return searcher_cls.from_config(
            config=self.config,
            state=self.state,
            volume=self.volume,
            pose=self.pose,
            noise=self.noise,
            device=self.device,
            device_mesh=self.device_mesh,
        )

    def _reset_pose_searcher(self) -> None:
        strategy = self.state.schedule.pose_search_strategy
        self.pose_searcher = None
        self.pose_searcher = self._build_pose_searcher(strategy)
        self.pose_search_strategy = strategy

    def refresh(self) -> None:
        strategy = self.state.schedule.pose_search_strategy

        if self.pose_searcher is None or self.pose_search_strategy != strategy:
            self._reset_pose_searcher()
        else:
            self.pose_searcher.refresh()

    def clear_memory_cache(self) -> None:
        if self.pose_searcher is None:
            return
        clear_fn = getattr(self.pose_searcher, "clear_memory_cache", None)
        if callable(clear_fn):
            clear_fn()

    @torch.no_grad()
    def search(
        self,
        image,
        *,
        particle_index: torch.LongTensor,
        ctf=None,
    ):
        if self.pose_searcher is None:
            raise RuntimeError("pose_searcher is not initialized. Call refresh() first.")
        return self.pose_searcher.search(image, particle_index=particle_index, ctf=ctf)