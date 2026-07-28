# optim/state.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass, fields
from typing import Any, Dict, Literal, cast

from cryoseed.config import MainConfig


PoseSearchCriterion = Literal["posterior", "correlation"]
PoseTranslationCenterMode = Literal["auto", "always", "never"]


def parse_pose_search_criterion(value: str) -> PoseSearchCriterion:
    """Validate and narrow a pose-search criterion string."""
    if value not in ("posterior", "correlation"):
        raise ValueError(f"Unsupported pose_search_criterion: {value!r}")
    return cast(PoseSearchCriterion, value)


def parse_pose_translation_center_mode(value: str) -> PoseTranslationCenterMode:
    """Validate and narrow a pose-translation center mode string."""
    if value not in ("auto", "always", "never"):
        raise ValueError(f"Unsupported pose_translation_center_mode: {value!r}")
    return cast(PoseTranslationCenterMode, value)


# -------------------------
# Small utilities
# -------------------------
def _to_builtin(obj: Any) -> Any:
    """Convert dataclass / dict / list structures to builtin Python types.
    Note: tensors/np arrays should be converted by callers (or handled in manager).
    """
    if is_dataclass(obj):
        return {k: _to_builtin(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    return obj


# -------------------------
# Namespaces
# -------------------------
@dataclass
class ProgressState:
    """Written by engine only."""
    epoch: int = 0
    half: int | None = None  # current half; None means the current phase has no half concept
    iter: int = 0  # completed batch count within the current phase/half
    num_epochs_without_resolution_gain: int = 0
    num_epochs_with_small_trans_update: int = 0
    num_checks_with_stable_side_length: int = 0
    num_checks_with_stable_pose: int = 0
    num_checks_ready_to_stop: int = 0
    has_converged: bool = False


@dataclass
class ScheduleState:
    """Written by scheduler only.

    ``pose_search_criterion`` currently accepts:
    - ``"posterior"``: full posterior route with probabilistic weighting
    - ``"correlation"``: first-epoch correlation route implemented via NCC
    """
    pose_search_scope: str = "global"
    pose_search_strategy: str = "healpix"
    pose_search_criterion: PoseSearchCriterion = "posterior"
    healpix_order: int = 2
    oversampling: int = 1
    side_length: int = 32
    trans_grid_extent: float = 5.0
    trans_grid_samples: int = 5
    pose_translation_center_mode: PoseTranslationCenterMode = "auto"
    use_pose_translation_as_center: bool = True
    use_particle_mask: bool = False
    particle_mask_extra_diameter_angstrom: float = 0.0
    proj_cache_backend: str = "none"
    initial_healpix_alignment_done: bool = False
    healpix_terminal_reached: bool = False
    is_final_epoch: bool = False
    activate_learning_rate_decay: bool = False
    full_backprojection: bool = False
    skip_external_reconstruct: bool = False

    def __post_init__(self) -> None:
        self.pose_search_criterion = parse_pose_search_criterion(
            self.pose_search_criterion
        )
        self.pose_translation_center_mode = parse_pose_translation_center_mode(
            self.pose_translation_center_mode
        )


@dataclass
class MetricsState:
    """Written by solver/logger; scheduler reads it."""
    ema_loss: float | None = None
    ema_loss_change: float | None = None

    confidence_sum: float = 0.0
    confidence_count: int = 0
    volume_class_confidence_sum: float = 0.0
    volume_class_confidence_count: int = 0
    volume_class_change_rate: float = 0.0
    ema_volume_class_change_rate: float | None = None

    rot_update_rms: float = 0.0
    trans_update_rms: float = 0.0
    ema_rot_update_rms: float | None = None
    ema_trans_update_rms: float | None = None

    relative_volume_change: float | None = None

    side_length_resolution: float | None = None
    fsc_scores: Any | None = None
    fsc_resolution: float | None = None
    fsc_resolution_change: float | None = None

    @property
    def avg_confidence(self) -> float:
        if self.confidence_count == 0:
            return 0.0
        return self.confidence_sum / self.confidence_count

    @property
    def avg_volume_class_confidence(self) -> float:
        if self.volume_class_confidence_count == 0:
            return 0.0
        return self.volume_class_confidence_sum / self.volume_class_confidence_count


@dataclass
class OptimState:
    """Shared state for runner + solvers + scheduler."""

    version: int = 1

    progress: ProgressState = field(default_factory=ProgressState)
    schedule: ScheduleState = field(default_factory=ScheduleState)
    metrics: MetricsState = field(default_factory=MetricsState)

    def to_dict(self) -> Dict[str, Any]:
        return _to_builtin(self)

    @staticmethod
    def _filter_dataclass_kwargs(cls, data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return {}
        allowed = {f.name for f in fields(cls)}
        return {k: v for k, v in data.items() if k in allowed}

    @staticmethod
    def _coerce_int_fields(data: Dict[str, Any], *keys: str) -> Dict[str, Any]:
        for k in keys:
            if k in data and data[k] is not None:
                data[k] = int(data[k])
        return data

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OptimState":
        st = cls(version=int(d.get("version", 1)))

        progress_kwargs = cls._filter_dataclass_kwargs(ProgressState, d.get("progress", {}))
        cls._coerce_int_fields(
            progress_kwargs,
            "epoch",
            "half",
            "iter",
            "num_epochs_without_resolution_gain",
            "num_epochs_with_small_trans_update",
            "num_checks_with_stable_side_length",
            "num_checks_with_stable_pose",
            "num_checks_ready_to_stop",
        )
        st.progress = ProgressState(**progress_kwargs)

        sched_kwargs = cls._filter_dataclass_kwargs(ScheduleState, d.get("schedule", {}))
        cls._coerce_int_fields(
            sched_kwargs,
            "healpix_order",
            "oversampling",
            "side_length",
            "trans_grid_samples",
        )
        st.schedule = ScheduleState(**sched_kwargs)

        metrics_kwargs = cls._filter_dataclass_kwargs(MetricsState, d.get("metrics", {}))
        st.metrics = MetricsState(**metrics_kwargs)
        return st

    @classmethod
    def from_config(cls, config: MainConfig) -> "OptimState":
        st = cls()
        st.schedule.healpix_order = int(config.pose_search.init_healpix_order)
        init_trans_grid_extent = config.pose_search.init_trans_grid_extent
        if init_trans_grid_extent is None:
            if int(config.data.image_size) <= 0:
                raise ValueError(
                    "pose_search.init_trans_grid_extent cannot be auto-derived when "
                    "data.image_size <= 0"
                )
            init_trans_grid_extent = float(int(config.data.image_size) // 2)
        st.schedule.trans_grid_extent = float(init_trans_grid_extent)
        st.schedule.trans_grid_samples = int(config.pose_search.trans_grid_samples)
        st.schedule.pose_translation_center_mode = parse_pose_translation_center_mode(
            str(config.scheduler.pose_translation_center_mode)
        )
        st.schedule.use_pose_translation_as_center = True
        st.schedule.use_particle_mask = (
            bool(config.data.particle_mask.enabled)
            and int(st.progress.epoch) >= int(config.data.particle_mask.protection_disable_epochs)
        )
        st.schedule.particle_mask_extra_diameter_angstrom = 0.0
        st.schedule.proj_cache_backend = "memory" if bool(config.scheduler.use_cache) else "none"
        st.schedule.full_backprojection = bool(config.reconstruction.full_backprojection)
        st.schedule.pose_search_criterion = parse_pose_search_criterion(
            "posterior"
        )

        return st