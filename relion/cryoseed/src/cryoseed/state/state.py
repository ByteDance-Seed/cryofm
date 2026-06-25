# optim/state.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass, fields
from typing import Any, Dict

from cryoseed.config import MainConfig


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
    half: int = 0  # current half
    num_epochs_without_resolution_gain: int = 0
    has_converged: bool = False


@dataclass
class ScheduleState:
    """Written by scheduler only"""
    pose_search_scope: str = "global"
    pose_search_strategy: str = "healpix"
    healpix_order: int = 2
    oversampling: int = 1
    side_length: int = 32
    trans_grid_extent: float = 5.0
    proj_cache_backend: str = "none"
    full_backprojection: bool = False


@dataclass
class MetricsState:
    """Written by solver/logger; scheduler reads it."""
    fsc_scores: Any | None = None
    fsc_resolution: float | None = None
    fsc_resolution_change: float | None = None
    healpix_order_from_resolution: int = 0
    trans_update_rms: float = 0.0

    confidence_sum: float = 0.0
    confidence_count: int = 0

    @property
    def avg_confidence(self) -> float:
        if self.confidence_count == 0:
            return 0.0
        return self.confidence_sum / self.confidence_count


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
            "num_epochs_without_resolution_gain",
        )
        st.progress = ProgressState(**progress_kwargs)

        sched_kwargs = cls._filter_dataclass_kwargs(ScheduleState, d.get("schedule", {}))
        cls._coerce_int_fields(
            sched_kwargs,
            "healpix_order",
            "oversampling",
            "side_length",
        )
        st.schedule = ScheduleState(**sched_kwargs)

        metrics_kwargs = cls._filter_dataclass_kwargs(MetricsState, d.get("metrics", {}))
        cls._coerce_int_fields(metrics_kwargs, "healpix_order_from_resolution")
        st.metrics = MetricsState(**metrics_kwargs)
        return st

    @classmethod
    def from_config(cls, config: MainConfig) -> "OptimState":
        st = cls()
        st.schedule.healpix_order = int(config.pose_search.init_healpix_order)
        st.schedule.trans_grid_extent = float(config.pose_search.init_trans_grid_extent)
        st.schedule.proj_cache_backend = "memory" if bool(config.scheduler.use_cache) else "none"
        st.schedule.full_backprojection = bool(config.reconstruction.full_backprojection)

        return st