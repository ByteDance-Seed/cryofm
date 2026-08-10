from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, Literal, cast

from cryoseed.config import MainConfig


PoseSearchCriterion = Literal["posterior", "correlation"]
PoseTranslationCenterMode = Literal["auto", "always", "never"]
SearchGradMode = Literal["full", "selected"]
CommandName = Literal["abinitio", "heterorefine", "homorefine"]
_USE_ACTIVE_COMMAND = object()


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


def parse_search_grad_mode(value: str) -> SearchGradMode:
    """Validate and narrow a differentiable search-route selector."""
    if value not in ("full", "selected"):
        raise ValueError(f"Unsupported search_grad_mode: {value!r}")
    return cast(SearchGradMode, value)


def parse_command_name(value: str | None) -> CommandName | None:
    """Validate and narrow an optional command string."""
    if value is None:
        return None
    if value not in ("abinitio", "heterorefine", "homorefine"):
        raise ValueError(f"Unknown command: {value}")
    return cast(CommandName, value)


def _to_builtin(obj: Any) -> Any:
    """Convert dataclass / dict / list structures to builtin Python types."""
    if is_dataclass(obj):
        return {k: _to_builtin(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    return obj


@dataclass
class ProgressState:
    """Written by engine only."""

    epoch: int = 0
    half: int | None = None
    iter: int = 0


@dataclass
class ScheduleState:
    """Shared execution settings consumed by multiple optim components."""

    pose_search_scope: str = "global"
    pose_search_strategy: str = "healpix"
    pose_search_criterion: PoseSearchCriterion = "posterior"
    search_grad_mode: SearchGradMode = "full"
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
    full_backprojection: bool = False

    def __post_init__(self) -> None:
        self.pose_search_criterion = parse_pose_search_criterion(
            self.pose_search_criterion
        )
        self.search_grad_mode = parse_search_grad_mode(self.search_grad_mode)
        self.pose_translation_center_mode = parse_pose_translation_center_mode(
            self.pose_translation_center_mode
        )


@dataclass
class AbInitioMetricsState:
    avg_confidence: float = 0.0
    avg_volume_class_confidence: float = 0.0
    volume_class_change_rate: float = 0.0
    ema_volume_class_change_rate: float | None = None
    rot_update_rms: float = 0.0
    trans_update_rms: float = 0.0
    ema_rot_update_rms: float | None = None
    ema_trans_update_rms: float | None = None
    side_length_resolution: float | None = None


@dataclass
class AbInitioEngineState:
    is_final_epoch: bool = False
    skip_external_reconstruct: bool = False


@dataclass
class AbInitioSolverState:
    activate_learning_rate_decay: bool = False


@dataclass
class AbInitioSchedulerState:
    num_checks_with_stable_side_length: int = 0
    num_checks_with_stable_pose: int = 0
    num_checks_ready_to_stop: int = 0
    has_converged: bool = False
    initial_healpix_alignment_done: bool = False
    healpix_terminal_reached: bool = False


@dataclass
class AbInitioState:
    engine: AbInitioEngineState = field(default_factory=AbInitioEngineState)
    solver: AbInitioSolverState = field(default_factory=AbInitioSolverState)
    scheduler: AbInitioSchedulerState = field(default_factory=AbInitioSchedulerState)
    metrics: AbInitioMetricsState = field(default_factory=AbInitioMetricsState)


@dataclass
class HeteroRefineEngineState:
    is_bootstrap_epoch: bool = False


@dataclass
class HeteroRefineSchedulerState:
    pass


@dataclass
class HeteroRefineMetricsState:
    avg_confidence: float = 0.0
    avg_volume_class_confidence: float = 0.0
    volume_class_change_rate: float = 0.0
    rot_update_rms: float = 0.0
    trans_update_rms: float = 0.0
    volume_occupancy: Any | None = None
    dvp_crossing_radius: Any | None = None
    dvp_resolution_per_volume: Any | None = None
    dvp_radius: float | None = None
    dvp_resolution: float | None = None
    side_length_resolution: float | None = None


@dataclass
class HeteroRefineState:
    engine: HeteroRefineEngineState = field(default_factory=HeteroRefineEngineState)
    scheduler: HeteroRefineSchedulerState = field(
        default_factory=HeteroRefineSchedulerState
    )
    metrics: HeteroRefineMetricsState = field(default_factory=HeteroRefineMetricsState)


@dataclass
class HomoRefineEngineState:
    is_final_epoch: bool = False
    skip_external_reconstruct: bool = False


@dataclass
class HomoRefineSchedulerState:
    num_epochs_without_resolution_gain: int = 0
    num_epochs_with_small_trans_update: int = 0
    has_converged: bool = False


@dataclass
class HomoRefineMetricsState:
    avg_confidence: float = 0.0
    avg_volume_class_confidence: float = 0.0
    rot_update_rms: float = 0.0
    trans_update_rms: float = 0.0
    fsc_scores: Any | None = None
    fsc_resolution: float | None = None
    fsc_resolution_change: float | None = None


@dataclass
class HomoRefineState:
    engine: HomoRefineEngineState = field(default_factory=HomoRefineEngineState)
    scheduler: HomoRefineSchedulerState = field(default_factory=HomoRefineSchedulerState)
    metrics: HomoRefineMetricsState = field(default_factory=HomoRefineMetricsState)


@dataclass
class OptimState:
    """Shared state for runner + solvers + scheduler."""

    progress: ProgressState = field(default_factory=ProgressState)
    schedule: ScheduleState = field(default_factory=ScheduleState)
    abinitio: AbInitioState = field(default_factory=AbInitioState)
    heterorefine: HeteroRefineState = field(default_factory=HeteroRefineState)
    homorefine: HomoRefineState = field(default_factory=HomoRefineState)
    _active_command: CommandName | None = field(default=None, repr=False)

    def _to_public_dict(self, command: CommandName | None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "progress": _to_builtin(self.progress),
            "schedule": _to_builtin(self.schedule),
        }
        if command in (None, "abinitio"):
            payload["abinitio"] = _to_builtin(self.abinitio)
        if command in (None, "heterorefine"):
            payload["heterorefine"] = _to_builtin(self.heterorefine)
        if command in (None, "homorefine"):
            payload["homorefine"] = _to_builtin(self.homorefine)
        return payload

    def to_dict(self, *, command: object = _USE_ACTIVE_COMMAND) -> Dict[str, Any]:
        effective_command = (
            self._active_command
            if command is _USE_ACTIVE_COMMAND
            else parse_command_name(cast(str | None, command))
        )
        return self._to_public_dict(effective_command)

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
        st = cls()

        progress_kwargs = cls._filter_dataclass_kwargs(
            ProgressState, d.get("progress", {})
        )
        cls._coerce_int_fields(progress_kwargs, "epoch", "half", "iter")
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

        abinitio_kwargs = cls._filter_dataclass_kwargs(
            AbInitioState, d.get("abinitio", {})
        )
        if "engine" in abinitio_kwargs:
            abinitio_kwargs["engine"] = AbInitioEngineState(
                **cls._filter_dataclass_kwargs(
                    AbInitioEngineState, abinitio_kwargs["engine"]
                )
            )
        if "solver" in abinitio_kwargs:
            abinitio_kwargs["solver"] = AbInitioSolverState(
                **cls._filter_dataclass_kwargs(
                    AbInitioSolverState, abinitio_kwargs["solver"]
                )
            )
        if "scheduler" in abinitio_kwargs:
            sched = cls._filter_dataclass_kwargs(
                AbInitioSchedulerState, abinitio_kwargs["scheduler"]
            )
            cls._coerce_int_fields(
                sched,
                "num_checks_with_stable_side_length",
                "num_checks_with_stable_pose",
                "num_checks_ready_to_stop",
            )
            abinitio_kwargs["scheduler"] = AbInitioSchedulerState(**sched)
        if "metrics" in abinitio_kwargs:
            abinitio_kwargs["metrics"] = AbInitioMetricsState(
                **cls._filter_dataclass_kwargs(
                    AbInitioMetricsState, abinitio_kwargs["metrics"]
                )
            )
        st.abinitio = AbInitioState(**abinitio_kwargs)

        heterorefine_kwargs = cls._filter_dataclass_kwargs(
            HeteroRefineState, d.get("heterorefine", {})
        )
        if "engine" in heterorefine_kwargs:
            heterorefine_kwargs["engine"] = HeteroRefineEngineState(
                **cls._filter_dataclass_kwargs(
                    HeteroRefineEngineState, heterorefine_kwargs["engine"]
                )
            )
        if "scheduler" in heterorefine_kwargs:
            heterorefine_kwargs["scheduler"] = HeteroRefineSchedulerState(
                **cls._filter_dataclass_kwargs(
                    HeteroRefineSchedulerState, heterorefine_kwargs["scheduler"]
                )
            )
        if "metrics" in heterorefine_kwargs:
            heterorefine_kwargs["metrics"] = HeteroRefineMetricsState(
                **cls._filter_dataclass_kwargs(
                    HeteroRefineMetricsState, heterorefine_kwargs["metrics"]
                )
            )
        st.heterorefine = HeteroRefineState(**heterorefine_kwargs)

        homorefine_kwargs = cls._filter_dataclass_kwargs(
            HomoRefineState, d.get("homorefine", {})
        )
        if "engine" in homorefine_kwargs:
            homorefine_kwargs["engine"] = HomoRefineEngineState(
                **cls._filter_dataclass_kwargs(
                    HomoRefineEngineState, homorefine_kwargs["engine"]
                )
            )
        if "scheduler" in homorefine_kwargs:
            sched = cls._filter_dataclass_kwargs(
                HomoRefineSchedulerState, homorefine_kwargs["scheduler"]
            )
            cls._coerce_int_fields(
                sched,
                "num_epochs_without_resolution_gain",
                "num_epochs_with_small_trans_update",
            )
            homorefine_kwargs["scheduler"] = HomoRefineSchedulerState(**sched)
        if "metrics" in homorefine_kwargs:
            homorefine_kwargs["metrics"] = HomoRefineMetricsState(
                **cls._filter_dataclass_kwargs(
                    HomoRefineMetricsState, homorefine_kwargs["metrics"]
                )
            )
        st.homorefine = HomoRefineState(**homorefine_kwargs)
        st._active_command = parse_command_name(d.get("active_command"))

        return st

    @classmethod
    def from_config(
        cls, config: MainConfig, *, command: CommandName | None = None
    ) -> "OptimState":
        normalized_command = parse_command_name(command)
        st = cls(_active_command=normalized_command)
        if normalized_command == "abinitio":
            scheduler_config = config.abinitio.scheduler
            use_cache = False
        elif normalized_command == "heterorefine":
            scheduler_config = config.heterorefine.scheduler
            use_cache = False
        elif normalized_command == "homorefine":
            scheduler_config = config.homorefine.scheduler
            use_cache = bool(config.homorefine.scheduler.use_cache)
        else:
            scheduler_config = None
            use_cache = False

        st.schedule.healpix_order = int(config.modules.search.init_healpix_order)
        if normalized_command == "heterorefine":
            st.schedule.pose_search_scope = "global"
            st.schedule.pose_search_strategy = "healpix"
            st.schedule.oversampling = 0
            st.schedule.search_grad_mode = "full"
        init_trans_grid_extent = config.modules.search.init_trans_grid_extent
        if init_trans_grid_extent is None:
            if int(config.data.image_size) <= 0:
                raise ValueError(
                    "modules.search.init_trans_grid_extent cannot be auto-derived when "
                    "data.image_size <= 0"
                )
            init_trans_grid_extent = float(int(config.data.image_size) // 2)
        st.schedule.trans_grid_extent = float(init_trans_grid_extent)
        st.schedule.trans_grid_samples = int(config.modules.search.trans_grid_samples)
        if scheduler_config is not None:
            st.schedule.pose_translation_center_mode = parse_pose_translation_center_mode(
                str(scheduler_config.pose_translation_center_mode)
            )
        st.schedule.use_pose_translation_as_center = True
        st.schedule.use_particle_mask = (
            bool(config.modules.search.particle_mask.enabled)
            and int(st.progress.epoch)
            >= int(config.modules.search.particle_mask.protection_disable_epochs)
        )
        st.schedule.particle_mask_extra_diameter_angstrom = 0.0
        st.schedule.proj_cache_backend = "memory" if use_cache else "none"
        st.schedule.full_backprojection = bool(
            config.modules.volume.voxel.full_backprojection
        )
        st.schedule.pose_search_criterion = parse_pose_search_criterion("posterior")
        return st