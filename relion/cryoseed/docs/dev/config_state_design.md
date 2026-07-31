## Config And State Design

This document records the current design rules for `config` and `state` in
cryoSeed. It is intended as a practical reference for future refactors, new
tasks, and new CLI entry points.

The goal is not to keep every past design idea alive. The goal is to preserve a
small set of rules that make it easy to decide where a new field should live.

### Design Goals

- Keep shared, cross-task capabilities stable as new commands are added.
- Keep task-specific runtime logic local to the task that owns it.
- Make `config` easy to override from CLI flags.
- Make `state` a communication layer for the optimization system, not a dump of
  every transient variable in the codebase.
- Prefer simple, project-native names over broad abstract labels unless the
  abstraction is genuinely useful.

### Core Separation

The project uses two different kinds of structure:

- `config` describes user-controlled inputs and persistent runtime policy.
- `state` describes shared runtime status that must be communicated across
  engines, schedulers, solvers, and related components during execution.

The two are related, but they are not mirrors of each other.

- A `config` field does not imply a `state` field.
- A `state` field does not need to be user-configurable.
- A monitoring snapshot does not need to mirror the whole `state`.

### Config Structure

The current top-level config layout is:

```python
MainConfig(
    io=...,
    data=...,
    logging=...,
    reproduce=...,
    modules=...,
    abinitio=...,
    homorefine=...,
)
```

This layout separates shared layers from task-specific layers.

#### Shared Top Level

The following sections are shared across commands:

- `io`: paths and output locations
- `data`: dataset and particle metadata
- `logging`: logging behavior
- `reproduce`: reproducibility controls
- `modules`: stable capabilities that are expected to be reused across tasks

#### Task-Specific Top Level

Each task owns its own runtime policy block:

- `abinitio.engine`
- `abinitio.solver`
- `abinitio.scheduler`
- `homorefine.engine`
- `homorefine.scheduler`

The current meaning of these sub-blocks is:

- `engine`: task runtime flow, initialization policy, and task-owned execution
  behavior
- `solver`: parameter-update behavior
- `scheduler`: schedule transitions and search/control policy

### Module-Related Config

Fields belong under `modules` when they describe a reusable capability that is
expected to stay conceptually stable even if new tasks are added.

Current examples:

- `modules.search`
- `modules.volume`
- `modules.statistics.noise`
- `modules.statistics.prior`

Concrete ownership choices already made in the codebase:

- `particle_mask` belongs to `modules.search`
- `backproject_chunk` and `full_backprojection` belong to `modules.volume`
- `accumulate_chunk` belongs to `modules.statistics.noise`
- `noise` and `prior` keep separate initialization fields such as
  `init_variance` and `precision_eps`

Use `modules` when a new task is likely to consume the same capability without
changing its core meaning.

### Task-Related Config

Fields belong under a task section when they mainly express the policy or flow
of a specific command.

Examples:

- `abinitio.scheduler.target_side_length_resolution`
- `abinitio.scheduler.auto_local_healpix_order`
- `homorefine.scheduler.first_epoch_ncc`
- `homorefine.engine.external_reconstruct`

Use a task block when the field is mainly about:

- when a task changes stage
- how a task decides convergence
- how a task schedules search granularity
- how a task runs its own special execution path

### Config Placement Rules

When adding a new config field, decide in this order:

1. Is this an input/output, data, logging, or reproducibility concern?
2. If not, is this a stable reusable capability shared by multiple tasks?
3. If not, does it belong to task runtime flow, solver behavior, or scheduler
   policy?
4. If it only matters inside one module implementation, should it stay internal
   instead of becoming config at all?

In practice:

- Put it under `modules.*` if it describes reusable capability.
- Put it under `<task>.engine` if it changes task runtime behavior.
- Put it under `<task>.solver` if it changes update logic.
- Put it under `<task>.scheduler` if it changes schedule/search/convergence
  control.
- Keep it internal if users do not need to configure it and no cross-module
  contract depends on it.

### Naming Rules For Config

Naming should stay close to current project language.

- Use `cryoSeed` as the default project name spelling. Use `CryoSeed` only when
  normal capitalization requires an uppercase initial. Do not use `cryoseed` as
  the project name in prose.
- Do not introduce broad new labels such as `workflow` or `task_state` unless
  they solve a real ambiguity.
- Prefer names that match existing code usage.
- If project style is unclear, prefer concepts that align with PyTorch-style
  meanings such as `engine`, `solver`, and `scheduler`.

### CLI Rules

Every config field should remain easy to override from CLI.

The current CLI policy is:

- Prefer the leaf field name by default
- Only add the shortest necessary prefix when leaf names would collide
- Keep flag names visually close to config field names

Examples:

- `modules.search.init_healpix_order` -> `--init-healpix-order`
- `modules.statistics.noise.enabled` -> `--noise-enabled`
- `modules.statistics.prior.enabled` -> `--prior-enabled`

Compatibility with old config layouts is intentionally not a design goal at the
current stage of the project. The project is still in internal development, so
clarity is preferred over legacy complexity.

### State Structure

The current top-level optimization state is:

```python
OptimState(
    progress=...,
    schedule=...,
    abinitio=...,
    homorefine=...,
)
```

Where task-local runtime state is nested under the task that owns it:

```python
AbInitioState(
    engine=...,
    solver=...,
    scheduler=...,
    metrics=...,
)

HomoRefineState(
    engine=...,
    scheduler=...,
    metrics=...,
)
```

### State Philosophy

`state` is an optimization-system communication tool.

A field should be in `state` only if it is meaningfully shared across runtime
components such as:

- engine
- scheduler
- solver
- checkpoint/resume boundaries

If a value does not need cross-module communication, it should usually stay as a
module attribute or another local runtime variable.

This is why several values were removed from shared state and kept internal to
engines.

Examples of values that are now engine-local instead of shared state:

- confidence accumulation buffers
- volume-confidence accumulation buffers
- EMA loss tracking used only by the engine

### Shared State Vs Local State

Use shared `state` when a value must be read or written by multiple runtime
components.

Examples:

- current search schedule in `state.schedule`
- scheduler-owned convergence counters
- task execution flags such as final-epoch status
- task metrics consumed by engine and scheduler

Keep a value local when it is:

- only used inside one engine implementation
- only used to compute a logging summary in one place
- only needed transiently within one method or one epoch

### Meaning Of The Current State Blocks

#### `progress`

`progress` should stay small. It only stores generic iteration progress:

- `epoch`
- `half`
- `iter`

It should not accumulate task-specific convergence counters or control flags.

#### `schedule`

`schedule` stores shared execution settings that affect multiple optimization
components, such as:

- pose search mode
- pose search strategy
- search resolution controls
- translation-grid controls
- particle-mask usage state
- projection cache mode
- full-backprojection flag

This block represents the current shared execution plan, not task-private
history.

#### `<task>.engine`

Use task engine state for task-owned execution flags that matter across runtime
boundaries.

Current examples:

- `is_final_epoch`
- `skip_external_reconstruct`

#### `<task>.solver`

Use task solver state for solver-owned execution toggles.

Current example:

- `abinitio.solver.activate_learning_rate_decay`

#### `<task>.scheduler`

Use task scheduler state for schedule progression and convergence bookkeeping.

Current examples:

- stable-side-length counters
- stable-pose counters
- stop counters
- convergence flags
- healpix stage bookkeeping
- homorefine convergence counters

#### `<task>.metrics`

Metrics are task-specific unless there is a strong reason to share them across
tasks.

The previous top-level `state.metrics` block was removed because the metrics had
clear task ownership.

Current split:

- `abinitio.metrics`
  - average confidence
  - average volume-class confidence
  - assignment change rate
  - pose RMS values and EMAs
  - side-length resolution
- `homorefine.metrics`
  - average confidence
  - average volume-class confidence
  - pose RMS values
  - FSC scores
  - FSC resolution
  - FSC resolution change

FSC-related metrics are specifically part of `homorefine`, not global shared
state.

### State Placement Rules

When adding a new runtime field, decide in this order:

1. Is this value shared across engine, scheduler, solver, or checkpoint/resume?
2. If yes, is it generic progress, shared schedule, task engine state, task
   solver state, task scheduler state, or task metrics?
3. If not, keep it local to the component that owns it.

A useful shortcut:

- communication need -> shared `state`
- no communication need -> local attribute

### Checkpoint And Resume Rule

`state` should contain the runtime information needed to resume shared
optimization behavior correctly. This is one of the main reasons to keep certain
task scheduler counters and execution flags in shared state.

At the same time, resume support does not justify storing every local scratch
value in `state`. Only keep what affects future shared behavior.

### Snapshot And Logging Rule

Monitoring snapshots should not mechanically mirror the whole `state`.

For example, `ScheduleCheckSnapshot` should only include values whose changes are
meaningful for schedule decisions or operator-facing logs. Once a value becomes
engine-private and no longer affects shared decisions, it should not
automatically remain in the snapshot.

In other words:

- `state` is for communication
- snapshots are for decisions and observability
- the two structures should stay related, but not identical

### Rules For New Tasks

When adding a new CLI command or task:

1. Reuse `io`, `data`, `logging`, `reproduce`, and `modules` whenever the
   meaning is unchanged.
2. Add a new top-level task block only for genuinely task-specific runtime
   policy.
3. Add task-local state under that task instead of expanding generic global
   state.
4. Only promote a field into shared top-level structures if multiple tasks truly
   share the same meaning.

The default direction should be:

- stable capability -> `modules`
- task runtime policy -> `<task>.*`
- cross-component runtime communication -> `state`
- one-owner transient value -> local attribute

### Current Concrete Decisions

These decisions are part of the current design baseline:

- `particle_mask` is owned by `modules.search`
- `full_backprojection` is owned by `modules.volume`
- `accumulate_chunk` is owned by `modules.statistics.noise`
- `solvent_mask` currently stays under each task engine
- `loss_ema_decay` and `pose_rms_ema_decay` are engine config
- `activate_learning_rate_decay` is solver state for `abinitio`
- `fsc_threshold` is not a tunable config for `homorefine`; the task uses
  `0.143`
- top-level shared `state.metrics` is removed in favor of task-local metrics

### Quick Checklist

Before adding a new field, ask:

- Is this user-configurable?
- Is this shared across tasks or specific to one task?
- Is this shared across runtime components or only owned by one implementation?
- Does this need checkpoint/resume continuity?
- Does this need CLI exposure?
- Does the current name match the project style?

If these questions are answered consistently, the field will usually end up in
the right place without extra abstraction.