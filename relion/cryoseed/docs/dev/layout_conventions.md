## Layout Conventions

cryoSeed uses a few stable layout orders instead of forcing one global order
everywhere. The active order should follow the primary semantics of the code
being written.

### Search-Space Code

Use `img -> vol -> rot -> trans` when multiple search axes appear together.

- Applies to local variables, indexing, comments, return values, and helper
  payloads in pose-search code.
- When image grouping is not part of the code block, use `vol -> rot -> trans`.
- This matches the common search-space view `B, K, Q, T`.

### Particle State

Use `rot -> trans -> vol` for per-particle runtime state.

- Applies to `Pose` and related particle-state containers.
- This intentionally differs from search-space order because the primary
  semantics are per-particle alignment state, not candidate-grid traversal.

### Engine Metrics And Logs

Use `quality -> assignment -> pose -> volume -> control` for engine-facing
metrics and summaries.

- `quality`: loss, confidence
- `assignment`: volume confidence, assignment change, occupancy
- `pose`: rotation and translation stability metrics
- `volume`: relative volume change and related map metrics
- `control`: final-epoch and stop-condition state

### Practical Rule

Before choosing an order, identify the primary subject of the code block:

- search-space candidate logic -> search-space order
- per-particle runtime state -> particle-state order
- human-facing monitoring and summaries -> engine metric order