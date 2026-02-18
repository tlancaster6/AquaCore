# Phase 2: Projection Protocol - Context

**Gathered:** 2026-02-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement the ProjectionModel protocol and RefractiveProjectionModel with Newton-Raphson back-projection. This phase delivers the refractive projection layer that sits on top of Phase 1's physics primitives (Snell's law, camera models, refraction). Calibration loading, undistortion, and I/O are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Method naming
- Protocol methods: `project()` and `back_project()` — symmetric pair, consistent with Phase 1 camera models
- AquaMVS's `cast_ray()` renamed to `back_project()` — rewiring guide will document the rename
- `project()` returns `(pixels: Tensor, valid_mask: Tensor)` — simple, no convergence metadata
- `back_project()` returns `(origins: Tensor, directions: Tensor)` — origins on water surface, directions into water

### Constructor design
- Primary constructor takes raw tensors: `RefractiveProjectionModel(K, R, t, water_z, normal, n_air, n_water)` — matches AquaMVS flexibility
- Factory method: `RefractiveProjectionModel.from_camera(camera, interface)` — takes Phase 1 typed objects (create_camera() result + InterfaceParams)
- Precompute and cache derived values at construction: K_inv, camera center C, n_ratio
- `.to(device)` method returns model on target device, matching AquaMVS and PyTorch conventions

### Multi-camera batching
- Per-camera model: one RefractiveProjectionModel per camera (proven AquaMVS pattern)
- Multi-camera helpers: `project_multi(models, points)` and `back_project_multi(models, pixels)` in same module (projection/refractive.py)
- Helpers loop sequentially over cameras, stack results into (M, N, 2) / (M, N, 3) output tensors
- NOTE: Sequential loop is v1 — future optimization could vectorize by stacking camera params into a single batched Newton-Raphson pass

### Claude's Discretion
- `back_project()` valid_mask: Claude determines whether air→water back-projection can fail and whether a mask is needed (physics says no TIR for air→water, but edge cases may exist)
- Protocol implementation: `typing.Protocol` vs `ABC` — Claude picks based on codebase patterns (AquaMVS uses `@runtime_checkable Protocol`)
- Model mutability: frozen vs mutable `.to(device)` semantics — Claude decides what fits best

</decisions>

<specifics>
## Specific Ideas

- Newton-Raphson uses fixed 10 iterations (no early exit) — deterministic for autodiff, matches AquaMVS
- Flat interface only (no Brent fallback for tilted interfaces) — decided during project init
- Invalid pixels set to NaN in output, valid_mask is boolean — matches AquaMVS convention
- Clamp r_p to [0, r_q] per Newton-Raphson iteration — matches both AquaCal and AquaMVS
- Epsilon 1e-12 to prevent division by zero in Newton-Raphson — matches AquaMVS

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-projection-protocol*
*Context gathered: 2026-02-18*
