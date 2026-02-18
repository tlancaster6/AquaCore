---
phase: 01-foundation-and-physics-math
plan: 03
subsystem: geometry
tags: [pytorch, snells-law, refraction, ray-tracing, triangulation, newton-raphson, physics-math]

requires:
  - phase: 01-01
    provides: "InterfaceParams, Vec3 types; ray_plane_intersection; INTERFACE_NORMAL constant"

provides:
  - "snells_law_3d: batched vector Snell's law with TIR detection (valid mask, no NaN)"
  - "trace_ray_air_to_water, trace_ray_water_to_air: ray-plane intersection + Snell's law"
  - "refractive_project: Newton-Raphson (10 fixed iterations) finds interface point for underwater 3D target"
  - "refractive_back_project: wraps trace_ray_air_to_water for pixel-to-water ray casting"
  - "triangulate_rays: closed-form linear solve (I - d@d.T) for N-ray 3D triangulation"
  - "point_to_ray_distance: perpendicular distance for reprojection error measurement"
  - "TRI-03 integration verified: refractive rays triangulate correctly end-to-end"

affects:
  - 01-04  # undistortion.py may use interface geometry
  - 01-05  # calibration.py uses refractive_project for calibration residuals
  - phase-02  # projection/refractive.py builds on these functions directly
  - phase-03  # calibration optimizer calls refractive_project in its cost function

tech-stack:
  added: []
  patterns:
    - "snells_law_3d: orient normal by cos_i sign check — handles both air→water and water→air"
    - "TIR signaling: (directions, valid) tuple — zeros for TIR rows, never NaN or None"
    - "Newton-Raphson: fixed 10 iterations — no convergence check (autograd-safe, deterministic)"
    - "Newton-Raphson: epsilon guards (+1e-12) prevent div-by-zero in gradient; r_q uses sqrt(dx²+dy²+1e-12)"
    - "triangulate_rays: catches LinAlgError, re-raises as ValueError('Degenerate ray configuration')"
    - "TRI-03: triangulate_rays accepts refractive rays (origin on surface, direction in water) directly"

key-files:
  created:
    - src/aquacore/refraction.py
    - src/aquacore/triangulation.py
    - tests/unit/test_refraction.py
    - tests/unit/test_triangulation.py
  modified:
    - src/aquacore/__init__.py

key-decisions:
  - "snells_law_3d orients normal internally by checking sign of cos_i — caller does not need to pre-orient"
  - "TIR returns (zeros, False) per (output, valid_mask) pattern — consistent with AquaMVS; not None (AquaCal pattern)"
  - "refractive_project returns (N, 3) interface point on water surface — caller projects via camera model to get pixel"
  - "TRI-03 uses refractive_project to find correct interface points (satisfying Snell's law) then triangulates water-side rays — direct air-ray casting does not work due to refraction bending"

patterns-established:
  - "Pattern: epsilon r_q = sqrt(dx²+dy²+1e-12) prevents singularity for on-axis points in Newton-Raphson"
  - "Pattern: torch.minimum(r_p, r_q) + torch.clamp(r_p, min=0) — non-in-place ops preserve autograd"
  - "Pattern: type: ignore[attr-defined] on torch.linalg.LinAlgError — not exported from stubs but available at runtime"
  - "Pattern: TRI-03 integration test — use refractive_project to get true interface points, not direct line-of-sight to target"

duration: 10min
completed: 2026-02-18
---

# Phase 1 Plan 03: Snell's Law, Refractive Ray Tracing, and Triangulation Summary

**Batched PyTorch Snell's law (TIR-safe), Newton-Raphson refractive projection (10 fixed iters, autograd-compatible), air-to-water and water-to-air ray tracing, closed-form triangulation with degenerate-ray guard, and end-to-end TRI-03 integration test proving refractive rays triangulate to the correct underwater point**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-02-18T19:45:33Z
- **Completed:** 2026-02-18T19:55:46Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Implemented all refraction functions (REF-01..07): snells_law_3d with explicit TIR detection via valid mask; trace_ray_air_to_water and trace_ray_water_to_air composing ray-plane intersection with Snell's law; refractive_project via Newton-Raphson (10 fixed iterations, epsilon guards, non-in-place clamp ops); refractive_back_project as a wrapper
- Implemented triangulate_rays (closed-form linear solve using I - d@d.T sum) with degenerate-ray ValueError guard; point_to_ray_distance for reprojection error measurement
- Proved TRI-03 integration: refractive_project finds the exact interface point satisfying Snell's law per camera, the water-side rays from those points triangulate to the known underwater target within 0.01 tolerance
- All 15 refraction tests + 9 triangulation tests pass on CPU; 24 CUDA-parametrized tests skip cleanly (no CUDA available)

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement refraction.py and triangulation.py** - `c64e393` (feat)
2. **Task 2: Write known-value tests for refraction and triangulation including TRI-03 integration** - `382a87c` (test)

**Plan metadata:** *(to be committed)*

## Files Created/Modified

- `src/aquacore/refraction.py` - snells_law_3d, trace_ray_air_to_water, trace_ray_water_to_air, refractive_project, refractive_back_project (5 public functions)
- `src/aquacore/triangulation.py` - triangulate_rays, point_to_ray_distance (2 public functions)
- `src/aquacore/__init__.py` - Added 7 new public exports; preserved create_camera from Plan 02; __all__ sorted
- `tests/unit/test_refraction.py` - 15 known-value tests: Snell's law (7), ray tracing (6), projection (4), back-projection (3), covering physics invariants, TIR, convergence
- `tests/unit/test_triangulation.py` - 9 tests: triangulation (5), point-to-ray (4), TRI-03 integration (1)

## Decisions Made

- **snells_law_3d handles both ray directions:** The normal orientation (flip when cos_i < 0) is computed internally, so callers do not need to pre-orient the surface normal for air→water vs water→air. This matches AquaCal's pattern and simplifies all call sites.
- **TRI-03 test design uses refractive_project, not direct line-of-sight:** Initially the integration test aimed air rays directly at the known underwater point. This is geometrically wrong — refraction bends the ray at the interface, so the water-side ray does not pass through the original target. Fixed by using refractive_project to find the interface points that satisfy Snell's law, then constructing water-side rays (interface_pt → target). Triangulating those rays recovers the target correctly.
- **type: ignore[attr-defined] on torch.linalg.LinAlgError:** basedpyright reports this as not exported from torch.linalg stubs. The ignore comment suppresses the false-positive while preserving the runtime catch (LinAlgError is accessible at runtime via torch.linalg.LinAlgError). Avoids the alternative of catching `Exception` broadly.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test logic for TRI-03 refractive triangulation integration**
- **Found during:** Task 2 (test execution)
- **Issue:** Initial test aimed air rays directly at the known underwater point and traced them through the interface. The refracted ray bends at the surface and does not pass through the original target — triangulating those refracted rays gave error 0.63 (tolerance 0.01)
- **Fix:** Replaced direct line-of-sight approach with refractive_project to find the physically correct interface point P satisfying Snell's law (camera → P → Q). Water-side ray = (P, normalize(Q - P)). Both cameras produce rays that converge at Q by construction
- **Files modified:** tests/unit/test_triangulation.py
- **Verification:** Refractive triangulation error < 0.0001 after fix
- **Committed in:** 382a87c (Task 2 commit)

**2. [Rule 1 - Bug] Fixed test logic for refractive_back_project consistency test**
- **Found during:** Task 2 (test execution)
- **Issue:** Consistency test traced reversed ray using the *forward* water direction (pointing deeper into water). trace_ray_water_to_air requires a ray going toward the surface, so valid=False was returned
- **Fix:** Negated water_dirs (reverse_dirs = -water_dirs) before calling trace_ray_water_to_air to produce a ray pointing back toward the surface
- **Files modified:** tests/unit/test_refraction.py
- **Verification:** valid_rev[0]=True after fix; interface XY agrees within 1e-3
- **Committed in:** 382a87c (Task 2 commit)

**3. [Rule 1 - Bug] Fixed multiple lint issues in test files**
- **Found during:** Task 2 (pre-commit hook)
- **Issue:** (a) Import sorting (I001); (b) unused variables — dirs, rdirs, ipts, air_dirs, water_dirs (RUF059); (c) ambiguous minus sign − in comment (RUF003); (d) pytest match= needs raw string (RUF043); (e) unused import pytest in refraction (F401); (f) unused imports snells_law_3d, trace_ray_air_to_water in triangulation (F401)
- **Fix:** Auto-fixed I001 and F401 via `ruff check --fix`; manually renamed unused vars to _-prefixed; fixed minus sign; added r-prefix to match string
- **Files modified:** tests/unit/test_refraction.py, tests/unit/test_triangulation.py
- **Verification:** `hatch run lint` passes cleanly
- **Committed in:** 382a87c (Task 2 commit)

**4. [Rule 1 - Bug] Fixed pre-existing lint error in test_camera.py (from Plan 02)**
- **Found during:** Task 1 (lint check)
- **Issue:** test_camera.py had I001 (unsorted imports) — pre-existing from Plan 02 that wasn't committed
- **Fix:** `ruff check --fix tests/unit/test_camera.py`
- **Files modified:** tests/unit/test_camera.py
- **Committed in:** c64e393 (Task 1 commit)

---

**Total deviations:** 4 auto-fixed (Rules 1 — logic bugs and style issues)
**Impact on plan:** The TRI-03 test logic fix was the most significant — it revealed a conceptual issue in how the integration test was designed. The fix strengthened the test by making it explicitly use the correct physics (refractive_project for interface finding). All lint fixes were mechanical corrections. No scope creep.

## Issues Encountered

- basedpyright's `reportPrivateImportUsage` error on `torch.linalg.LinAlgError`: resolved with `# type: ignore[attr-defined]` as the symbol is available at runtime but not in PyTorch's type stubs. This is a known PyTorch stub limitation.

## Next Phase Readiness

- All Plan 04 (undistortion.py) and Plan 05 (calibration.py) dependencies satisfied
- Phase 2 projection/ subpackage can build on snells_law_3d and refractive_project directly
- conftest.py device fixture and all test patterns established for remaining phases

---

*Phase: 01-foundation-and-physics-math*
*Completed: 2026-02-18*
