---
phase: 01-foundation-and-physics-math
plan: 01
subsystem: geometry
tags: [pytorch, types, transforms, rodrigues, ray-plane-intersection, camera-geometry]

requires: []

provides:
  - "CameraIntrinsics, CameraExtrinsics, InterfaceParams dataclasses with typed tensor fields"
  - "Vec2, Vec3, Mat3 type aliases and INTERFACE_NORMAL constant"
  - "ray_plane_intersection with valid mask (interface.py)"
  - "rvec_to_matrix, matrix_to_rvec (pure PyTorch Rodrigues with theta=0 and theta=pi edge cases)"
  - "compose_poses, invert_pose, camera_center (transforms.py)"
  - "device-parametrized conftest.py fixture shared by all tests"

affects:
  - 01-02  # camera.py depends on CameraIntrinsics, CameraExtrinsics
  - 01-03  # refraction.py depends on InterfaceParams, Vec3
  - 01-04  # triangulation.py depends on Vec3
  - 01-05  # calibration.py depends on all types
  - all    # conftest.py device fixture used by all test modules

tech-stack:
  added: []
  patterns:
    - "TypeAlias for Vec2/Vec3/Mat3 with documented shapes (documentation-only, not enforced at runtime)"
    - "Dataclasses with torch.Tensor fields and property computed from tensor fields (CameraExtrinsics.C)"
    - "Pure-PyTorch Rodrigues with explicit edge-case guards at theta=0 and theta=pi"
    - "ray_plane_intersection returns (points, valid) tuple — no NaN, no None"
    - "conftest.py device fixture parametrizes CPU + CUDA-skip pattern for all geometry tests"
    - "torch.testing.assert_close(atol=1e-5, rtol=0) for all geometry float comparisons"

key-files:
  created:
    - src/aquacore/types.py
    - src/aquacore/interface.py
    - src/aquacore/transforms.py
    - tests/conftest.py
    - tests/unit/test_types.py
    - tests/unit/test_transforms.py
    - tests/unit/test_interface.py
  modified:
    - src/aquacore/__init__.py

key-decisions:
  - "Pure-PyTorch Rodrigues (not cv2.Rodrigues) — enables GPU/autograd, handles theta=0 and theta=pi edge cases explicitly"
  - "dist_coeffs stored as float64 — OpenCV requires float64; K is float32 following AquaMVS"
  - "ray_plane_intersection returns (points, valid) mask — no NaN, consistent with AquaMVS error signaling pattern"
  - "INTERFACE_NORMAL = [0, 0, -1] — upward from water, matches CLAUDE.md coordinate convention"
  - "conftest.py at tests/ root (not tests/unit/) — shared by all test subdirectories"

patterns-established:
  - "Pattern: (output, valid_mask) return type for all functions that can fail on individual elements"
  - "Pattern: device-agnostic — all tensors follow input device, no .cuda() calls anywhere"
  - "Pattern: dataclass with float32 tensor fields, float64 only for dist_coeffs (OpenCV boundary)"

duration: 25min
completed: 2026-02-18
---

# Phase 1 Plan 01: Foundation Types, Transforms, and Interface Summary

**Pure-PyTorch geometry foundation: Vec2/Vec3/Mat3 aliases, CameraIntrinsics/CameraExtrinsics/InterfaceParams dataclasses, Rodrigues rotation with edge-case guards, pose composition/inversion, ray-plane intersection with valid mask, and device-parametrized test fixture**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-02-18
- **Completed:** 2026-02-18
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Implemented all foundation types (CameraIntrinsics, CameraExtrinsics, InterfaceParams, Vec2, Vec3, Mat3, INTERFACE_NORMAL) with documented shapes and coordinate-system conventions
- Implemented pure-PyTorch Rodrigues (rvec_to_matrix / matrix_to_rvec) with explicit guards for theta=0 (identity) and theta=pi (special-case axis extraction); removed cv2 dependency from transforms
- Implemented ray_plane_intersection returning (points, valid) tuple — handles parallel rays and behind-origin intersections correctly
- Created shared conftest.py device fixture parametrized over CPU + CUDA-skip, used by all 18 geometry tests

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement types.py, interface.py, transforms.py, and update __init__.py** - `2398fb9` (feat)
2. **Task 2: Create conftest.py device fixture and write tests** - `e7b826c` (test)

## Files Created/Modified

- `src/aquacore/types.py` - Vec2/Vec3/Mat3 aliases, INTERFACE_NORMAL, CameraIntrinsics, CameraExtrinsics (with C property), InterfaceParams dataclasses
- `src/aquacore/interface.py` - ray_plane_intersection(origins, directions, plane_normal, plane_d) -> (points, valid)
- `src/aquacore/transforms.py` - rvec_to_matrix, matrix_to_rvec, compose_poses, invert_pose, camera_center
- `src/aquacore/__init__.py` - Public API re-exports and sorted __all__
- `tests/conftest.py` - Shared device fixture (CPU + CUDA-skipif)
- `tests/unit/test_types.py` - 5 tests for foundation types
- `tests/unit/test_transforms.py` - 8 tests for all transform functions
- `tests/unit/test_interface.py` - 4 tests for ray-plane intersection

## Decisions Made

- **Pure-PyTorch Rodrigues:** Implemented Rodrigues formula directly in PyTorch instead of wrapping cv2.Rodrigues. This makes transforms.py device-agnostic and autograd-compatible. cv2.Rodrigues is CPU-only and returns NumPy arrays, which is incompatible with the device-agnostic principle. Edge-case guards added at theta < 1e-10 (identity) and theta > pi - 1e-6 (special-case axis extraction using diagonal entries).
- **dist_coeffs as float64:** Stored in CameraIntrinsics as float64 to match OpenCV's API requirement. K matrix is float32 following AquaMVS convention.
- **conftest.py at tests/ root:** Placed at the tests/ root rather than tests/unit/ so the device fixture is accessible to all test subdirectories (unit, integration, e2e).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed lint errors blocking commit**
- **Found during:** Task 1 (pre-commit hook)
- **Issue:** (a) `__all__` in `__init__.py` not sorted per RUF022; (b) ambiguous variable name `I` in transforms.py (E741)
- **Fix:** Used `ruff check --fix` to auto-sort `__all__`; renamed `I` to `eye3` in rvec_to_matrix
- **Files modified:** src/aquacore/__init__.py, src/aquacore/transforms.py
- **Verification:** `hatch run lint` passes cleanly
- **Committed in:** 2398fb9 (Task 1 commit)

**2. [Rule 1 - Bug] Fixed import sorting in test file blocking commit**
- **Found during:** Task 2 (pre-commit hook)
- **Issue:** Import block in test_transforms.py flagged as unsorted (I001)
- **Fix:** Used `ruff check --fix` to reformat imports into parenthesized grouped form
- **Files modified:** tests/unit/test_transforms.py
- **Verification:** `hatch run lint` passes cleanly
- **Committed in:** e7b826c (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - lint/style issues caught by pre-commit)
**Impact on plan:** Both fixes were minor style/formatting corrections. No logic changes, no scope creep.

## Issues Encountered

None — plan executed as specified. All 18 tests pass on CPU; 15 CUDA-parametrized tests skip cleanly (CUDA not available on this machine, which is expected and documented as a known blocker in STATE.md).

## Next Phase Readiness

- All Phase 1, Plan 02 dependencies satisfied: types.py exports CameraIntrinsics, CameraExtrinsics, InterfaceParams; __init__.py updated
- conftest.py device fixture ready for all subsequent test modules
- Plan 02 (camera.py with create_camera factory) can proceed immediately

---

*Phase: 01-foundation-and-physics-math*
*Completed: 2026-02-18*
