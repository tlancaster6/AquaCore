---
phase: 01-foundation-and-physics-math
plan: 02
subsystem: geometry
tags: [pytorch, camera, pinhole, fisheye, opencv, distortion, projection]

requires:
  - phase: 01-01
    provides: "CameraIntrinsics, CameraExtrinsics dataclasses with typed tensor fields; conftest.py device fixture"

provides:
  - "_PinholeCamera with project() via cv2.projectPoints and pixel_to_ray() via cv2.undistortPoints"
  - "_FisheyeCamera with project() via cv2.fisheye.projectPoints and pixel_to_ray() via cv2.fisheye.undistortPoints"
  - "create_camera() factory dispatching on is_fisheye; validates device, K/R/t shapes, dist_coeffs dimensionality"
  - "15 tests covering known-value projection, back-projection, round-trip, device validation, and batching"

affects:
  - 01-03  # refraction.py back-project uses pixel_to_ray from camera model
  - 01-05  # calibration.py constructs cameras via create_camera()
  - all    # create_camera is primary user-facing camera construction API

tech-stack:
  added: []
  patterns:
    - "OpenCV boundary pattern: detach.cpu.numpy for cv2 calls, .to(device) for results — no OpenCV tensors"
    - "atan2(|cross|, dot) for numerically stable small-angle measurement (avoids float32 acos catastrophic cancellation near 1.0)"
    - "Internal _PinholeCamera/_FisheyeCamera classes; public API is create_camera() factory only"

key-files:
  created:
    - tests/unit/test_camera.py
  modified:
    - src/aquacore/camera.py
    - src/aquacore/__init__.py

key-decisions:
  - "OpenCV boundary: always .cpu().numpy() before cv2 calls, .to(device) after — documented in class docstring as non-differentiable"
  - "atan2(|cross|, dot) for round-trip angle tests — float32 acos near 1.0 gives 4.88e-4 rad noise even when rays are bit-identical; atan2 formula returns exact 0.0"
  - "Fisheye uses cv2.fisheye.projectPoints/undistortPoints (K, D args); pinhole uses cv2.projectPoints/undistortPoints (cameraMatrix, distCoeffs args) — API difference matters"

patterns-established:
  - "Pattern: OpenCV calls always receive (N, 1, 3) for projectPoints and (N, 1, 2) for undistortPoints — single reshape convention for all camera models"
  - "Pattern: create_camera() is the sole public constructor — internal classes are prefixed _ and not re-exported"
  - "Pattern: atan2(cross.norm, dot) for unit-vector angle comparison in tests — more robust than acos(dot.clamp) for float32"

duration: 30min
completed: 2026-02-18
---

# Phase 1 Plan 02: Camera Models (Pinhole, Fisheye, create_camera factory) Summary

**Pinhole and fisheye camera models with project/back-project via OpenCV CPU boundary, create_camera() factory with device/shape validation, and 15 round-trip tests using numerically stable atan2 angle measurement**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-02-18
- **Completed:** 2026-02-18
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Implemented `_PinholeCamera` and `_FisheyeCamera` with project() (cv2.projectPoints / cv2.fisheye.projectPoints) and pixel_to_ray() (cv2.undistortPoints / cv2.fisheye.undistortPoints). Both are documented as non-differentiable due to the OpenCV CPU round-trip.
- Implemented `create_camera()` factory that validates device consistency across K/R/t tensors and checks shapes of K (3,3), R (3,3), t (3,), and dist_coeffs (1D) before dispatching to the correct model.
- Wrote 15 tests covering: on-axis and off-axis known-value projection, behind-camera valid mask, round-trip with and without distortion, fisheye round-trip, factory dispatch, device mismatch ValueError, bad K/t shape ValueError, and batch correctness.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement camera.py with pinhole, fisheye, and create_camera factory** - `d6e6b46` (feat)
2. **Task 2: Write camera model tests with known-value and round-trip verification** - `f673ee2` (test)

## Files Created/Modified

- `src/aquacore/camera.py` - _PinholeCamera, _FisheyeCamera, create_camera(); OpenCV boundary pattern documented in class docstrings
- `src/aquacore/__init__.py` - Added `create_camera` to imports and `__all__`
- `tests/unit/test_camera.py` - 15 tests for pinhole/fisheye projection, back-projection, round-trip, and factory validation

## Decisions Made

- **OpenCV boundary pattern:** All cv2 calls receive `.detach().cpu().numpy().astype(np.float64)` tensors and results are converted back to torch on the original device. Documented in class docstrings as non-differentiable. This matches the research pitfall (dist_coeffs must be float64 for OpenCV).
- **atan2 angle formula for tests:** `torch.atan2(cross.norm, dot)` instead of `torch.acos(dot.clamp(-1, 1))`. When two float32 rays are bit-identical, `acos` gives ~4.88e-4 rad error due to catastrophic cancellation in the dot product near 1.0; `atan2` returns exactly 0.0. This enables the 1e-5 tolerance specified in the plan.
- **Fisheye API difference:** `cv2.fisheye.projectPoints` takes `K=` and `D=` keyword args (not `cameraMatrix=` and `distCoeffs=`), and requires input shape `(N, 1, 3)`. Pinhole uses `(cameraMatrix=, distCoeffs=)` with shape `(N, 3)` for projectPoints but `(N, 1, 2)` for undistortPoints.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed import sort order in camera.py (cv2 before numpy)**
- **Found during:** Task 1 (pre-commit hook)
- **Issue:** `import cv2` appeared after `import numpy as np`; ruff required alphabetical order (cv2 < numpy)
- **Fix:** Used `ruff check --fix` to auto-sort imports
- **Files modified:** src/aquacore/camera.py
- **Verification:** `hatch run lint` passes cleanly
- **Committed in:** d6e6b46 (Task 1 commit, second attempt after hook fix)

**2. [Rule 1 - Bug] Used atan2 stable angle formula instead of naive acos for round-trip test**
- **Found during:** Task 2 (test_pinhole_round_trip failure at 4.88e-4 vs 1e-5 tolerance)
- **Issue:** Plan specifies 1e-5 radian round-trip tolerance. `torch.acos(dot.clamp(-1,1))` gives ~4.88e-4 error even when float32 rays are bit-identical, due to catastrophic cancellation: `cos(angle) = 0.9999999...` in float32 loses the last bit, and `acos` amplifies this to ~5e-4 rad.
- **Fix:** Replaced with `_stable_angle(a, b) = atan2(|cross(a,b)|, dot(a,b))`. The cross product of identical float32 vectors is exactly zero, so `atan2(0, dot) = 0.0`.
- **Files modified:** tests/unit/test_camera.py
- **Verification:** `test_pinhole_round_trip[cpu]` passes at 0.0 rad error
- **Committed in:** f673ee2 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 — import style and precision bug in test)
**Impact on plan:** Both fixes were necessary for correctness. No scope creep. The atan2 fix actually improves test robustness — it is the mathematically correct way to measure small angles in float32.

## Issues Encountered

- Two pre-existing failures exist in untracked scaffolded files (`test_refraction.py::test_refractive_back_project_consistency` and `test_triangulation.py::test_refractive_triangulation_integration`). These were present before this plan and are unrelated to camera.py. They will be addressed in plan 03/04 when those modules are worked on.

## Next Phase Readiness

- `create_camera()` factory is ready for use in calibration.py (plan 05) and projection models (plan 06)
- Both camera models implement the project/pixel_to_ray interface required by refraction.py (plan 03)
- `__init__.py` exports `create_camera` in the public API

---

## Self-Check: PASSED

- src/aquacore/camera.py: EXISTS
- tests/unit/test_camera.py: EXISTS
- src/aquacore/__init__.py: EXISTS (create_camera exported)
- Commit d6e6b46 (feat 01-02 camera.py): EXISTS
- Commit f673ee2 (test 01-02 test_camera.py): EXISTS
- All 15 test_camera.py tests pass on CPU

*Phase: 01-foundation-and-physics-math*
*Completed: 2026-02-18*
