---
phase: 03-calibration-and-undistortion
plan: 01
subsystem: calibration
tags: [json, dataclass, pytorch, calibration, loader]

# Dependency graph
requires:
  - phase: 01-foundation-and-physics-math
    provides: CameraIntrinsics, CameraExtrinsics, InterfaceParams types from types.py

provides:
  - CameraData dataclass composing CameraIntrinsics and CameraExtrinsics
  - CalibrationData dataclass with cameras dict, InterfaceParams, camera_list, core_cameras(), auxiliary_cameras()
  - load_calibration_data() function accepting str | Path | dict
  - 25 comprehensive tests covering load paths, type composition, dtypes, edge cases, error paths

affects:
  - 03-02 (undistortion will consume CalibrationData.cameras[*].intrinsics)
  - downstream consumers (AquaCal, AquaMVS) that load calibration files

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Parse-and-validate in _parse_camera: fail fast per camera, log warning, skip and continue"
    - "Backward compat alias: interface_distance accepted as water_z for camera entries"
    - "water_z stored in InterfaceParams.water_z extracted from first valid camera"
    - "t shape normalization: (3,1) -> (3,) silently on load (known AquaCal quirk)"

key-files:
  created:
    - src/aquakit/calibration.py
    - tests/unit/test_calibration.py
  modified:
    - src/aquakit/__init__.py

key-decisions:
  - "water_z stored in InterfaceParams (not separate field on CalibrationData) - consistent with Phase 1 types"
  - "Bad camera entries skipped with UserWarning (not crash) - resilient loading for real-world partial calibrations"
  - "No AquaCal dependency - calibration.py uses only json, warnings, torch, pathlib (stdlib + torch)"
  - "K as float32, dist_coeffs as float64, R/t as float32 - matches Phase 1 and OpenCV requirements"

patterns-established:
  - "Helper _load_suppressing_warnings() in tests to keep pytest.raises blocks single-statement (PT012 compliance)"
  - "copy.deepcopy() at top-level import in test file for per-test mutation safety"

# Metrics
duration: 15min
completed: 2026-02-18
---

# Phase 3 Plan 01: AquaCal JSON Loader Summary

**CameraData and CalibrationData dataclasses backed by Phase 1 types with load_calibration_data() accepting file path or dict, graceful per-camera error handling, and 25 tests at 100% pass rate**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-02-18T00:00:00Z
- **Completed:** 2026-02-18
- **Tasks:** 2
- **Files modified:** 3 (calibration.py created/replaced, __init__.py updated, test_calibration.py created)

## Accomplishments

- Implemented CameraData and CalibrationData dataclasses that compose Phase 1 CameraIntrinsics, CameraExtrinsics, InterfaceParams
- load_calibration_data() accepts str, Path, or pre-parsed dict with full backward compatibility (interface_distance alias)
- Graceful skipping of invalid camera entries with UserWarning, ValueError only when all cameras fail
- 25 comprehensive tests covering valid loads, dtype verification, t-shape normalization, fisheye flag, auxiliary flag, CalibrationData methods, error paths, and warning cases

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement CameraData, CalibrationData, and load_calibration_data** - `e040ab8` (feat)
2. **Task 2: Comprehensive tests for calibration loader** - `e9926d3` (test)

## Files Created/Modified

- `src/aquakit/calibration.py` - CameraData, CalibrationData dataclasses and load_calibration_data function; no AquaCal imports
- `src/aquakit/__init__.py` - Added CalibrationData, CameraData, load_calibration_data to imports and __all__
- `tests/unit/test_calibration.py` - 25 tests using synthetic JSON fixtures; all pass

## Decisions Made

- water_z lives in InterfaceParams (not a separate CalibrationData field) - consistent with Phase 1 type design where InterfaceParams already owns water_z
- Per-camera validation skips bad cameras with warning rather than crashing - real-world calibration files sometimes have partial data; resilient loading is safer
- No AquaCal dependency - only json, warnings, torch, pathlib; aquakit stays importable with AquaCal uninstalled

## Deviations from Plan

None - plan executed exactly as written. Lint auto-fixes applied by pre-commit hooks (ruff import sorting, ruff format) handled inline without altering logic. PT012 compliance required extracting a one-line helper `_load_suppressing_warnings` in the test file.

## Issues Encountered

- Pre-commit hooks (ruff) auto-reformatted calibration.py and test_calibration.py on first commit; required re-staging modified files before the second commit attempt. Standard workflow for this project.
- PT012 lint rule (pytest.raises block must contain single simple statement) required moving the `warnings.catch_warnings()` call into a small helper function so the `pytest.raises` block remained a single call.

## Next Phase Readiness

- CalibrationData is ready for consumption by Plan 02 (undistortion remap) via CalibrationData.cameras[*].intrinsics
- All Phase 1 types compose correctly through calibration layer
- No blockers for 03-02

## Self-Check: PASSED

Files verified:
- FOUND: src/aquakit/calibration.py
- FOUND: tests/unit/test_calibration.py
- FOUND: src/aquakit/__init__.py
- FOUND: .planning/phases/03-calibration-and-undistortion/03-01-SUMMARY.md

Commits verified:
- FOUND: e040ab8 (feat: calibration implementation)
- FOUND: e9926d3 (test: calibration tests)

---
*Phase: 03-calibration-and-undistortion*
*Completed: 2026-02-18*
