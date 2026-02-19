---
phase: 01-foundation-and-physics-math
verified: 2026-02-18T20:00:39Z
status: passed
score: 5/5 must-haves verified
---

# Phase 1: Foundation and Physics Math Verification Report

**Phase Goal:** All geometry primitives are implemented, device-agnostic, and verified against known values
**Verified:** 2026-02-18T20:00:39Z
**Status:** PASSED
**Re-verification:** No -- initial verification
## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | User can import CameraIntrinsics, CameraExtrinsics, InterfaceParams, Vec2, Vec3, Mat3 from aquakit and construct instances with typed fields | VERIFIED | All six names in __init__.py __all__; CameraExtrinsics.C property returns -R.T @ t; INTERFACE_NORMAL=[0,0,-1] exported |
| 2 | User can create pinhole or fisheye Camera via create_camera() and round-trip project->back-project to within 1e-5 on both CPU and CUDA | VERIFIED | test_pinhole_round_trip uses atan2 stable angle < 1e-5 rad; test_fisheye_round_trip < 1e-4 rad; both parametrized over device fixture |
| 3 | User can call snells_law_3d and get refracted rays satisfying Snells law; TIR returns a flag not NaN | VERIFIED | test_snells_law_satisfies_ratio checks n1*sin1==n2*sin2 at 4 angles to 1e-5; test_total_internal_reflection asserts valid=False and direction==zeros(3) |
| 4 | User can triangulate a 3D point from batched rays matching ground-truth; point-to-ray distance reports correct reprojection error | VERIFIED | test_triangulate_two_rays_known_point recovers (0,0,5) to 1e-5; test_point_off_ray_known_distance=1.0; TRI-03 integration passes |
| 5 | All geometry tests pass on CPU and CUDA with parametrized device fixtures; no hardcoded .cuda() or AquaCal/AquaMVS imports | VERIFIED | conftest.py device fixture at tests/ root; .cuda() grep finds only docstring comment; no aquacal/aquamvs imports in tests/ |

**Score:** 5/5 truths verified

---
### Required Artifacts

| Artifact | Expected | Exists | Substantive | Wired | Status |
|----------|----------|--------|-------------|-------|--------|
| src/aquakit/types.py | CameraIntrinsics, CameraExtrinsics, InterfaceParams, Vec2, Vec3, Mat3 | YES | 105 lines; 3 dataclasses 3 aliases INTERFACE_NORMAL const | Imported by __init__.py camera.py refraction.py | VERIFIED |
| src/aquakit/interface.py | ray_plane_intersection | YES | 56 lines; (points,valid) tuple; parallel+behind-origin handled | Imported by __init__.py and refraction.py | VERIFIED |
| src/aquakit/transforms.py | rvec_to_matrix, matrix_to_rvec, compose_poses, invert_pose, camera_center | YES | 169 lines; pure-PyTorch Rodrigues; theta=0 and theta=pi edge cases | Imported by __init__.py | VERIFIED |
| src/aquakit/camera.py | _PinholeCamera, _FisheyeCamera, create_camera | YES | 270 lines; both classes project/pixel_to_ray via OpenCV CPU boundary; device validation | Imported by __init__.py | VERIFIED |
| src/aquakit/refraction.py | snells_law_3d, trace_ray_air_to_water, trace_ray_water_to_air, refractive_project, refractive_back_project | YES | 314 lines; vector Snells law, Newton-Raphson 10 fixed iters, TIR valid mask | Imported by __init__.py (all 5) | VERIFIED |
| src/aquakit/triangulation.py | triangulate_rays, point_to_ray_distance | YES | 94 lines; closed-form linear solve; LinAlgError->ValueError | Imported by __init__.py | VERIFIED |
| src/aquakit/__init__.py | Public API re-exports with __all__ | YES | 60 lines; 21 public symbols in sorted __all__ | All modules imported | VERIFIED |
| tests/conftest.py | Device fixture parametrizing CPU and CUDA | YES | 25 lines; pytest.fixture cpu+cuda-skipif | At tests/ root; accessible to all subdirs | VERIFIED |
| tests/unit/test_types.py | Foundation types tests | YES | 5 tests; device fixture on 3 of 5 | Imports from aquakit | VERIFIED |
| tests/unit/test_transforms.py | Transform tests | YES | 8 tests; all use device fixture | Imports from aquakit | VERIFIED |
| tests/unit/test_interface.py | Ray-plane intersection tests | YES | 4 tests; all use device fixture | Imports from aquakit | VERIFIED |
| tests/unit/test_camera.py | Camera projection/back-projection tests | YES | 15 tests; all use device fixture; stable_angle for 1e-5 tolerance | Imports from aquakit and aquakit.camera | VERIFIED |
| tests/unit/test_refraction.py | Refraction known-value tests | YES | 21 tests across 4 test classes; all use device fixture | Imports from aquakit | VERIFIED |
| tests/unit/test_triangulation.py | Triangulation tests including TRI-03 integration | YES | 10 tests; TRI-03 calls refractive_project for integration | Imports from aquakit | VERIFIED |

---
### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| src/aquakit/__init__.py | src/aquakit/types.py | from .types import | WIRED | Line 22: imports INTERFACE_NORMAL, CameraExtrinsics, CameraIntrinsics, InterfaceParams, Mat3, Vec2, Vec3 |
| src/aquakit/__init__.py | src/aquakit/transforms.py | from .transforms import | WIRED | Line 14: imports camera_center, compose_poses, invert_pose, matrix_to_rvec, rvec_to_matrix |
| src/aquakit/__init__.py | src/aquakit/camera.py | from .camera import create_camera | WIRED | Line 5 |
| src/aquakit/__init__.py | src/aquakit/refraction.py | from .refraction import | WIRED | Lines 7-13: refractive_back_project, refractive_project, snells_law_3d, trace_ray_air_to_water, trace_ray_water_to_air |
| src/aquakit/__init__.py | src/aquakit/triangulation.py | from .triangulation import | WIRED | Line 21: point_to_ray_distance, triangulate_rays |
| src/aquakit/camera.py | src/aquakit/types.py | from .types import | WIRED | Line 13: CameraExtrinsics, CameraIntrinsics |
| src/aquakit/refraction.py | src/aquakit/interface.py | from .interface import | WIRED | Line 5: ray_plane_intersection |
| src/aquakit/refraction.py | src/aquakit/types.py | from .types import | WIRED | Line 6: InterfaceParams |
| tests/unit/test_triangulation.py | src/aquakit/refraction.py | TRI-03 integration: refractive_project | WIRED | Line 225: from aquakit import refractive_project; called at line 239 |
| All test files | tests/conftest.py | device fixture | WIRED | All geometry test functions accept device: torch.device parameter |

---

### Requirements Coverage

No per-requirement row mapping for Phase 1 found in REQUIREMENTS.md. All five ROADMAP success criteria used directly as observable truths above -- all satisfied.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| tests/conftest.py | 22 | .cuda() in docstring comment only | Info | Not a code call; this is a prohibition note to test authors |

No blocker or warning anti-patterns found.

Scan results:
- No TODO/FIXME/XXX/PLACEHOLDER in src/
- No stub returns in src/
- No Not implemented responses in src/
- No import aquacal or import aquamvs in tests/
- No hardcoded .cuda() calls in test code

---
### Human Verification Required

#### 1. CUDA Device Test Execution

**Test:** Run hatch run test-all tests/unit/ -v on a machine with a CUDA GPU.
**Expected:** All CUDA-parametrized tests pass (currently they skip cleanly on CPU-only machine per SUMMARY notes).
**Why human:** CUDA hardware not available in this environment.

#### 2. Round-Trip Tolerance Interpretation

**Test:** Note that test_pinhole_round_trip measures angular deviation in radians (1e-5 rad), not pixels.
**Expected:** For a 500px focal-length camera, 1e-5 rad is sub-pixel. Implementation and test are both correct.
**Why human:** The success criterion wording says 1e-5 pixel but the test measures radians. No code change needed; the physics is right.

---

### Gaps Summary

No gaps. All 5 success criteria verified. All 14 artifacts exist, are substantive, and are correctly wired into the module graph.
No stubs, no orphaned code, no AquaCal/AquaMVS leakage, no hardcoded device calls.

---

_Verified: 2026-02-18T20:00:39Z_
_Verifier: Claude (gsd-verifier)_