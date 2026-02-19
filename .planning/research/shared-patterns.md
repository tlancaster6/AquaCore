# Cross-Repo Patterns: AquaCal vs AquaMVS

Mapped: 2026-02-18

This document highlights where the two repos agree, disagree, and what AquaKit needs to reconcile.

---

## 1. Consistent Across Both Repos

These patterns are stable and AquaKit should preserve them.

### Coordinate Systems
- **World frame:** Z-down (into water), origin at reference camera
- **Camera frame:** Z-forward, X-right, Y-down (OpenCV standard)
- **Extrinsics:** P_cam = R @ P_world + t (world-to-camera)
- **Camera center:** C = -R^T @ t
- **Interface normal:** [0, 0, -1] (points upward, from water toward air)

### Fisheye Model
- Both use OpenCV equidistant fisheye (4 coefficients: k1-k4)
- Both dispatch to `cv2.fisheye.*` functions based on `is_fisheye` flag

### Refraction Physics
- Both use same Snell's law formulation: `n_ratio = n_air / n_water`
- Same interface model: horizontal plane at Z = water_z
- Same refractive indices: n_air=1.0, n_water=1.333

### Triangulation Algorithm
- Both use identical closed-form linear least squares: `A = sum(I - d_i @ d_i^T)`
- Both require >= 2 rays

### Newton-Raphson for Forward Projection
- Both use same algorithm for projecting underwater 3D points through refractive interface
- Both clamp intermediate values to valid ranges

### Calibration JSON Schema
- Both read the same AquaCal JSON format (version "1.0")
- Same field names: K, dist_coeffs, image_size, is_fisheye, R, t, water_z

---

## 2. Key Differences

### Compute Backend

| Aspect | AquaCal | AquaMVS |
|--------|---------|---------|
| **Library** | NumPy | PyTorch |
| **Dtype** | float64 | float32 (except dist_coeffs: float64) |
| **GPU** | No | Yes (CPU + CUDA) |
| **Differentiable** | No | Yes |

**AquaKit decision:** PyTorch (per project requirements). Follow AquaMVS patterns.

### Point Shape Conventions

| Aspect | AquaCal | AquaMVS |
|--------|---------|---------|
| **Single point** | (3,) | (N, 3) where N=1 |
| **Batch points** | (N, 3) in batch functions | (N, 3) always |
| **Single pixel** | (2,) | (N, 2) where N=1 |

**AquaKit decision:** Always batch — (N, 3) shapes. Matches AquaMVS.

### Translation Vector Shape

| Aspect | AquaCal | AquaMVS |
|--------|---------|---------|
| **t shape** | (3,) or (3,1) inconsistent | (3,) always (normalizes at load) |

**AquaKit decision:** (3,) always. Normalize at load boundary.

### Error Handling for Invalid Projections

| Aspect | AquaCal | AquaMVS |
|--------|---------|---------|
| **Single-point failure** | Returns None | N/A (always batched) |
| **Batch failure** | NaN in output array | NaN + valid_mask boolean |
| **TIR** | Returns None | Clamp sin^2 >= 0, no explicit flag |

**AquaKit decision:** Return `(output, valid_mask)` tuple — matches AquaMVS, works for batches.

### Newton-Raphson Convergence

| Aspect | AquaCal | AquaMVS |
|--------|---------|---------|
| **Iterations** | max_iterations=10, stops on tolerance=1e-9 | Fixed 10, no early exit |
| **Tolerance** | 1e-9 meters | N/A (always runs 10) |
| **Rationale** | Efficiency | Deterministic gradients |

**AquaKit decision:** Fixed iterations (AquaMVS approach) — needed for autodiff support.

### Projection API

| Aspect | AquaCal | AquaMVS |
|--------|---------|---------|
| **Forward** | `refractive_project(camera, interface, point)` | `model.project(points)` — method on model |
| **Back** | `refractive_back_project(camera, interface, pixel)` | `model.cast_ray(pixels)` — method on model |
| **Model** | Standalone functions + Camera + Interface | `RefractiveProjectionModel` bundles everything |

**AquaKit decision:** Object-oriented (AquaMVS approach) with `ProjectionModel` protocol.

### Camera Construction

| Aspect | AquaCal | AquaMVS |
|--------|---------|---------|
| **Pattern** | `Camera(name, intrinsics, extrinsics)` class | `CameraData` dataclass (no methods) |
| **Fisheye** | `FisheyeCamera` subclass of `Camera` | `is_fisheye` flag on `CameraData` |
| **Methods** | project(), pixel_to_ray() on Camera | Projection via separate `ProjectionModel` |

**AquaKit decision:** Separate concerns — data (CameraIntrinsics, CameraExtrinsics) from behavior (ProjectionModel). Context says `create_camera()` factory only.

---

## 3. Inconsistencies to Resolve

### 3.1 dist_coeffs dtype
- AquaCal: float64 (NumPy default)
- AquaMVS: float64 (explicit, because OpenCV requires it)
- **Note:** Both agree on float64 for dist_coeffs. AquaKit should keep this even though other tensors are float32.

### 3.2 Back-projection return value
- AquaCal: returns `(intersection_point, refracted_direction)` — origin is ON the interface
- AquaMVS: returns `(origins, directions)` — origins also on water surface
- **Note:** Consistent in practice. AquaKit should document that ray origins are at the interface.

### 3.3 Flat interface restriction
- AquaCal: `refractive_project()` supports tilted interfaces (Brent fallback); `refractive_project_batch()` requires flat
- AquaMVS: Only supports flat interfaces (Newton-Raphson only)
- **Note:** User chose simplified air-to-water model. Flat-only is fine.

### 3.4 Undistortion
- AquaCal: `undistort_points()` operates on point arrays
- AquaMVS: `compute_undistortion_maps()` + `undistort_image()` operates on images
- **Note:** AquaKit Phase 3 needs image undistortion. Both patterns may be needed.

---

## 4. What AquaKit Extracts from Each

### From AquaCal (reference implementations)
- Type definitions: CameraIntrinsics, CameraExtrinsics, InterfaceParams, Vec2, Vec3, Mat3
- Physics: snells_law_3d, ray_plane_intersection
- Transforms: rvec_to_matrix, matrix_to_rvec, compose_poses, invert_pose
- Triangulation: triangulate_rays, point_to_ray_distance
- Calibration JSON schema and field names
- Error handling conventions (what to do on TIR, behind-camera, etc.)

### From AquaMVS (PyTorch patterns)
- ProjectionModel protocol (project + cast_ray)
- RefractiveProjectionModel implementation
- Batch tensor conventions (N, 3)
- Device handling (.to(), infer from input)
- Validity masking (output, valid_mask) tuples
- Fixed-iteration Newton-Raphson
- CameraData / CalibrationData dataclasses
- UndistortionData and undistortion pipeline
- ImageDirectorySet I/O pattern
- Test patterns (CPU/CUDA parametrization, torch.testing.assert_close)

---

## 5. Function Cross-Reference

| Capability | AquaCal | AquaMVS | AquaKit Target |
|-----------|---------|---------|-----------------|
| Snell's law | `snells_law_3d()` | Inline in RefractiveProjectionModel | Standalone function (Phase 1) |
| Forward project | `refractive_project()` | `model.project()` | `ProjectionModel.project()` (Phase 2) |
| Back-project | `refractive_back_project()` | `model.cast_ray()` | `ProjectionModel.cast_ray()` (Phase 2) |
| Triangulate | `triangulate_rays()` | `triangulate_rays()` | `triangulate_rays()` (Phase 1) |
| Ray-point distance | `point_to_ray_distance()` | N/A (inline) | `point_to_ray_distance()` (Phase 1) |
| Load calibration | `load_calibration()` | `load_calibration_data()` | `load_calibration_data()` (Phase 3) |
| Undistort image | N/A | `undistort_image()` | `undistort_image()` (Phase 3) |
| Frame I/O | N/A | `ImageDirectorySet` | `ImageSet` (Phase 4) |
| Pinhole project | `Camera.project()` | N/A (via ProjectionModel) | `create_camera()` (Phase 1) |
| Fisheye project | `FisheyeCamera.project()` | N/A (via ProjectionModel) | `create_camera()` (Phase 1) |
| Rodrigues | `rvec_to_matrix()` | N/A | `rvec_to_matrix()` (Phase 1) |
| Pose math | `compose_poses()`, `invert_pose()` | N/A | Phase 1 transforms |
