# AquaCal Codebase Reference

Source: `C:\Users\tucke\PycharmProjects\AquaCal`
Mapped: 2026-02-18

## 1. Module Layout

### Core Modules (Primary for AquaCore)

- `core/camera.py` — Camera intrinsics/extrinsics, pinhole & fisheye projection, back-projection
- `core/refractive_geometry.py` — Ray tracing through air-water interface using Snell's law
- `core/interface_model.py` — Refractive interface (water surface) representation
- `core/board.py` — ChArUco board 3D geometry and corner transformations
- `triangulation/triangulate.py` — Multi-ray triangulation and ray-to-point distance
- `utils/transforms.py` — Rodrigues rotation vectors, pose composition and inversion
- `config/schema.py` — All dataclasses, type aliases, exceptions (central to API design)

### I/O and Validation

- `io/serialization.py` — JSON save/load for CalibrationResult
- `validation/reprojection.py` — Reprojection error computation

### Calibration Modules (Reference only)

- `calibration/intrinsics.py` — Per-camera intrinsic calibration
- `calibration/extrinsics.py` — Extrinsic initialization via pose graph and PnP
- `calibration/interface_estimation.py` — Joint refractive optimization
- `calibration/refinement.py` — Post-optimization refinement
- `calibration/_optim_common.py` — Shared parameter packing/unpacking, residuals, Jacobian

---

## 2. Key Classes and Types

### Type Aliases (from `config/schema.py`)

```python
Vec3 = NDArray[np.float64]  # shape (3,)
Mat3 = NDArray[np.float64]  # shape (3, 3)
Vec2 = NDArray[np.float64]  # shape (2,)
```

### CameraIntrinsics

**Fields:**
- `K: Mat3` — 3x3 intrinsic matrix
- `dist_coeffs: NDArray[np.float64]` — Pinhole: length 5 or 8; Fisheye: length 4
- `image_size: tuple[int, int]` — (width, height)
- `is_fisheye: bool` — default False

### CameraExtrinsics

**Fields:**
- `R: Mat3` — 3x3 rotation matrix (world to camera)
- `t: Vec3` — 3x1 translation vector (world to camera)

**Property:**
- `C: Vec3` — Camera center in world: `-R.T @ t`

### Camera

**Constructor:** `Camera(name: str, intrinsics: CameraIntrinsics, extrinsics: CameraExtrinsics)`

**Key Properties:** `.K`, `.dist_coeffs`, `.R`, `.t`, `.C`, `.image_size`, `.P` (3x4 projection matrix)

**Key Methods:**

| Method | Input | Output | Notes |
|--------|-------|--------|-------|
| `world_to_camera(point_world)` | Vec3 | Vec3 | `R @ p + t` |
| `project(point_world, apply_distortion=True)` | Vec3 | Vec2 or None | None if behind camera (Z <= 0) |
| `pixel_to_ray(pixel, undistort=True)` | Vec2 | Vec3 (unit, camera frame) | |
| `pixel_to_ray_world(pixel, undistort=True)` | Vec2 | (Vec3, Vec3) origin+direction | World frame |

### FisheyeCamera

Extends `Camera`; overrides `project()` and `pixel_to_ray()` using `cv2.fisheye` module.
- 4 distortion coefficients (equidistant model)

### Interface

**Constructor:**
```python
Interface(
    normal: Vec3,                       # unit normal, typically [0,0,-1]
    camera_distances: dict[str, float], # Per-camera Z-coord of water surface
    n_air: float = 1.0,
    n_water: float = 1.333
)
```

**Properties:** `n_ratio_air_to_water` (~0.75), `n_ratio_water_to_air` (~1.33)

---

## 3. Geometry & Math Functions

### Snell's Law (3D)

```python
snells_law_3d(
    incident_direction: Vec3,  # Unit vector toward interface
    surface_normal: Vec3,      # Unit normal (typically [0,0,-1])
    n_ratio: float             # n1/n2
) -> Vec3 | None  # None for total internal reflection
```

- Handles normal orientation internally (checks cos_i < 0 to flip)
- Formula: `t = n_ratio * d + (cos_t - n_ratio * cos_i) * n`
- Output normalized to unit vector

### Ray-Plane Intersection

```python
ray_plane_intersection(
    ray_origin: Vec3, ray_direction: Vec3,
    plane_point: Vec3, plane_normal: Vec3
) -> (Vec3, float) | (None, None)
```

- Returns None if parallel (|dot| < 1e-10)
- Returns ANY t (including negative)

### Refractive Projection (Forward)

```python
refractive_project(
    camera: Camera, interface: Interface,
    point_3d: Vec3, max_iterations=10, tolerance=1e-9
) -> Vec2 | None
```

- **Flat interface:** Newton-Raphson (2-4 iterations, ~50x faster)
- **Tilted interface:** Brent-search fallback
- Returns None on failure

### Refractive Back-Projection

```python
refractive_back_project(camera, interface, pixel) -> (Vec3, Vec3) | (None, None)
# Returns (intersection_point, refracted_direction)
```

### Batch Projection

```python
refractive_project_batch(
    camera, interface, points_3d: NDArray,  # (N, 3)
    max_iterations=10, tolerance=1e-9
) -> NDArray  # (N, 2), NaN for invalid
```

- Only supports flat interfaces (raises ValueError otherwise)

### Triangulation

```python
triangulate_rays(rays: list[tuple[Vec3, Vec3]]) -> Vec3
# Closed-form linear least squares: A @ P = b where A = sum(I - d_i d_i^T)

point_to_ray_distance(point: Vec3, ray_origin: Vec3, ray_direction: Vec3) -> float

triangulate_point(calibration, observations: dict[str, Vec2]) -> Vec3 | None
# Back-projects pixels to rays in water, then triangulates
```

### Transforms

```python
rvec_to_matrix(rvec: Vec3) -> Mat3
matrix_to_rvec(R: Mat3) -> Vec3
compose_poses(R1, t1, R2, t2) -> (Mat3, Vec3)
invert_pose(R, t) -> (Mat3, Vec3)
camera_center(R, t) -> Vec3  # -R.T @ t
```

---

## 4. Conventions

### Tensor Shapes

| Quantity | Shape | Units |
|----------|-------|-------|
| 3D point | (3,) | meters |
| 2D pixel | (2,) | pixels |
| Rotation matrix | (3, 3) | - |
| Translation | (3,) | meters |
| Rodrigues vector | (3,) | radians |
| Intrinsic K | (3, 3) | pixels |
| Distortion coeffs | (5,) or (8,) pinhole, (4,) fisheye | unitless |
| Batch corners | (N, 3) | meters |
| Batch pixels | (N, 2) | pixels |

### Coordinate Systems

- **World:** Z-down (into water), origin at reference camera
- **Camera:** Z-forward (along optical axis), X-right, Y-down (OpenCV standard)
- **Interface normal:** [0, 0, -1] (points from water toward air)

### Error Handling

**Return None on failure:** `project()`, `snells_law_3d()`, `refractive_project()`, `triangulate_point()`

**Raise exceptions:** `refractive_project_batch()` for non-flat interfaces, `triangulate_rays()` for <2 rays

**Input validation:** Minimal — relies on NumPy broadcasting, no strict shape checking

### Numerical Tolerances

- `1e-10` — parallel ray checks
- `1e-9` — Newton-Raphson convergence (meters)
- `1e-6` — flat interface detection

---

## 5. Calibration I/O (JSON Schema v1.0)

### Top-level structure

```json
{
  "version": "1.0",
  "cameras": { "cam0": { ... } },
  "interface": { "normal": [0,0,-1], "n_air": 1.0, "n_water": 1.333 },
  "board": { ... },
  "diagnostics": { ... },
  "metadata": { ... }
}
```

### Camera entry

```json
{
  "name": "cam0",
  "intrinsics": {
    "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "dist_coeffs": [k1, k2, p1, p2, k3],
    "image_size": [640, 480],
    "is_fisheye": false
  },
  "extrinsics": {
    "R": [[...], [...], [...]],
    "t": [tx, ty, tz]
  },
  "water_z": 0.15,
  "is_auxiliary": false
}
```

**Functions:** `save_calibration(result, path)`, `load_calibration(path) -> CalibrationResult`

---

## 6. Public Exports

### Core

```python
BoardGeometry, Camera, undistort_points, Interface,
ray_plane_intersection, snells_law_3d, trace_ray_air_to_water,
refractive_project, refractive_project_batch, refractive_back_project
```

### Triangulation

```python
triangulate_point, triangulate_rays, point_to_ray_distance
```

### Schema

```python
Vec3, Mat3, Vec2,
CameraIntrinsics, CameraExtrinsics, CameraCalibration,
InterfaceParams, CalibrationResult, BoardConfig,
CalibrationError, InsufficientDataError, ConvergenceError, ConnectivityError
```

### Transforms

```python
rvec_to_matrix, matrix_to_rvec, compose_poses, invert_pose, camera_center
```

---

## 7. Implementation Notes

- **All NumPy** — no GPU support, no PyTorch
- **Single-point API** — batch functions exist but are secondary
- **None as error signal** — not exceptions (for most functions)
- **No hidden state** — all parameters explicit, no caching
- **OpenCV dependency** — cv2.projectPoints, cv2.undistortPoints, cv2.fisheye.*, cv2.solvePnP, cv2.Rodrigues
