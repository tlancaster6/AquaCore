# AquaMVS Codebase Reference

Source: `C:\Users\tucke\PycharmProjects\AquaMVS`
Mapped: 2026-02-18

## 1. Module Layout

```
src/aquamvs/
├── calibration.py              # Camera/refraction calibration data loading
├── projection/
│   ├── protocol.py            # ProjectionModel protocol definition
│   └── refractive.py          # RefractiveProjectionModel implementation
├── triangulation.py            # Sparse 3D point triangulation
├── fusion.py                   # Depth map fusion into point clouds
├── io.py                       # Image/video frame I/O adapters
├── surface.py                  # Surface reconstruction
├── coloring.py                 # Best-view color selection
├── preprocess.py               # Temporal median filtering
├── masks.py                    # ROI mask handling
├── config.py                   # Configuration schemas (Pydantic)
├── features/                   # Sparse feature extraction/matching
├── dense/                      # Dense matching (plane sweep)
├── evaluation/                 # Evaluation metrics
├── pipeline/                   # End-to-end pipeline
└── visualization/              # Plotting (excluded from AquaKit scope)
```

---

## 2. Key Classes and Types

### CameraData

```python
@dataclass
class CameraData:
    name: str
    K: torch.Tensor              # (3, 3), float32
    dist_coeffs: torch.Tensor    # (N,), float64
    R: torch.Tensor              # (3, 3), float32
    t: torch.Tensor              # (3,), float32
    image_size: tuple[int, int]  # (width, height)
    is_fisheye: bool
    is_auxiliary: bool
```

- K, R are float32; dist_coeffs stays float64 (OpenCV requirement)
- t normalized to (3,) from AquaCal's potential (3,1)

### CalibrationData

```python
@dataclass
class CalibrationData:
    cameras: dict[str, CameraData]
    water_z: float
    interface_normal: torch.Tensor   # (3,), float32
    n_air: float
    n_water: float
```

**Methods:** `ring_cameras()`, `auxiliary_cameras()`, `camera_positions()`

### UndistortionData

```python
@dataclass
class UndistortionData:
    K_new: torch.Tensor          # (3, 3), float32
    map_x: np.ndarray            # (H, W), float32
    map_y: np.ndarray            # (H, W), float32
```

### ProjectionModel (Protocol)

```python
@runtime_checkable
class ProjectionModel(Protocol):
    def project(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # points: (N, 3) -> pixels: (N, 2), valid: (N,) bool
        ...

    def cast_ray(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # pixels: (N, 2) -> origins: (N, 3), directions: (N, 3)
        ...
```

### RefractiveProjectionModel

```python
class RefractiveProjectionModel:
    def __init__(self, K, R, t, water_z, normal, n_air, n_water): ...
    # Precomputed: K_inv, C (camera center), n_ratio

    def to(self, device) -> self: ...
    def project(self, points: (N, 3)) -> ((N, 2), (N,)): ...
    def cast_ray(self, pixels: (N, 2)) -> ((N, 3), (N, 3)): ...
```

---

## 3. Geometry & Math Functions

### RefractiveProjectionModel.cast_ray()

1. Pinhole back-projection: pixels -> rays via K_inv
2. Camera-to-world: apply R^T
3. Ray-plane intersection at Z = water_z
4. Snell's law refraction (n_ratio)
5. Returns: origins on water surface, unit directions into water

Fully differentiable, no branching.

### RefractiveProjectionModel.project()

1. Compute h_c (camera above water), h_q (point below water)
2. Initial guess: straight-line intersection at water surface
3. Newton-Raphson: **fixed 10 iterations** (for deterministic gradients)
4. Pinhole projection: P -> pixels via K, R, t
5. Validity: below water AND in front of camera

- Clamps r_p to [0, r_q] per iteration
- Epsilon 1e-12 to prevent division by zero
- Invalid pixels set to NaN

### Triangulation

```python
triangulate_rays(rays: list[tuple[Tensor, Tensor]]) -> Tensor  # (3,)
# Same algorithm as AquaCal: A = sum(I - d_i d_i^T)

_triangulate_two_rays_batch(
    origins_a, dirs_a, origins_b, dirs_b  # all (M, 3)
) -> (points: (M, 3), valid: (M,))
# Vectorized via einsum
```

```python
triangulate_pair(
    model_ref, model_src, matches,
    min_angle=2.0,          # degrees
    max_reproj_error=3.0    # pixels
) -> dict  # points_3d, scores, ref_pixels, src_pixels, valid
```

Quality filters: positive depth, minimum angle, reprojection error.

### Undistortion

```python
compute_undistortion_maps(camera: CameraData) -> UndistortionData
undistort_image(image: np.ndarray, undistortion: UndistortionData) -> np.ndarray
```

Dispatches to cv2.fisheye.* or cv2.* based on is_fisheye.

### Depth Fusion

```python
filter_depth_map(ref_name, ref_model, ref_depth, ...) -> (depth, confidence, count)
backproject_depth_map(model, depth_map, image, ...) -> dict  # points, colors, confidence
fuse_depth_maps(ring_cameras, ...) -> o3d.PointCloud
```

---

## 4. Conventions

### Tensor Shapes

| Concept | Shape | Dtype |
|---------|-------|-------|
| 3D point | (N, 3) | float32 |
| 2D pixel | (N, 2) | float32 |
| Ray origin/direction | (N, 3) | float32 |
| Camera K | (3, 3) | float32 |
| Rotation R | (3, 3) | float32 |
| Translation t | (3,) | float32 |
| Distortion coeffs | (N,) | float64 |
| Image | (H, W, 3) uint8 BGR or (H, W) float32 grayscale |
| Depth map | (H, W) | float32, NaN for invalid |

### Coordinate Systems

- **World:** Z-down (into water), same as AquaCal
- **Camera:** Z-forward, X-right, Y-down (OpenCV)
- **Depth convention:** Ray-depth (distance along refracted ray), not Z-coordinate

### Device Handling

- Infer device from input tensors
- `.to(device)` on projection models
- All tensors in a batch must be on same device — no implicit transfers

### Validity Masking

- Return `(output, valid_mask)` — caller decides how to handle
- Invalid pixel values are NaN
- No exceptions for invalid inputs in projection; silent masking

### Error Handling

- Preferred: return (output, valid_mask)
- Exceptions only for: file I/O, config validation, degenerate math (e.g., <2 rays)

### Numerical Tolerances

- Triangulation: det > 1e-6
- Newton-Raphson: 1e-12 epsilon
- NCC: 1e-8 denominator
- Refraction: clamp sin^2(theta) >= 0

---

## 5. I/O Patterns

### ImageDirectorySet

```python
class ImageDirectorySet:
    def read_frame(self, frame_idx: int) -> dict[str, np.ndarray]: ...
    def iterate_frames(self, start, stop, step) -> Iterator: ...
```

- All cameras must have identical file counts
- Files matched by sorted name
- Raises ValueError on mismatch

### Calibration Loading

```python
load_calibration_data(calibration_path: str | Path) -> CalibrationData
# Loads AquaCal JSON, converts to PyTorch tensors
# Handles t shape: (3, 1) -> (3,)
```

**Only AquaCal import:** `from aquacal.io.serialization import load_calibration`

---

## 6. Dependency on AquaCal

### Translation at loading boundary

| AquaCal | AquaMVS | Notes |
|---------|---------|-------|
| intrinsics.K (numpy) | CameraData.K (torch float32) | dtype conversion |
| intrinsics.dist_coeffs | CameraData.dist_coeffs (torch float64) | stays float64 for OpenCV |
| extrinsics.R, t (numpy) | CameraData.R, t (torch float32) | t: (3,1) -> (3,) |
| result.interface | CalibrationData fields | shape: (3,1) -> (3,) |

AquaMVS does NOT depend on AquaCal data structures after loading.

---

## 7. Implementation Notes

- **All PyTorch** — GPU-ready, differentiable
- **Batch-first API** — (N, 3) shapes throughout
- **Fixed iteration count** in Newton-Raphson — for deterministic gradients
- **Pydantic v2** for configuration
- **Protocol-based** projection model — duck typing via runtime_checkable
- **Tests parametrized** over CPU/CUDA with `torch.testing.assert_close()`
