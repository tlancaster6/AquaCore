# AquaKit Rewiring Guide

This guide maps every AquaCal and AquaMVS import to its AquaKit replacement.
Use the import tables as a find-and-replace reference. Signature changes are
documented where the call site must be updated beyond swapping the import path.
Intentional gaps (modules not ported) are listed with explanations.

## Prerequisites

PyTorch is **not** declared as an aquakit dependency. Users install their preferred
variant (CPU, CUDA, ROCm) separately:

```bash
# Install PyTorch first — choose your variant from https://pytorch.org/get-started/
pip install torch  # CPU-only example

# Then install aquakit
pip install aquakit
```

Failing to install PyTorch before importing aquakit results in:
`ModuleNotFoundError: No module named 'torch'`

---

## AquaCal Users

AquaCal implements geometry with NumPy arrays and OpenCV Camera objects. AquaKit
re-implements the same operations as PyTorch-native, batched functions that work
on any device (CPU or CUDA). The data types at call sites change, but the
mathematical semantics are identical.

### Quick Reference: Import Replacement Table

| Old Import (aquacal) | New Import (aquakit) | Notes |
|---|---|---|
| `from aquacal.core.refractive_geometry import snells_law_3d` | `from aquakit import snells_law_3d` | Signature changed — see below |
| `from aquacal.core.refractive_geometry import trace_ray_air_to_water` | `from aquakit import trace_ray_air_to_water` | Signature changed — see below |
| `from aquacal.core.refractive_geometry import refractive_back_project` | `from aquakit import refractive_back_project` | Signature changed — see below |
| `from aquacal.core.refractive_geometry import refractive_project` | `from aquakit import refractive_project` | Signature changed — see below |
| `from aquacal.core.refractive_geometry import refractive_project_fast` | *(removed)* | Deprecated shim — use `refractive_project` |
| `from aquacal.core.refractive_geometry import refractive_project_fast_batch` | *(removed)* | Deprecated shim — use `refractive_project` |
| `from aquacal.utils.transforms import rvec_to_matrix` | `from aquakit import rvec_to_matrix` | numpy → torch |
| `from aquacal.utils.transforms import matrix_to_rvec` | `from aquakit import matrix_to_rvec` | numpy → torch |
| `from aquacal.utils.transforms import compose_poses` | `from aquakit import compose_poses` | numpy → torch |
| `from aquacal.utils.transforms import invert_pose` | `from aquakit import invert_pose` | numpy → torch |
| `from aquacal.utils.transforms import camera_center` | `from aquakit import camera_center` | numpy → torch |
| `from aquacal.config.schema import CameraIntrinsics` | `from aquakit import CameraIntrinsics` | Tensors inside (K, dist_coeffs) instead of numpy |
| `from aquacal.config.schema import CameraExtrinsics` | `from aquakit import CameraExtrinsics` | Tensors inside (R, t) instead of numpy |
| `from aquacal.config.schema import InterfaceParams` | `from aquakit import InterfaceParams` | `water_z` field added (was separate in AquaCal) |
| `from aquacal.config.schema import Vec2, Vec3, Mat3` | `from aquakit import Vec2, Vec3, Mat3` | Now `torch.Tensor` type aliases (were numpy) |
| `from aquacal.config.schema import INTERFACE_NORMAL` | `from aquakit import INTERFACE_NORMAL` | Now `torch.Tensor([0., 0., -1.])` (was numpy) |
| `from aquacal.io.serialization import load_calibration` | `from aquakit import load_calibration_data` | Name changed; return type changed — see below |
| `from aquacal.io.video import VideoSet` | `from aquakit import VideoSet` | Same FrameSet protocol |
| `from aquacal.io.images import ImageSet` | `from aquakit import ImageSet` | Same FrameSet protocol |
| `from aquacal.io.frameset import FrameSet` | `from aquakit import FrameSet` | Same runtime-checkable Protocol |
| `from aquacal.io.images import create_frameset` | `from aquakit import create_frameset` | Same factory signature |
| `from aquacal.core.interface_model import ray_plane_intersection` | `from aquakit import ray_plane_intersection` | numpy → torch |
| `from aquacal.triangulation.triangulate import triangulate_point` | `from aquakit import triangulate_rays` | API shape changed — see below |

### Signature Changes

#### `snells_law_3d` — Return type for TIR cases

AquaCal returned `None` when total internal reflection (TIR) occurred:

```python
# AquaCal
result = snells_law_3d(incident, normal, n_ratio)
if result is None:
    # TIR case
    ...
refracted_dir = result  # Vec3 (numpy array)
```

AquaKit uses a `(directions, valid)` tuple. TIR rows are zeros in `directions`
and `False` in `valid`:

```python
# AquaKit
directions, valid = snells_law_3d(incident, normal, n_ratio)
# incident: torch.Tensor shape (N, 3)
# directions: torch.Tensor shape (N, 3) — zero for TIR rows
# valid:      torch.Tensor shape (N,) bool — False for TIR rows

# Filter out TIR cases:
refracted = directions[valid]
```

The input is now batched: `incident_directions` has shape `(N, 3)` instead of a
single `Vec3`. Wrap a single vector: `incident.unsqueeze(0)`.

#### `trace_ray_air_to_water` — Takes tensors, not Camera/Interface objects

AquaCal passed `Camera` and `Interface` objects:

```python
# AquaCal
interface_pt, refracted, valid = trace_ray_air_to_water(camera, interface, pixel)
```

AquaKit takes raw tensors and an `InterfaceParams` dataclass:

```python
# AquaKit
from aquakit import trace_ray_air_to_water, InterfaceParams

interface = InterfaceParams(
    normal=torch.tensor([0., 0., -1.]),
    water_z=0.0,
    n_air=1.0,
    n_water=1.333,
)
# origins: shape (N, 3) — camera centers in world frame
# directions: shape (N, 3) — unit ray directions in air (world frame)
interface_points, refracted_dirs, valid = trace_ray_air_to_water(
    origins, directions, interface
)
# Returns three (N, 3), (N, 3), (N,) tensors
```

#### `refractive_project` — Two-step process; returns interface point, not pixel

AquaCal returned a 2D pixel coordinate directly.

AquaKit returns the **interface point on the water surface** (shape `(N, 3)`).
You then project that point through the camera model to get the pixel:

```python
# AquaKit — two steps
from aquakit import refractive_project, create_camera

# Step 1: find the interface point on the water surface
interface_points, valid = refractive_project(
    points,          # (N, 3) underwater 3D points
    camera_center,   # (3,) camera optical center
    interface,       # InterfaceParams
)

# Step 2: project interface point through camera to get pixel
camera = create_camera(intrinsics, extrinsics)
pixels, proj_valid = camera.project(interface_points)
```

#### `refractive_back_project` — Camera objects replaced by tensors

AquaCal: passed a `Camera` object.
AquaKit: passes raw tensors:

```python
# AquaKit
from aquakit import refractive_back_project

# pixel_rays: (N, 3) unit ray directions in air (world frame)
# camera_centers: (N, 3) or (3,) camera optical center(s)
interface_points, water_dirs, valid = refractive_back_project(
    pixel_rays, camera_centers, interface
)
```

#### `load_calibration` → `load_calibration_data` — New name, new return type

AquaCal:

```python
from aquacal.io.serialization import load_calibration
result = load_calibration("calibration.json")  # returns CalibrationResult
cameras = result.cameras  # list of Camera objects (numpy-based)
```

AquaKit:

```python
from aquakit import load_calibration_data
data = load_calibration_data("calibration.json")  # returns CalibrationData
# data.cameras: dict[str, CameraData]
# Each CameraData has: intrinsics (CameraIntrinsics), extrinsics (CameraExtrinsics)
# All tensors are PyTorch float32 on CPU
# data.interface: InterfaceParams (if present in JSON)
```

#### `triangulate_point` → `triangulate_rays` — List of ray tuples, not separate arrays

AquaCal passed separate lists of origins and directions:

```python
# AquaCal
point = triangulate_point(origins_list, directions_list)
```

AquaKit takes a list of `(origin, direction)` tuples, each a `(3,)` tensor:

```python
# AquaKit
from aquakit import triangulate_rays

rays = [
    (origin_cam1, direction_cam1),  # each (3,) float32 tensor
    (origin_cam2, direction_cam2),
]
point = triangulate_rays(rays)  # returns (3,) tensor
```

For refractive triangulation, pass the refracted ray origins (on the water surface)
and refracted directions (in water) as the ray list.

#### `create_camera` — No `name` argument; takes dataclasses, not raw arrays

AquaCal's factory took a string name and raw intrinsic/extrinsic arrays:

```python
# AquaCal
camera = create_camera("cam0", intrinsics_dict, extrinsics_dict)
```

AquaKit takes `CameraIntrinsics` and `CameraExtrinsics` dataclasses directly:

```python
# AquaKit
from aquakit import create_camera, CameraIntrinsics, CameraExtrinsics
import torch

intrinsics = CameraIntrinsics(
    K=torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32),
    dist_coeffs=torch.tensor([k1, k2, p1, p2], dtype=torch.float64),
    image_size=(width, height),
    is_fisheye=False,
)
extrinsics = CameraExtrinsics(
    R=torch.eye(3, dtype=torch.float32),
    t=torch.zeros(3, dtype=torch.float32),
)
camera = create_camera(intrinsics, extrinsics)
# Returns _PinholeCamera or _FisheyeCamera (internal types — use Protocol interface)
pixels, valid = camera.project(points)   # (N, 3) → (N, 2), (N,)
rays = camera.pixel_to_ray(pixels)       # (N, 2) → (N, 3)
```

### NOT PORTED (AquaCal modules with no AquaKit equivalent)

These modules are intentionally absent from AquaKit. They belong to the calibration
pipeline or detection workflow, which are AquaCal-specific concerns.

| AquaCal Module | What It Does | Why Not Ported |
|---|---|---|
| `aquacal.core.camera.Camera` | NumPy Camera class with cv2 projection | AquaKit uses plain tensors + `create_camera`; no numpy Camera class |
| `aquacal.core.camera.FisheyeCamera` | Fisheye variant of above | Same — internal `_FisheyeCamera` not exported |
| `aquacal.core.board` | ChArUco board detection and corner extraction | Calibration-specific; not a geometry primitive |
| `aquacal.core.interface_model.Interface` | NumPy Interface class | AquaKit uses `InterfaceParams` dataclass; no class wrapper |
| `aquacal.calibration.*` | Full calibration optimization pipeline (bundle adjustment etc.) | Calibration-specific pipeline; not ported |
| `aquacal.validation.*` | Reprojection and reconstruction validation tools | Calibration-specific diagnostics; not ported |
| `aquacal.io.detection.*` | ChArUco detection I/O | Calibration-specific; not ported |
| `aquacal.io.serialization.save_calibration` | Write calibration JSON | Write-side not ported; `load_calibration_data` covers read |
| `aquacal.config.schema.CalibrationResult` | Full calibration result struct | Replaced by `CalibrationData` in AquaKit |
| `aquacal.config.schema.BoardConfig` | ChArUco board specification | Calibration-specific; not ported |
| `aquacal.datasets.*` | Synthetic dataset generation | Deferred to v2; not ported |

---

## AquaMVS Users

AquaMVS carries its own copies of calibration and projection modules. These move
to AquaKit without API changes — the signatures are identical.

### Quick Reference: Import Replacement Table

| Old Import (aquamvs) | New Import (aquakit) | Notes |
|---|---|---|
| `from aquamvs.calibration import CalibrationData` | `from aquakit import CalibrationData` | Same structure; identical field names |
| `from aquamvs.calibration import CameraData` | `from aquakit import CameraData` | Same |
| `from aquamvs.calibration import load_calibration_data` | `from aquakit import load_calibration_data` | Identical function, moved |
| `from aquamvs.calibration import compute_undistortion_maps` | `from aquakit import compute_undistortion_maps` | Identical function, moved |
| `from aquamvs.calibration import undistort_image` | `from aquakit import undistort_image` | Identical function, moved |
| `from aquamvs.projection import ProjectionModel` | `from aquakit import ProjectionModel` | Same Protocol |
| `from aquamvs.projection import RefractiveProjectionModel` | `from aquakit import RefractiveProjectionModel` | Same class |
| `from aquamvs.triangulation import triangulate_rays` | `from aquakit import triangulate_rays` | Identical function, moved |

### Signature Changes

There are no signature changes for AquaMVS users. All ported functions and classes
have identical APIs — this is a pure import-path migration.

### NOT PORTED (AquaMVS modules with no AquaKit equivalent)

These modules implement the full MVS pipeline and are out of scope for AquaKit,
which provides only the geometry primitives that the pipeline builds on.

| AquaMVS Module | What It Does | Why Not Ported |
|---|---|---|
| `aquamvs.triangulation.triangulate_pair` | Pair-wise triangulation with quality filtering | Pipeline-specific logic; not a geometry primitive |
| `aquamvs.triangulation.triangulate_all_pairs` | All-pairs triangulation aggregation | Pipeline-specific |
| `aquamvs.triangulation.filter_sparse_cloud` | Point cloud outlier filtering | Pipeline-specific |
| `aquamvs.triangulation.compute_depth_ranges` | Depth range estimation for plane sweep | Pipeline-specific |
| `aquamvs.io.ImageDirectorySet` | Image directory input (partial overlap with `ImageSet`) | `aquakit.ImageSet` covers the core use case |
| `aquamvs.features.*` | Feature extraction and matching (RoMA) | Pipeline-specific; not a geometry primitive |
| `aquamvs.dense.*` | Dense stereo and plane sweep | Pipeline-specific |
| `aquamvs.fusion.*` | Depth map fusion | Pipeline-specific |
| `aquamvs.pipeline.*` | Full MVS pipeline orchestration | Pipeline-specific |
| `aquamvs.evaluation.*` | Metrics and point cloud alignment | Pipeline-specific |
| `aquamvs.visualization.*` | 3D visualization utilities | Pipeline-specific |

---

## New in AquaKit (no equivalent in AquaCal or AquaMVS)

These exports are new to AquaKit. Consumer teams may adopt them to replace
hand-rolled equivalents:

| Export | Module | What It Does |
|---|---|---|
| `trace_ray_water_to_air` | `aquakit.refraction` | Trace underwater rays upward through the surface (TIR-aware) |
| `back_project_multi` | `aquakit.projection` | Batch back-project pixels across multiple cameras via the ProjectionModel Protocol |
| `project_multi` | `aquakit.projection` | Batch project 3D points across multiple cameras via the ProjectionModel Protocol |
| `point_to_ray_distance` | `aquakit.triangulation` | Perpendicular distance from a 3D point to a ray (triangulation quality metric) |
| `create_frameset` | `aquakit.io` | Factory: infer `ImageSet` or `VideoSet` from a filesystem path |
