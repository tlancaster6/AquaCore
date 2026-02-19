# Phase 3: Calibration and Undistortion - Research

**Researched:** 2026-02-18
**Domain:** AquaCal JSON calibration loading, PyTorch dataclass design, OpenCV undistortion maps
**Confidence:** HIGH — research drawn directly from AquaMVS and AquaCal source code on this machine

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Loader API surface:**
- `load_calibration_data()` accepts a file path (str | Path) OR a pre-parsed dict — flexible for testing and pipelines
- Returns `CalibrationData` with a single global `water_z` (world-frame property of the rig, not per-camera)
- `CalibrationData.cameras` is `dict[str, CameraData]` keyed by camera name
- Ordered list property for index-based iteration (e.g., `calibration.camera_list`)
- `CameraData.is_auxiliary` flag; `CalibrationData.core_cameras()` and `CalibrationData.auxiliary_cameras()` helper methods
- `CameraData.name` stores the string identifier from the JSON key

**Schema tolerance:**
- Strict validation on required fields (cameras, interface); silently ignore optional sections (board, diagnostics, metadata)
- Normalize `t` shape from (3,1) to (3,) silently — known AquaCal quirk, not a user error
- Check `version` field; warn on unknown version but attempt to load anyway
- If a camera entry is missing a required field: skip that camera with a warning, don't fail the entire load

**Undistortion pipeline:**
- `compute_undistortion_maps(camera_data: CameraData)` accepts CameraData object directly
- Returns NumPy `(map_x, map_y)` tuple — maps stay in NumPy since cv2.remap requires it
- No UndistortionData wrapper dataclass — just the map tuple
- `undistort_image()` accepts and returns PyTorch tensors — converts to NumPy internally for cv2.remap
- Fisheye vs pinhole dispatch based on `is_fisheye` flag (same as AquaCal/AquaMVS)

**Type mapping:**
- `CameraData` composes existing Phase 1 types: `intrinsics: CameraIntrinsics`, `extrinsics: CameraExtrinsics`
- `CalibrationData` stores `interface: InterfaceParams` (reuses Phase 1 type)
- `CameraData` and `CalibrationData` live in `calibration.py` alongside the loader function
- `CameraData.name: str` carries the JSON key for logging and dict reconstruction

**Claude's Discretion:**
- Exact warning/logging mechanism (Python warnings module vs logging)
- How `camera_list` property orders cameras (insertion order from JSON, alphabetical, etc.)
- Internal conversion details for tensor ↔ NumPy at cv2 boundaries
- Error message wording for validation failures

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

---

## Summary

Phase 3 delivers `calibration.py` and `undistortion.py` — the two stubs that are currently empty one-liners. The work is almost entirely a port of AquaMVS's `calibration.py` with three specific divergences from the AquaMVS reference: (1) no AquaCal import dependency — parse raw JSON directly; (2) `CameraData` composes Phase 1 `CameraIntrinsics`/`CameraExtrinsics` typed objects rather than storing raw tensors directly; (3) `undistort_image()` accepts and returns PyTorch tensors rather than NumPy arrays, with the NumPy conversion happening internally.

The AquaCal JSON schema is fully documented in `aquacal/io/serialization.py` and `aquacal/config/schema.py`, both readable locally. The top-level format has two required sections (`cameras`, `interface`) and four optional/ignorable sections (`board`, `diagnostics`, `metadata`, `version`). Within a camera entry, all fields are required except `is_fisheye` (defaults to False) and `is_auxiliary` (defaults to False). The one known quirk is that `t` may be serialized as either `[tx, ty, tz]` (shape 3) or `[[tx], [ty], [tz]]` (shape 3×1) — normalize to (3,) silently. The `water_z` field lives inside each camera entry in the AquaCal schema, but the decision is to extract a single global value for `CalibrationData`.

The undistortion pipeline is a near-verbatim port of AquaMVS's `compute_undistortion_maps` plus a reshaped `undistort_image`. The key difference from AquaMVS is that AquaMVS wraps maps in an `UndistortionData` dataclass; this phase returns raw NumPy `(map_x, map_y)` tuples. The `undistort_image` function must convert the input PyTorch tensor to a NumPy array for `cv2.remap`, then convert the result back to a PyTorch tensor on the same device — following the project's established OpenCV boundary pattern.

**Primary recommendation:** Parse JSON directly with `json.load()`, compose Phase 1 types (`CameraIntrinsics`, `CameraExtrinsics`, `InterfaceParams`), put `CameraData`/`CalibrationData` and the loader in `calibration.py`, and put the undistortion functions in `undistortion.py`. No new dependencies needed.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | >=2.0 | Tensor types for K, R, t, dist_coeffs, interface_normal | Project-wide requirement; Phase 1 types use torch.Tensor |
| numpy | >=1.24 | cv2 boundary conversion; map_x/map_y stay as np.ndarray | cv2.remap and cv2.fisheye.* require numpy float32 arrays |
| opencv-python | >=4.8 | `initUndistortRectifyMap`, `fisheye.initUndistortRectifyMap`, `remap`, `getOptimalNewCameraMatrix`, `fisheye.estimateNewCameraMatrixForUndistortRectify` | Only library providing calibrated undistortion maps; standard in all Python CV workflows |
| json (stdlib) | stdlib | Parse AquaCal JSON without AquaCal dependency | Avoids AquaCal import; schema is stable JSON |
| warnings (stdlib) | stdlib | Warn on unknown version, skip-with-warning on bad camera entry | Lightweight; appropriate for data loading warnings (vs logging which needs setup) |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pathlib.Path | stdlib | Accept str\|Path in load_calibration_data() | Normalize path input consistently |
| dataclasses | stdlib | `@dataclass` for CameraData, CalibrationData | Same pattern as Phase 1 types; pure data containers |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `warnings` module | Python `logging` module | `warnings` requires no caller setup; emits once-per-location by default; appropriate for data quality warnings during loading. `logging` is better for applications. Since AquaKit is a library, `warnings` is the standard choice. |
| json.load() direct | pydantic, marshmallow, jsonschema | Zero extra dependencies; the schema is small enough that manual validation is readable. Pydantic would add a dependency and boilerplate. The AquaMVS reference uses no schema library. |
| Return (map_x, map_y) tuple | `UndistortionData` dataclass | Tuple is simpler; no wrapper needed since Phase 3 decision explicitly rejects the wrapper. AquaMVS's `UndistortionData` also carries `K_new` — but the decision here omits K_new from the return value. |

**Installation:** No new dependencies. `torch`, `numpy`, `opencv-python` are already in `pyproject.toml`.

---

## Architecture Patterns

### Module-to-File Mapping

Phase 3 fills exactly two stubs:

```
src/aquakit/
├── calibration.py       # CameraData, CalibrationData, load_calibration_data()
└── undistortion.py      # compute_undistortion_maps(), undistort_image()
```

### Pattern 1: CameraData Composes Phase 1 Types

**What:** `CameraData` is a `@dataclass` that stores `intrinsics: CameraIntrinsics` and `extrinsics: CameraExtrinsics` instead of raw K, R, t tensors. This is the key AquaKit divergence from AquaMVS, which stored raw tensors directly on `CameraData`.

**When to use:** Always — this is the locked design decision.

**Why:** Phase 1's `CameraIntrinsics` and `CameraExtrinsics` are the canonical typed representations. Composing them in `CameraData` makes the types self-describing and enables `create_camera(camera_data.intrinsics, camera_data.extrinsics)` without any field extraction.

```python
# Source: types.py (Phase 1) + this phase's design
from dataclasses import dataclass
from .types import CameraIntrinsics, CameraExtrinsics

@dataclass
class CameraData:
    """Per-camera calibration data.

    Attributes:
        name: Camera identifier string (from JSON key).
        intrinsics: Intrinsic parameters (K, dist_coeffs, image_size, is_fisheye).
        extrinsics: Extrinsic parameters (R, t world-to-camera).
        is_auxiliary: True if registered post-hoc against fixed poses.
    """
    name: str
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    is_auxiliary: bool = False
```

### Pattern 2: CalibrationData Composes InterfaceParams

**What:** `CalibrationData` stores `interface: InterfaceParams` (Phase 1 type) rather than flat `interface_normal`, `n_air`, `n_water` fields. Adds `water_z: float` as a global rig property.

**Note on water_z:** AquaCal stores `water_z` per-camera (in `CameraCalibration.water_z`). AquaMVS reads `next(iter(result.cameras.values())).water_z` to get the global value. AquaKit's decision follows the same approach — extract a single `water_z` from the first valid camera entry during loading.

```python
# Source: types.py (Phase 1 InterfaceParams) + this phase's design
@dataclass
class CalibrationData:
    """Complete calibration data.

    Attributes:
        cameras: Per-camera data keyed by camera name string.
        water_z: Z-coordinate of water surface in world frame (meters).
            Global rig property, same for all cameras after optimization.
        interface: Refractive interface parameters (normal, n_air, n_water).
    """
    cameras: dict[str, CameraData]
    water_z: float
    interface: InterfaceParams

    @property
    def camera_list(self) -> list[CameraData]:
        """Cameras in insertion order for index-based access."""
        return list(self.cameras.values())

    def core_cameras(self) -> dict[str, CameraData]:
        """Non-auxiliary cameras (keyed by name)."""
        return {k: v for k, v in self.cameras.items() if not v.is_auxiliary}

    def auxiliary_cameras(self) -> dict[str, CameraData]:
        """Auxiliary cameras (keyed by name)."""
        return {k: v for k, v in self.cameras.items() if v.is_auxiliary}
```

### Pattern 3: load_calibration_data() with Dual Input

**What:** Function accepts `str | Path | dict`. When given a path, calls `json.load()`. When given a dict, uses it directly. This enables testing without writing files.

**Version check and tolerance:** Warn on unknown version (don't raise); skip individual camera entries with missing required fields (warn, don't fail entire load).

```python
# Source: AquaCal serialization.py + AquaMVS calibration.py (adapted — no AquaCal import)
import json
import warnings
from pathlib import Path
from typing import Any

def load_calibration_data(source: str | Path | dict[str, Any]) -> CalibrationData:
    """Load AquaCal calibration from a JSON file or pre-parsed dict.

    Args:
        source: Path to AquaCal JSON file, or a pre-parsed dict for testing.

    Returns:
        CalibrationData with all parameters as PyTorch tensors.

    Raises:
        FileNotFoundError: If source is a path and file does not exist.
        ValueError: If required top-level fields (cameras, interface) are missing.
    """
    if isinstance(source, dict):
        data = source
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")
        with open(path) as f:
            data = json.load(f)

    # Version check — warn, don't fail
    version = data.get("version")
    if version != "1.0":
        warnings.warn(
            f"Unknown calibration version '{version}'; expected '1.0'. "
            "Attempting to load anyway.",
            stacklevel=2,
        )

    # Required top-level fields
    if "cameras" not in data:
        raise ValueError("Missing required field 'cameras' in calibration data.")
    if "interface" not in data:
        raise ValueError("Missing required field 'interface' in calibration data.")

    # Parse cameras (skip individual bad entries with warning)
    cameras: dict[str, CameraData] = {}
    water_z: float | None = None
    for cam_name, cam_dict in data["cameras"].items():
        camera_data, cam_water_z = _parse_camera_entry(cam_name, cam_dict)
        if camera_data is None:
            continue
        cameras[cam_name] = camera_data
        if water_z is None:
            water_z = cam_water_z

    if not cameras:
        raise ValueError("No valid camera entries could be loaded.")

    # Parse interface
    interface = _parse_interface(data["interface"])

    return CalibrationData(
        cameras=cameras,
        water_z=water_z,  # type: ignore[arg-type]
        interface=interface,
    )
```

### Pattern 4: t Shape Normalization (Known AquaCal Quirk)

**What:** AquaCal's `CameraExtrinsics.t` docstring says `Vec3` (shape 3,) but some older serialized files produce `[[tx], [ty], [tz]]` (shape 3×1 when deserialized). Normalize silently at load time.

**Source:** AquaMVS calibration.py lines 131-136 (verified):

```python
# Source: AquaMVS/src/aquamvs/calibration.py (t shape normalization)
t_numpy = cam_dict["extrinsics"]["t"]  # might be [[tx], [ty], [tz]]
t_array = np.array(t_numpy, dtype=np.float64)
if t_array.ndim == 2:
    t_array = t_array.squeeze()  # (3, 1) -> (3,)
t = torch.from_numpy(t_array).to(torch.float32)
```

### Pattern 5: compute_undistortion_maps() Pinhole/Fisheye Dispatch

**What:** Dispatch on `camera_data.intrinsics.is_fisheye`. Both paths use OpenCV at the NumPy boundary. Returns `(map_x, map_y)` as `np.ndarray` objects.

**Critical detail — dist_coeffs reshape for fisheye:** `cv2.fisheye.*` requires dist_coeffs as shape `(4, 1)`, not `(4,)`. The pinhole path uses dist_coeffs as-is.

**Source:** AquaMVS compute_undistortion_maps() (adapted — no UndistortionData wrapper, returns tuple):

```python
# Source: AquaMVS/src/aquamvs/calibration.py compute_undistortion_maps() (adapted)
import cv2
import numpy as np

def compute_undistortion_maps(
    camera_data: CameraData,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute undistortion remap tables for a camera.

    Args:
        camera_data: Per-camera calibration data from load_calibration_data().

    Returns:
        Tuple of (map_x, map_y), both numpy float32 arrays of shape (H, W).
        Pass directly to cv2.remap or undistort_image().
    """
    K_np = camera_data.intrinsics.K.cpu().numpy().astype(np.float64)
    dist_np = camera_data.intrinsics.dist_coeffs.cpu().numpy().astype(np.float64)
    image_size = camera_data.intrinsics.image_size  # (width, height)

    if camera_data.intrinsics.is_fisheye:
        D = dist_np.reshape(4, 1)  # cv2.fisheye requires (4, 1) not (4,)
        K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K_np, D, image_size, np.eye(3)
        )
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
            K_np, D, np.eye(3), K_new, image_size, cv2.CV_32FC1
        )
    else:
        K_new, _roi = cv2.getOptimalNewCameraMatrix(
            K_np, dist_np, image_size, alpha=0, newImgSize=image_size
        )
        map_x, map_y = cv2.initUndistortRectifyMap(
            K_np, dist_np, None, K_new, image_size, cv2.CV_32FC1
        )

    return map_x, map_y
```

### Pattern 6: undistort_image() PyTorch Tensor I/O

**What:** Accepts a PyTorch tensor, converts to NumPy for `cv2.remap`, converts result back to PyTorch tensor on the original device.

**Tensor format:** AquaKit images are `(H, W, 3)` uint8 BGR (OpenCV convention) or `(H, W)` float32 grayscale. `cv2.remap` handles both. The function preserves dtype and shape.

**Device handling:** The image tensor may be on CUDA. Convert with `.cpu().numpy()` before cv2, then use `torch.from_numpy(...).to(device)` after.

```python
# Source: AquaMVS undistort_image() + OpenCV boundary pattern from camera.py
def undistort_image(
    image: torch.Tensor,
    maps: tuple[np.ndarray, np.ndarray],
) -> torch.Tensor:
    """Apply precomputed undistortion to an image tensor.

    Args:
        image: Input image tensor. Shape (H, W, 3) uint8 or (H, W) float32.
        maps: Precomputed map pair from compute_undistortion_maps().

    Returns:
        Undistorted image tensor, same shape, dtype, and device as input.
    """
    device = image.device
    map_x, map_y = maps
    image_np = image.detach().cpu().numpy()
    result_np = cv2.remap(image_np, map_x, map_y, cv2.INTER_LINEAR)
    return torch.from_numpy(result_np).to(device)
```

### Recommended Project Structure

```
src/aquakit/
├── calibration.py       # CameraData, CalibrationData, load_calibration_data()
│                        # Imports: json, warnings, pathlib, numpy, torch, types.py
└── undistortion.py      # compute_undistortion_maps(), undistort_image()
                         # Imports: cv2, numpy, torch, calibration.CameraData
```

### Anti-Patterns to Avoid

- **Importing from aquacal:** AquaKit must be importable without AquaCal installed. Parse JSON directly with `json.load()`. Never `from aquacal.io.serialization import load_calibration`.
- **Storing raw tensors on CameraData instead of composing types:** `CameraData` must hold `intrinsics: CameraIntrinsics` and `extrinsics: CameraExtrinsics` — not flat K, R, t fields. This enables `create_camera(camera_data.intrinsics, camera_data.extrinsics)` seamlessly.
- **Failing entire load on one bad camera:** The decision is to skip bad camera entries with a warning. Do not raise on per-camera parsing failures.
- **Not normalizing t shape:** AquaCal can produce `t` as `[[tx], [ty], [tz]]`. If AquaKit fails on this, existing AquaCal JSON files will break silently.
- **Not reshaping dist_coeffs for fisheye:** `cv2.fisheye.initUndistortRectifyMap` requires `D` as shape `(4, 1)`, not `(4,)`. Passing `(4,)` raises an OpenCV error.
- **Wrapping maps in a dataclass:** The decision explicitly rejects an `UndistortionData` wrapper. Return a plain `(map_x, map_y)` tuple.
- **Moving image to device inside undistort_image before remap:** cv2.remap requires NumPy on CPU. The pattern is `.detach().cpu().numpy()` before remap, then `.to(device)` after.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Undistortion remap table generation | Custom pixel-wise undistortion loop | `cv2.initUndistortRectifyMap` / `cv2.fisheye.initUndistortRectifyMap` | Handles camera matrix optimization (`alpha` parameter), all distortion models, efficient C++ implementation |
| Optimal new camera matrix | Custom FOV calculation | `cv2.getOptimalNewCameraMatrix(alpha=0)` | `alpha=0` crops to valid pixels only — non-trivial to implement correctly for arbitrary distortion |
| JSON schema validation | Custom field-existence checker | Python dict `.get()` with explicit required-field checks | Schema is small enough that manual checks are clearer than adding a validation library |
| Fisheye matrix estimation | Custom scaling formula | `cv2.fisheye.estimateNewCameraMatrixForUndistortRectify` | Handles fisheye-specific FOV math |

**Key insight:** The undistortion pipeline is entirely OpenCV — do not implement any map generation math manually. The calibration loading is entirely stdlib json — do not add schema validation libraries.

---

## Common Pitfalls

### Pitfall 1: dist_coeffs Shape for Fisheye

**What goes wrong:** `cv2.fisheye.initUndistortRectifyMap(K, D, ...)` raises `cv2.error: (-215:Assertion failed) D.depth() == CV_64F && D.cols == 1 && D.rows == 4`.

**Why it happens:** `CameraIntrinsics.dist_coeffs` for fisheye is stored as shape `(4,)` (1D). OpenCV's fisheye functions require exactly `(4, 1)` (column vector).

**How to avoid:** Always reshape: `D = dist_np.reshape(4, 1)` before calling any `cv2.fisheye.*` function.

**Warning signs:** Works for pinhole cameras, fails immediately for fisheye cameras with a cryptic OpenCV assertion error.

### Pitfall 2: AquaCal Import at Module Level

**What goes wrong:** `import aquakit` raises `ModuleNotFoundError: No module named 'aquacal'` even when user has not installed AquaCal.

**Why it happens:** A module-level import of AquaCal (e.g., `from aquacal.io.serialization import load_calibration`) runs when the `aquakit` package is imported, regardless of whether the calibration functions are called.

**How to avoid:** Parse JSON directly with `json.load()` — zero AquaCal imports. Never import AquaCal anywhere in AquaKit.

**Warning signs:** AquaKit test suite fails in CI unless AquaCal is also installed. This violates the phase success criterion that "aquakit is importable with AquaCal uninstalled."

### Pitfall 3: Failing Loudly on Per-Camera Missing Fields

**What goes wrong:** A calibration file with one malformed camera entry raises `ValueError` and loads nothing.

**Why it happens:** Naive implementation raises on first missing key rather than skipping.

**How to avoid:** Wrap per-camera parsing in `try/except`, catch `KeyError`/`ValueError`, emit `warnings.warn()` with the camera name, and `continue` to the next camera.

**Warning signs:** Adding an auxiliary camera to an existing calibration file breaks loading for all existing cameras.

### Pitfall 4: water_z Extraction from Empty Camera Dict

**What goes wrong:** If all camera entries are skipped (all malformed), `water_z` remains `None` and `CalibrationData(water_z=None, ...)` creates a type-incorrect object.

**Why it happens:** `water_z` is extracted from the first successfully parsed camera entry. If no camera is parsed, the value is never set.

**How to avoid:** After the camera parsing loop, check `if not cameras: raise ValueError("No valid camera entries could be loaded.")` — this also catches the `water_z is None` case.

**Warning signs:** `CalibrationData.water_z` is `None` despite appearing to load successfully.

### Pitfall 5: undistort_image() CUDA Tensor Round-Trip

**What goes wrong:** `cv2.remap(image_np, ...)` fails with `TypeError: Expected Ptr<cv::UMat> for argument 'src'` or similar, OR the result tensor is on CPU when the caller expected it on the original CUDA device.

**Why it happens:** (a) The NumPy conversion of a CUDA tensor must use `.detach().cpu()` first — skipping `.cpu()` raises. (b) `torch.from_numpy(result_np)` always returns a CPU tensor; forgetting `.to(device)` silently produces a device mismatch downstream.

**How to avoid:** Always: `device = image.device`, then `image.detach().cpu().numpy()`, then `torch.from_numpy(result).to(device)`. This is the established pattern in `camera.py` (see `_to_numpy()` and `_rays_to_world()`).

**Warning signs:** Undistorted images mysteriously on CPU when input was on CUDA; downstream device mismatch errors in projection.

### Pitfall 6: image_size Convention for OpenCV

**What goes wrong:** `cv2.initUndistortRectifyMap(..., imageSize, ...)` expects `(width, height)` but many NumPy/PyTorch shapes are `(height, width)`. Passing the wrong order produces transposed maps.

**Why it happens:** AquaKit's `CameraIntrinsics.image_size` stores `(width, height)` per project convention. NumPy arrays are `(H, W, ...)`. Confusing the two is easy.

**How to avoid:** Pass `camera_data.intrinsics.image_size` directly to `imageSize` arguments — it is already in `(width, height)` order, which is what OpenCV expects. Never derive image_size from a numpy array's `.shape` (which gives H, W).

**Warning signs:** Maps are computed for a transposed image size; `map_x` and `map_y` shapes are `(W, H)` instead of `(H, W)`.

### Pitfall 7: InterfaceParams Does Not Include water_z

**What goes wrong:** Caller accesses `calibration_data.interface.water_z` expecting the rig's water depth, but `InterfaceParams` has no `water_z` field — the field lives on `CalibrationData.water_z`.

**Why it happens:** Phase 1's `InterfaceParams` includes `water_z` as a field. But `CalibrationData.interface` is an `InterfaceParams` for the normal and refractive indices only. The global `water_z` is stored separately on `CalibrationData`.

**How to avoid:** This is by design — `water_z` is on `CalibrationData` directly, not on its `interface` sub-field. Document clearly in `CalibrationData` docstring.

**Warning signs:** Callers use `calibration.interface.water_z` instead of `calibration.water_z`.

---

## Code Examples

Verified patterns from AquaMVS and AquaCal source code:

### JSON Schema — Complete AquaCal v1.0 Format

```json
{
  "version": "1.0",
  "cameras": {
    "cam0": {
      "name": "cam0",
      "intrinsics": {
        "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "dist_coeffs": [k1, k2, p1, p2, k3],
        "image_size": [640, 480],
        "is_fisheye": false
      },
      "extrinsics": {
        "R": [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]],
        "t": [tx, ty, tz]
      },
      "water_z": 0.15,
      "is_auxiliary": false
    }
  },
  "interface": {
    "normal": [0.0, 0.0, -1.0],
    "n_air": 1.0,
    "n_water": 1.333
  },
  "board": { "...": "..." },
  "diagnostics": { "...": "..." },
  "metadata": { "...": "..." }
}
```

**Required fields:** `cameras`, `interface`, and within each camera: `intrinsics.K`, `intrinsics.dist_coeffs`, `intrinsics.image_size`, `extrinsics.R`, `extrinsics.t`, `water_z`.

**Optional fields (ignore silently):** `board`, `diagnostics`, `metadata`, `is_fisheye` (default False), `is_auxiliary` (default False).

**Backward compatibility field:** `interface_distance` is an old name for `water_z` in camera entries. AquaCal's own deserializer handles this (see `_deserialize_camera_calibration()`). AquaKit should handle it too for legacy files.

### Parsing a Camera Entry

```python
# Source: AquaCal/src/aquacal/io/serialization.py + AquaMVS/src/aquamvs/calibration.py
import numpy as np
import torch
from .types import CameraIntrinsics, CameraExtrinsics

def _parse_camera_entry(
    cam_name: str,
    cam_dict: dict,
) -> tuple["CameraData | None", "float | None"]:
    """Parse one camera JSON entry. Returns (CameraData, water_z) or (None, None)."""
    try:
        intr = cam_dict["intrinsics"]
        extr = cam_dict["extrinsics"]

        K = torch.from_numpy(
            np.array(intr["K"], dtype=np.float32)
        )
        dist_coeffs = torch.from_numpy(
            np.array(intr["dist_coeffs"], dtype=np.float64)
        )
        image_size = tuple(int(x) for x in intr["image_size"])
        is_fisheye = intr.get("is_fisheye", False)

        R = torch.from_numpy(
            np.array(extr["R"], dtype=np.float32)
        )
        t_raw = np.array(extr["t"], dtype=np.float32)
        if t_raw.ndim == 2:
            t_raw = t_raw.squeeze()  # (3, 1) -> (3,)
        t = torch.from_numpy(t_raw)

        # water_z: accept both 'water_z' (current) and 'interface_distance' (legacy)
        if "water_z" in cam_dict:
            water_z = float(cam_dict["water_z"])
        elif "interface_distance" in cam_dict:
            water_z = float(cam_dict["interface_distance"])
        else:
            raise KeyError("water_z")

        is_auxiliary = cam_dict.get("is_auxiliary", False)

        intrinsics = CameraIntrinsics(
            K=K,
            dist_coeffs=dist_coeffs,
            image_size=image_size,
            is_fisheye=is_fisheye,
        )
        extrinsics = CameraExtrinsics(R=R, t=t)

        camera_data = CameraData(
            name=cam_name,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            is_auxiliary=is_auxiliary,
        )
        return camera_data, water_z

    except (KeyError, ValueError) as exc:
        import warnings
        warnings.warn(
            f"Skipping camera '{cam_name}': missing required field {exc}.",
            stacklevel=3,
        )
        return None, None
```

### Parsing Interface Parameters

```python
# Source: AquaCal/src/aquacal/io/serialization.py _deserialize_interface_params (adapted)
from .types import InterfaceParams

def _parse_interface(interface_dict: dict) -> InterfaceParams:
    normal = torch.tensor(interface_dict["normal"], dtype=torch.float32)
    if normal.ndim == 2:
        normal = normal.squeeze()  # handle potential (3,1) shape
    return InterfaceParams(
        normal=normal,
        water_z=0.0,  # placeholder — water_z is stored on CalibrationData, not here
        n_air=float(interface_dict["n_air"]),
        n_water=float(interface_dict["n_water"]),
    )
```

**Note:** `InterfaceParams` has a `water_z` field (from Phase 1 types.py). When constructing the interface portion, populate `water_z` from the global `CalibrationData.water_z`, or use 0.0 as placeholder since callers will use `CalibrationData.water_z` directly.

### compute_undistortion_maps() — Full Implementation

```python
# Source: AquaMVS/src/aquamvs/calibration.py compute_undistortion_maps() (adapted — returns tuple, not UndistortionData)
import cv2
import numpy as np

def compute_undistortion_maps(
    camera_data: "CameraData",
) -> tuple[np.ndarray, np.ndarray]:
    K_np = camera_data.intrinsics.K.cpu().numpy().astype(np.float64)
    dist_np = camera_data.intrinsics.dist_coeffs.cpu().numpy().astype(np.float64)
    image_size = camera_data.intrinsics.image_size  # (width, height) — correct for cv2

    if camera_data.intrinsics.is_fisheye:
        D = dist_np.reshape(4, 1)  # REQUIRED: cv2.fisheye needs (4, 1) not (4,)
        K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K_np, D, image_size, np.eye(3)
        )
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
            K_np, D, np.eye(3), K_new, image_size, cv2.CV_32FC1
        )
    else:
        K_new, _roi = cv2.getOptimalNewCameraMatrix(
            K_np, dist_np, image_size, alpha=0, newImgSize=image_size
        )
        map_x, map_y = cv2.initUndistortRectifyMap(
            K_np, dist_np, None, K_new, image_size, cv2.CV_32FC1
        )

    return map_x, map_y
```

### undistort_image() — Tensor I/O

```python
# Source: AquaMVS undistort_image() (adapted — accepts/returns torch.Tensor)
# OpenCV boundary pattern: camera.py _to_numpy() + _rays_to_world()
import cv2
import numpy as np
import torch

def undistort_image(
    image: torch.Tensor,
    maps: tuple[np.ndarray, np.ndarray],
) -> torch.Tensor:
    device = image.device
    map_x, map_y = maps
    image_np = image.detach().cpu().numpy()
    result_np = cv2.remap(image_np, map_x, map_y, cv2.INTER_LINEAR)
    return torch.from_numpy(result_np).to(device)
```

### Test: Known-Value Calibration Loading

```python
# Pattern: construct minimal dict input to test loading without file I/O
import torch
from aquakit.calibration import load_calibration_data

MINIMAL_CALIBRATION_DICT = {
    "version": "1.0",
    "cameras": {
        "cam0": {
            "name": "cam0",
            "intrinsics": {
                "K": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
                "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
                "image_size": [640, 480],
            },
            "extrinsics": {
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "t": [0.0, 0.0, 0.0],
            },
            "water_z": 1.0,
        }
    },
    "interface": {
        "normal": [0.0, 0.0, -1.0],
        "n_air": 1.0,
        "n_water": 1.333,
    },
}

def test_load_calibration_from_dict():
    calib = load_calibration_data(MINIMAL_CALIBRATION_DICT)
    assert "cam0" in calib.cameras
    assert calib.water_z == 1.0
    cam = calib.cameras["cam0"]
    assert cam.intrinsics.K.shape == (3, 3)
    assert cam.intrinsics.dist_coeffs.dtype == torch.float64
    assert cam.extrinsics.t.shape == (3,)
```

### Test: Undistortion Maps Shape

```python
# Maps must have shape (H, W), not (W, H)
def test_undistortion_maps_shape(device):
    calib = load_calibration_data(MINIMAL_CALIBRATION_DICT)
    cam = calib.cameras["cam0"]
    map_x, map_y = compute_undistortion_maps(cam)
    W, H = cam.intrinsics.image_size  # (width, height) convention
    assert map_x.shape == (H, W)   # maps are (H, W)
    assert map_y.shape == (H, W)
    assert map_x.dtype == np.float32
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| AquaMVS: import from aquacal.io.serialization | AquaKit: json.load() direct parse | Phase 3 (this phase) | AquaKit importable without AquaCal installed |
| AquaMVS: CameraData with flat K, R, t tensors | AquaKit: CameraData with intrinsics/extrinsics typed objects | Phase 3 (this phase) | `create_camera(cam.intrinsics, cam.extrinsics)` works directly |
| AquaMVS: UndistortionData dataclass with K_new | AquaKit: plain (map_x, map_y) tuple | Phase 3 (this phase) | Simpler API; K_new dropped since maps already encode the new camera matrix |
| AquaMVS: undistort_image() takes np.ndarray | AquaKit: undistort_image() takes torch.Tensor | Phase 3 (this phase) | Consistent with project convention; NumPy conversion happens internally |
| AquaMVS: ring_cameras() method | AquaKit: core_cameras() method | Phase 3 (this phase) | Term "ring cameras" is AquaMVS-specific; AquaKit uses "core cameras" |
| AquaMVS: camera_positions() method | Not in AquaKit CalibrationData | Phase 3 — omitted | CameraExtrinsics.C property covers this; no need to duplicate on CalibrationData |

**Deprecated/outdated:**
- `UndistortionData` wrapper class: AquaMVS uses it; AquaKit drops it. Consumers that need `K_new` must recompute it separately.
- `ring_cameras()` method name: AquaMVS calls non-auxiliary cameras "ring cameras". AquaKit calls them "core cameras". Rewiring guide must document this rename.

---

## Open Questions

1. **InterfaceParams.water_z when constructing the interface field**
   - What we know: `InterfaceParams` has a `water_z` field (from Phase 1). When parsing the interface section of AquaCal JSON, there is no `water_z` in `interface` — it lives in each camera entry. `CalibrationData.water_z` holds the global value.
   - What's unclear: Should `CalibrationData.interface.water_z` be set to the same value as `CalibrationData.water_z`, or left as a placeholder (0.0)?
   - Recommendation: Set `interface.water_z = water_z` (same value as `CalibrationData.water_z`) so that code that passes `CalibrationData.interface` directly to `RefractiveProjectionModel.from_camera(camera, interface)` works correctly. The `from_camera()` factory reads `interface.water_z` to set the model's `water_z`.

2. **camera_list ordering**
   - What we know: Claude's discretion — could be insertion order from JSON or alphabetical.
   - What's unclear: JSON dict insertion order is preserved in Python 3.7+ (and AquaKit requires Python 3.11+). Alphabetical sorting is more deterministic across serialization libraries.
   - Recommendation: Use insertion order from JSON for `camera_list` (return `list(self.cameras.values())`). This preserves the original file's camera ordering, which may be meaningful (e.g., cameras are indexed). Document this. If alphabetical is needed, `sorted(self.cameras.values(), key=lambda c: c.name)` is easy.

3. **Should undistortion.py import from calibration.py?**
   - What we know: `compute_undistortion_maps(camera_data: CameraData)` takes a `CameraData` argument, so `undistortion.py` must import `CameraData` from `calibration.py`.
   - What's unclear: This creates a `calibration → undistortion` dependency chain which is fine per the architecture (undistortion depends on calibration, not the reverse).
   - Recommendation: Yes, `undistortion.py` imports `CameraData` from `.calibration`. This is consistent with the layered architecture (ARCHITECTURE.md layer 4: undistortion depends on calibration). No circular import risk.

---

## Sources

### Primary (HIGH confidence — direct source code inspection)

- `C:/Users/tucke/PycharmProjects/AquaMVS/src/aquamvs/calibration.py` — Complete reference implementation: CameraData, CalibrationData, UndistortionData, load_calibration_data(), compute_undistortion_maps(), undistort_image()
- `C:/Users/tucke/PycharmProjects/AquaCal/src/aquacal/io/serialization.py` — AquaCal JSON serialization: exact schema structure, field names, backward-compat `interface_distance`, version constant "1.0"
- `C:/Users/tucke/PycharmProjects/AquaCal/src/aquacal/config/schema.py` — AquaCal type definitions: CameraCalibration.water_z placement, InterfaceParams fields (no water_z!), CameraIntrinsics/CameraExtrinsics shapes
- `C:/Users/tucke/PycharmProjects/AquaKit/src/aquakit/types.py` — Phase 1 types: CameraIntrinsics, CameraExtrinsics, InterfaceParams field definitions — what CameraData must compose
- `C:/Users/tucke/PycharmProjects/AquaKit/src/aquakit/camera.py` — OpenCV boundary pattern: `_to_numpy()`, `detach().cpu().numpy()`, `torch.from_numpy().to(device)`
- `C:/Users/tucke/PycharmProjects/AquaKit/src/aquakit/__init__.py` — Current public API; needs to be updated to export CameraData, CalibrationData, load_calibration_data, compute_undistortion_maps, undistort_image
- `C:/Users/tucke/PycharmProjects/AquaKit/pyproject.toml` — Dependencies (torch via hatch env, numpy>=1.24, opencv-python>=4.8); no new deps needed
- `C:/Users/tucke/PycharmProjects/AquaKit/tests/conftest.py` — Device fixture: `cpu` + CUDA skipif; all tests should use this fixture

### Secondary (MEDIUM confidence)

- `C:/Users/tucke/PycharmProjects/AquaKit/.planning/research/aquamvs-map.md` — Pre-mapped AquaMVS architecture; used for navigation
- `C:/Users/tucke/PycharmProjects/AquaKit/.planning/research/aquacal-map.md` — Pre-mapped AquaCal architecture; calibration I/O section
- `C:/Users/tucke/PycharmProjects/AquaKit/.planning/research/shared-patterns.md` — Cross-repo consistency notes; translation table for AquaCal → AquaMVS field types
- `C:/Users/tucke/PycharmProjects/AquaKit/.planning/phases/02-projection-protocol/02-VERIFICATION.md` — Phase 2 verification: confirmed test patterns, device parametrization, torch.testing.assert_close

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies; all libraries already in pyproject.toml; direct source code verification
- Architecture: HIGH — exact dataclass structures verified against AquaMVS and Phase 1 types; JSON schema verified against AquaCal serialization.py
- Pitfalls: HIGH — all identified from actual source code (dist_coeffs reshape from AquaMVS, t shape from AquaMVS comment, AquaCal import from phase success criterion, device pattern from camera.py)
- Test patterns: HIGH — minimal dict input test pattern is the standard approach for standalone calibration tests (no file I/O needed)

**Research date:** 2026-02-18
**Valid until:** 2026-08-18 (stable domain — AquaCal JSON schema is versioned at 1.0; OpenCV API is stable; PyTorch dataclass patterns are stable)
