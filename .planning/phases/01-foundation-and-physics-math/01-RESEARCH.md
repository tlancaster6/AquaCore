# Phase 1: Foundation and Physics Math - Research

**Researched:** 2026-02-18
**Domain:** PyTorch geometry, refractive optics, camera models, triangulation
**Confidence:** HIGH — research drawn directly from AquaCal and AquaMVS source code on this machine

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **Guiding Principle: Unify Existing Implementations** — AquaKit v1 extracts and unifies existing behavior from AquaCal and AquaMVS. The default for any gray area is to match existing implementations unless there is a clear reason to diverge (broken, inconsistent, or undocumented behavior).
- **Source repositories (same machine):** AquaCal at `C:\Users\tucke\PycharmProjects\AquaCal`, AquaMVS at `C:\Users\tucke\PycharmProjects\AquaMVS`
- **Camera Model API:** Fisheye distortion model is OpenCV fisheye (k1-k4 equidistant). `create_camera()` factory is the only public construction API; underlying classes are internal.
- **Refraction Model:** Simplified air-to-water (single interface, one refractive index ratio). Not the full 3-layer air-glass-water chain. Flat interface only (no tilted interface support needed).
- **Total internal reflection handling:** Match existing AquaCal/AquaMVS behavior — AquaCal returns None; AquaMVS clamps `sin²(θ) >= 0` (no explicit TIR flag, no NaN). AquaKit must return a validity flag, not NaN.
- **Validation strategy:** Validate at boundaries (factory functions like `create_camera()` validate and raise); internal math functions trust their inputs.
- **Device mismatch:** Raise on mismatch with clear error message — no silent tensor moves between devices.
- **Backend:** PyTorch (follow AquaMVS patterns).
- **Batch shapes:** Always (N, 3) — not single-point (3,) API.
- **Newton-Raphson:** Fixed iterations (AquaMVS approach) — needed for autodiff support.
- **Error handling:** Return `(output, valid_mask)` tuples — matches AquaMVS, works for batches.
- **t shape:** (3,) always.

### Claude's Discretion

- Camera class architecture (separate classes vs single class with model parameter)
- How intrinsics/extrinsics relate to camera objects
- Exception hierarchy (standard Python vs custom)
- All decisions not explicitly locked above

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

---

## Summary

Phase 1 implements the complete geometry foundation for AquaKit by porting existing NumPy implementations from AquaCal to PyTorch and unifying with AquaMVS patterns. The work is a *translation*, not a greenfield design — every function has a known-good reference implementation to port. The main engineering challenge is the NumPy-to-PyTorch translation: replacing `np.linalg.*` with `torch.linalg.*`, replacing vectorized NumPy loops with batched tensor ops, and adding device-agnostic patterns throughout.

The two source codebases are broadly consistent in physics and math conventions (same Snell's law formula, same triangulation algorithm, same coordinate system). Key differences are the compute backend (NumPy vs PyTorch), point shapes (single-point vs batch-first), and error signaling (None vs validity mask). AquaKit adopts all AquaMVS patterns for these decisions since AquaMVS is the PyTorch-native codebase.

The most complex individual component is the pinhole/fisheye camera `create_camera()` factory, because it must bridge PyTorch data storage with OpenCV-based distortion (which requires NumPy). The approach used in AquaMVS — store tensors in Python, convert to NumPy only at the OpenCV boundary — is the established pattern to follow.

**Primary recommendation:** Port each AquaCal function directly, one module at a time, replacing NumPy operations with PyTorch equivalents. Use AquaMVS tests as acceptance criteria. Write ground-truth test values by running the AquaCal NumPy reference implementation and capturing outputs.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | >=2.0 (already in pyproject.toml) | All math — tensors, linalg, autograd | Project requirement; AquaMVS established PyTorch as the compute backend |
| opencv-python | >=4.8 (already in pyproject.toml) | Fisheye distortion, pinhole distortion, Rodrigues | Both AquaCal and AquaMVS use cv2 for distortion; no pure-PyTorch replacement for cv2.fisheye |
| numpy | >=1.24 (already in pyproject.toml) | Serialization boundary; OpenCV interop | cv2 requires numpy arrays; keep at boundary only |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | already in hatch env | Test runner | All tests |
| pytest parametrize | built into pytest | CPU/CUDA device parametrization | QA-01, QA-02 — every test |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| cv2.Rodrigues for rvec/matrix | torch-based Rodrigues | AquaCal uses cv2.Rodrigues; but AquaMVS uses no Rodrigues at all. AquaKit should implement pure-PyTorch Rodrigues to avoid OpenCV dependency in transforms.py since it is pure math |
| cv2.fisheye for distortion | kornia.geometry.camera | kornia is an optional dep in pyproject.toml; prefer cv2 to match existing behavior exactly |

**Note on Rodrigues:** AquaCal uses `cv2.Rodrigues` in `transforms.py`. AquaMVS does not expose transforms at all. For AquaKit, implementing pure-PyTorch Rodrigues (using the Rodrigues formula: `R = I*cos(θ) + sin(θ)*[k]× + (1-cos(θ))*k⊗k`) removes the cv2 dependency from transforms.py and makes it GPU-compatible. This is a rare justified divergence from AquaCal — cv2.Rodrigues is CPU-only and returns NumPy, making it incompatible with device-agnostic tensors.

**Installation:** No new dependencies needed — all are in pyproject.toml already.

---

## Architecture Patterns

### Module-to-File Mapping

All scaffold files already exist (empty stubs). Phase 1 fills these files:

```
src/aquakit/
├── types.py          # TYPE-01..05: dataclasses + type aliases
├── interface.py      # REF-05: InterfaceParams dataclass + ray_plane_intersection
├── camera.py         # CAM-01..06: pinhole/fisheye internals + create_camera()
├── transforms.py     # TRN-01..04: rvec↔matrix, compose, invert, camera_center
├── refraction.py     # REF-01..07: snells_law_3d, trace_ray_*, cast_ray, project
└── triangulation.py  # TRI-01..03: triangulate_rays, point_to_ray_distance
```

The `projection/` subpackage (Phase 2) is NOT part of Phase 1, but `refraction.py` must expose the standalone physics functions that `projection/refractive.py` will build on.

### Pattern 1: Type Aliases as torch.Tensor (not NDArray)

AquaCal defines `Vec3 = NDArray[np.float64]`. AquaKit must redefine as `torch.Tensor` type aliases with documented shapes.

```python
# src/aquakit/types.py
# Source: AquaCal config/schema.py + AquaMVS calibration.py
from typing import TypeAlias
import torch

Vec2: TypeAlias = torch.Tensor  # shape (2,), float32
Vec3: TypeAlias = torch.Tensor  # shape (3,), float32
Mat3: TypeAlias = torch.Tensor  # shape (3, 3), float32
```

Note: These are documentation aliases only — Python's type system cannot enforce tensor shapes at runtime. Document shapes in docstrings.

### Pattern 2: Dataclasses with Tensor Fields

```python
# src/aquakit/types.py
# Pattern from AquaCal config/schema.py, translated to torch.Tensor fields
from dataclasses import dataclass
import torch

@dataclass
class CameraIntrinsics:
    """Intrinsic camera parameters.

    Attributes:
        K: Intrinsic matrix, shape (3, 3), float32.
        dist_coeffs: Distortion coefficients, shape (N,), float64.
            Pinhole: N=5 or N=8. Fisheye: N=4.
        image_size: (width, height) in pixels.
        is_fisheye: True for equidistant fisheye model.
    """
    K: torch.Tensor          # (3, 3), float32
    dist_coeffs: torch.Tensor  # (N,), float64 — OpenCV requires float64
    image_size: tuple[int, int]  # (width, height)
    is_fisheye: bool = False

@dataclass
class CameraExtrinsics:
    """Extrinsic camera parameters.

    Attributes:
        R: Rotation matrix (world to camera), shape (3, 3), float32.
        t: Translation vector (world to camera), shape (3,), float32.
    """
    R: torch.Tensor  # (3, 3), float32
    t: torch.Tensor  # (3,), float32

    @property
    def C(self) -> torch.Tensor:
        """Camera center in world coordinates: -R.T @ t, shape (3,)."""
        return -self.R.T @ self.t

@dataclass
class InterfaceParams:
    """Refractive interface parameters.

    Attributes:
        normal: Unit normal from water toward air, shape (3,). Typically [0,0,-1].
        water_z: Z-coordinate of water surface in world frame (meters).
        n_air: Refractive index of air (default 1.0).
        n_water: Refractive index of water (default 1.333).
    """
    normal: torch.Tensor  # (3,), float32
    water_z: float
    n_air: float = 1.0
    n_water: float = 1.333
```

**Design decision for InterfaceParams:** AquaCal's `Interface` stores `camera_distances: dict[str, float]` (per-camera water_z). AquaMVS's `CalibrationData` stores a single `water_z: float`. Phase 1 uses a single `water_z: float` matching AquaMVS — after optimization all cameras share the same water_z, and the dict complexity is only needed during calibration (Phase 3).

### Pattern 3: create_camera() Factory with Internal Classes

```python
# src/aquakit/camera.py
# Pattern from AquaCal core/camera.py, adapted to PyTorch + create_camera() API

class _PinholeCamera:
    """Internal pinhole camera. Not part of public API."""
    def __init__(self, intrinsics: CameraIntrinsics, extrinsics: CameraExtrinsics): ...
    def project(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
    def pixel_to_ray(self, pixels: torch.Tensor) -> torch.Tensor: ...

class _FisheyeCamera:
    """Internal fisheye camera. Not part of public API."""
    # same interface, different distortion via cv2.fisheye
    ...

def create_camera(
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
) -> _PinholeCamera | _FisheyeCamera:
    """Create Camera from intrinsics. Dispatches on intrinsics.is_fisheye."""
    if intrinsics.is_fisheye:
        return _FisheyeCamera(intrinsics, extrinsics)
    return _PinholeCamera(intrinsics, extrinsics)
```

**Note:** AquaCal's `Camera` takes a `name` parameter. AquaKit does not need this since the `name` is used only for the AquaCal `camera_distances` dict lookup (which AquaKit eliminates by using a single `water_z`). The camera object itself is anonymous.

### Pattern 4: Pinhole Projection (PyTorch, Batch-First)

The camera projection needs to handle distortion via OpenCV. The key insight from AquaMVS is that `RefractiveProjectionModel.project()` projects the **interface point P** (which is on the flat water surface, undistorted geometry) rather than the underwater 3D point directly. The distortion for back-projection happens via `K_inv` on pixels.

For `_PinholeCamera.project()`:
```python
def project(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # points: (N, 3) float32, world frame
    # Transform to camera frame
    p_cam = (self.R @ points.T).T + self.t  # (N, 3)
    valid = p_cam[:, 2] > 0  # (N,)

    # Apply distortion via OpenCV (CPU-only boundary)
    # Convert to numpy, call cv2.projectPoints, convert back
    K_np = self.K.cpu().numpy().astype(np.float64)
    dist_np = self.dist_coeffs.cpu().numpy()
    p_cam_np = p_cam.detach().cpu().numpy().astype(np.float64)
    # ... cv2.projectPoints ...
    pixels = torch.from_numpy(pixels_np).to(points.device, dtype=points.dtype)
    return pixels, valid
```

**Pitfall:** This CPU round-trip breaks autograd. For Phase 1 (basic camera), this is acceptable. Phase 2 (RefractiveProjectionModel) uses `K_inv` back-projection which is fully differentiable. The round-trip in camera.project() only matters for forward projection through distortion, which Phase 2 avoids by projecting the undistorted interface point.

### Pattern 5: Snell's Law in PyTorch (Batch)

Direct port of AquaCal `snells_law_3d` to PyTorch batch form:

```python
# src/aquakit/refraction.py
# Source: AquaCal core/refractive_geometry.py snells_law_3d
# + AquaMVS projection/refractive.py cast_ray (inline Snell's law)

def snells_law_3d(
    incident_directions: torch.Tensor,  # (N, 3) unit vectors
    surface_normal: torch.Tensor,       # (3,) unit normal
    n_ratio: float,                     # n1 / n2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Snell's law in 3D.

    Returns:
        directions: (N, 3) refracted unit vectors (zeros for TIR cases)
        valid: (N,) bool, False where total internal reflection occurs
    """
    # cos_i = dot(d, normal) -- handle orientation internally
    cos_i = (incident_directions * surface_normal).sum(dim=-1)  # (N,)

    # Orient normal into destination medium
    # If cos_i < 0, ray travels against normal (normal needs to flip)
    flip = cos_i < 0
    n_oriented = torch.where(flip.unsqueeze(-1), -surface_normal, surface_normal)
    cos_i = cos_i.abs()  # always positive after orientation

    sin_t_sq = n_ratio ** 2 * (1.0 - cos_i ** 2)  # (N,)
    valid = sin_t_sq <= 1.0  # False = TIR

    cos_t = torch.sqrt(torch.clamp(1.0 - sin_t_sq, min=0.0))  # (N,)
    directions = (
        n_ratio * incident_directions
        + (cos_t - n_ratio * cos_i).unsqueeze(-1) * n_oriented
    )  # (N, 3)

    # Normalize
    norms = torch.linalg.norm(directions, dim=-1, keepdim=True).clamp(min=1e-12)
    directions = directions / norms

    # Zero out TIR directions (caller uses valid mask)
    directions = torch.where(valid.unsqueeze(-1), directions, torch.zeros_like(directions))

    return directions, valid
```

**Key difference from AquaMVS:** AquaMVS's inline Snell's law in `cast_ray()` uses `torch.clamp(1.0 - sin_t_sq, min=0.0)` without an explicit TIR flag (it silently continues). AquaKit needs the explicit `valid` flag per the requirements (REF-02). The formula is identical; we add the flag on top.

### Pattern 6: Newton-Raphson Refractive Projection (Fixed Iterations)

```python
# src/aquakit/refraction.py
# Source: AquaMVS projection/refractive.py project() method
# Source: AquaCal refractive_geometry.py _refractive_project_newton

def refractive_project(
    points: torch.Tensor,          # (N, 3) underwater points, world frame
    camera_center: torch.Tensor,   # (3,) camera center in world frame
    water_z: float,
    n_air: float,
    n_water: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Returns (interface_points, valid)
    # interface_points: (N, 3) on water surface (Z = water_z)
    # valid: (N,) bool
    ...
```

Note: This function returns the **interface point** (not the pixel). The caller uses the camera model to project the interface point to a pixel. This separation matches AquaMVS's two-step approach: find P on surface, then project P via `K @ (R @ P + t)`.

### Pattern 7: Triangulation (PyTorch Port)

Direct port of AquaMVS `triangulate_rays`:

```python
# src/aquakit/triangulation.py
# Source: AquaMVS triangulation.py triangulate_rays (identical algorithm to AquaCal)

def triangulate_rays(rays: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    # Same closed-form: A = sum(I - d @ d.T), solve A @ P = b
    # Infers device from first ray
    ...

def point_to_ray_distance(
    point: torch.Tensor,       # (3,)
    ray_origin: torch.Tensor,  # (3,)
    ray_direction: torch.Tensor,  # (3,) unit
) -> torch.Tensor:
    # Returns scalar distance
    v = point - ray_origin
    proj = (v * ray_direction).sum() * ray_direction
    return torch.linalg.norm(v - proj)
```

### Pattern 8: Rodrigues in Pure PyTorch

AquaCal uses `cv2.Rodrigues` (CPU + NumPy). AquaKit should implement pure-PyTorch Rodrigues to enable GPU usage and autograd:

```python
# src/aquakit/transforms.py

def rvec_to_matrix(rvec: torch.Tensor) -> torch.Tensor:
    """Convert Rodrigues vector to rotation matrix.

    Args:
        rvec: Rotation vector, shape (3,). angle = ||rvec||, axis = rvec/||rvec||.

    Returns:
        R: Rotation matrix, shape (3, 3).
    """
    angle = torch.linalg.norm(rvec)
    if angle < 1e-12:
        return torch.eye(3, dtype=rvec.dtype, device=rvec.device)
    k = rvec / angle  # unit axis
    K = torch.zeros(3, 3, dtype=rvec.dtype, device=rvec.device)
    K[0, 1] = -k[2]; K[0, 2] = k[1]
    K[1, 0] = k[2];  K[1, 2] = -k[0]
    K[2, 0] = -k[1]; K[2, 1] = k[0]
    # Rodrigues formula: R = I*cos(θ) + sin(θ)*K + (1-cos(θ))*k⊗k
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    R = (cos_a * torch.eye(3, dtype=rvec.dtype, device=rvec.device)
         + sin_a * K
         + (1 - cos_a) * torch.outer(k, k))
    return R
```

For `matrix_to_rvec`, use `torch.linalg.matrix_exp` inverse path or the closed-form formula:
- `θ = arccos((trace(R) - 1) / 2)`
- `k = (R - R.T) / (2 * sin(θ))`
- `rvec = θ * k_vec` where `k_vec = [K[2,1], K[0,2], K[1,0]]`
- Handle degenerate cases (θ ≈ 0, θ ≈ π) carefully.

**Alternative:** Call `cv2.Rodrigues` in `rvec_to_matrix` / `matrix_to_rvec` with a numpy round-trip at the call site. This is simpler but CPU-only and breaks grad. Since `transforms.py` is used at serialization/loading boundaries (not in the hot path), the cv2 approach is acceptable for Phase 1. Recommend pure-PyTorch to be consistent with the device-agnostic principle.

### Pattern 9: Test Device Parametrization

Follow AquaMVS `tests/test_triangulation.py` device fixture:

```python
# tests/unit/conftest.py (or per-file)

import pytest
import torch

@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
))])
def device(request):
    return torch.device(request.param)
```

Use `torch.testing.assert_close(result, expected, atol=1e-5, rtol=0)` for geometry comparisons (not `assert_allclose`).

### Anti-Patterns to Avoid

- **Hardcoding `.cuda()` in tests:** Always parametrize over device fixture, never call `.cuda()` directly.
- **Importing AquaCal/AquaMVS in tests:** Tests must be self-contained. Compute known ground-truth values analytically or from first principles, not by calling the reference implementations.
- **Using `torch.float64` throughout:** Follow AquaMVS — float32 for geometry tensors, float64 only for `dist_coeffs` (OpenCV requirement).
- **Returning None for TIR:** Return `(directions, valid_mask)` — None is the AquaCal pattern, not the AquaKit pattern.
- **Silent device moves:** Never call `.to(device)` inside math functions without explicit parameter.
- **In-place ops on CUDA in Newton-Raphson:** AquaMVS uses `r_p = torch.clamp(r_p, min=0.0)` + `r_p = torch.minimum(r_p, r_q)` (non-in-place) for autograd compatibility. Do not use `r_p.clamp_()`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Fisheye distortion | Custom equidistant model | `cv2.fisheye.projectPoints`, `cv2.fisheye.undistortPoints` | Both AquaCal and AquaMVS use this; 4-coefficient equidistant model is exactly the OpenCV fisheye API |
| Pinhole distortion | Custom polynomial | `cv2.projectPoints`, `cv2.undistortPoints` | AquaCal uses this; handles k1/k2/p1/p2/k3 and rational model (k4-k6) correctly |
| Batched linear solve | Loop over rays | `torch.linalg.solve` with (M, 3, 3) input | AquaMVS uses batched solve for _triangulate_two_rays_batch; torch.linalg.solve supports batch dims |

**Key insight:** The physics math (Snell's law, Newton-Raphson for refractive projection) must be hand-rolled because no library implements refractive geometry. Everything else has library support.

---

## Common Pitfalls

### Pitfall 1: dist_coeffs dtype must be float64

**What goes wrong:** Passing float32 `dist_coeffs` to `cv2.fisheye.projectPoints` raises an OpenCV error or silently produces wrong results.
**Why it happens:** OpenCV's Python bindings require float64 for distortion coefficient arrays.
**How to avoid:** Store `dist_coeffs` as `torch.float64` in `CameraIntrinsics`. When calling cv2, always convert: `dist_coeffs.cpu().numpy().astype(np.float64)`.
**Warning signs:** OpenCV error messages containing "CV_64F" or "wrong type".

**Evidence:** Both AquaCal (NumPy default is float64) and AquaMVS (`torch.float64` explicitly) keep dist_coeffs as float64.

### Pitfall 2: Newton-Raphson needs epsilon to avoid div-by-zero gradient

**What goes wrong:** `f_prime = n_air * h_c**2 / (d_air_sq * d_air)` — if `d_air` approaches zero (point directly below camera), gradient is undefined.
**Why it happens:** `r_p = 0` initially possible for on-axis points.
**How to avoid:** Add epsilon: `f / (f_prime + 1e-12)`. AquaMVS uses `1e-12`. Also use `r_q = sqrt(dx*dx + dy*dy + 1e-12)` to avoid zero gradient at on-axis singularity.
**Warning signs:** NaN gradients in tests; `loss.backward()` producing `nan` tensors.

### Pitfall 3: Normal orientation in Snell's law

**What goes wrong:** The interface normal `[0, 0, -1]` points from water toward air. For an air-to-water ray (going +Z), `dot(ray, normal) < 0` — the normal points opposite to the ray. The implementation must flip the normal to point into the destination medium.
**Why it happens:** Snell's law formula requires the normal to point from incident medium toward transmission medium.
**How to avoid:** Implement the `cos_i < 0` check from AquaCal and flip n accordingly. AquaMVS uses `cos_i = -(rays_world * self.normal).sum(dim=-1)` with hardcoded negation because it always does air-to-water. AquaKit's standalone `snells_law_3d` handles both directions.
**Warning signs:** Refracted rays pointing in wrong direction; TIR reported where none should occur.

### Pitfall 4: torch.linalg.solve fails on degenerate triangulation systems

**What goes wrong:** When all rays are parallel, `A_sum = sum(I - d @ d.T) = 0` is singular. `torch.linalg.solve` raises `LinAlgError`.
**Why it happens:** Parallel rays have no unique intersection.
**How to avoid:** Catch `torch.linalg.LinAlgError` and re-raise as `ValueError("Degenerate ray configuration")`. AquaMVS does this already.
**Warning signs:** Tests fail with uncaught `LinAlgError`.

### Pitfall 5: Rodrigues singularity at θ = π

**What goes wrong:** At exactly 180° rotation, the axis extraction formula `k = (R - R.T) / (2*sin(θ))` has `sin(π) = 0` in denominator, producing NaN.
**Why it happens:** The standard Rodrigues formula is numerically unstable at θ ≈ 0 and θ ≈ π.
**How to avoid:** Add epsilon guards. For θ < 1e-10: return identity. For θ > π - 1e-6: use the special-case formula `k_i = sqrt((R[i,i] + 1) / 2)`.
**Warning signs:** `test_180_degree_rotation` producing NaN (the AquaCal test checks only norm, not values, which is a hint about this instability).

### Pitfall 6: Camera project() autograd breakage

**What goes wrong:** `camera.project()` calls `cv2.projectPoints` via a numpy round-trip, which breaks the PyTorch autograd graph.
**Why it happens:** NumPy operations are not tracked by autograd.
**How to avoid:** Document that `_PinholeCamera.project()` is not differentiable. Phase 2 `RefractiveProjectionModel` avoids this by projecting the undistorted interface point (no distortion application needed). The camera distortion is handled at the back-projection boundary (`K_inv @ pixel`).
**Warning signs:** `loss.backward()` raises "element 0 of tensors does not require grad" or produces zero gradients.

---

## Code Examples

Verified patterns from source code:

### Snell's Law (AquaCal reference)

```python
# Source: AquaCal/src/aquacal/core/refractive_geometry.py
def snells_law_3d(incident_direction, surface_normal, n_ratio):
    d = incident_direction / np.linalg.norm(incident_direction)
    cos_i = np.dot(d, surface_normal)
    if cos_i < 0:
        n = -surface_normal
        cos_i = -cos_i
    else:
        n = surface_normal
    sin_t_sq = n_ratio**2 * (1 - cos_i**2)
    if sin_t_sq > 1.0:
        return None  # TIR
    cos_t = np.sqrt(1 - sin_t_sq)
    t = n_ratio * d + (cos_t - n_ratio * cos_i) * n
    return t / np.linalg.norm(t)
```

### Snell's Law (AquaMVS inline, no TIR flag)

```python
# Source: AquaMVS/src/aquamvs/projection/refractive.py cast_ray()
cos_i = -(rays_world * self.normal).sum(dim=-1)  # (N,)
n_oriented = -self.normal.unsqueeze(0)           # always air-to-water
sin_t_sq = self.n_ratio**2 * (1.0 - cos_i**2)
cos_t = torch.sqrt(torch.clamp(1.0 - sin_t_sq, min=0.0))  # clamp hides TIR
directions = (self.n_ratio * rays_world
              + (cos_t - self.n_ratio * cos_i).unsqueeze(-1) * n_oriented)
directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
```

### Newton-Raphson (AquaMVS reference, fixed 10 iterations)

```python
# Source: AquaMVS/src/aquamvs/projection/refractive.py project()
r_p = r_q * h_c / (h_c + h_q + 1e-12)  # initial guess
for _ in range(10):
    d_air_sq = r_p * r_p + h_c * h_c
    d_air = torch.sqrt(d_air_sq)
    r_diff = r_q - r_p
    d_water_sq = r_diff * r_diff + h_q * h_q
    d_water = torch.sqrt(d_water_sq)
    sin_air = r_p / d_air
    sin_water = r_diff / d_water
    f = self.n_air * sin_air - self.n_water * sin_water
    f_prime = (self.n_air * h_c * h_c / (d_air_sq * d_air)
               + self.n_water * h_q * h_q / (d_water_sq * d_water))
    r_p = r_p - f / (f_prime + 1e-12)
    r_p = torch.clamp(r_p, min=0.0)
    r_p = torch.minimum(r_p, r_q)
```

### Triangulation (AquaMVS reference)

```python
# Source: AquaMVS/src/aquamvs/triangulation.py triangulate_rays()
A_sum = torch.zeros(3, 3, device=device, dtype=dtype)
b_sum = torch.zeros(3, device=device, dtype=dtype)
for origin, direction in rays:
    d = direction / torch.linalg.norm(direction)
    I_minus_ddT = torch.eye(3, device=device, dtype=dtype) - torch.outer(d, d)
    A_sum += I_minus_ddT
    b_sum += I_minus_ddT @ origin
try:
    P = torch.linalg.solve(A_sum, b_sum)
except torch.linalg.LinAlgError as err:
    raise ValueError("Degenerate ray configuration") from err
```

### Device Parametrized Test Fixture (AquaMVS pattern)

```python
# Source: AquaMVS/tests/test_triangulation.py
@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)
```

### Known-Value Test for Snell's Law

```python
# From AquaCal tests/unit/test_refractive_geometry.py (adapted)
def test_normal_incidence(device):
    incident = torch.tensor([[0.0, 0.0, 1.0]], device=device)
    normal = torch.tensor([0.0, 0.0, -1.0], device=device)
    dirs, valid = snells_law_3d(incident, normal, n_ratio=1.0/1.333)
    assert valid[0]
    # At normal incidence, refraction does not change direction
    torch.testing.assert_close(dirs[0], torch.tensor([0.0, 0.0, 1.0], device=device), atol=1e-6, rtol=0)

def test_known_angle(device):
    # n_air * sin(30°) = n_water * sin(theta_water)
    # sin(theta_water) = sin(30°) / 1.333 = 0.5 / 1.333 ≈ 0.3751
    # theta_water ≈ 22.02°
    import math
    theta_air = math.radians(30.0)
    incident = torch.tensor([[math.sin(theta_air), 0.0, math.cos(theta_air)]], device=device)
    normal = torch.tensor([0.0, 0.0, -1.0], device=device)
    dirs, valid = snells_law_3d(incident, normal, n_ratio=1.0/1.333)
    assert valid[0]
    sin_theta_water = dirs[0, 0].item()
    expected = math.sin(theta_air) / 1.333
    assert abs(sin_theta_water - expected) < 1e-5
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| NumPy for geometry (AquaCal) | PyTorch for geometry (AquaMVS, AquaKit) | AquaMVS from start | GPU support, autograd |
| Single-point API (AquaCal) | Batch-first (N, 3) API (AquaMVS, AquaKit) | AquaMVS from start | Vectorized operations |
| None for TIR (AquaCal) | (output, valid_mask) for all failures (AquaMVS, AquaKit) | AquaMVS from start | Batch-compatible error signaling |
| Convergence-based N-R (AquaCal) | Fixed 10 iterations N-R (AquaMVS, AquaKit) | AquaMVS from start | Deterministic autograd |

---

## Open Questions

1. **Camera.project() differentiability**
   - What we know: `cv2.projectPoints` breaks autograd; Phase 2 avoids this by projecting undistorted interface points.
   - What's unclear: Does anything in Phase 1 actually need differentiable `camera.project()`?
   - Recommendation: Document that `camera.project()` is not differentiable in Phase 1. Phase 2 RefractiveProjectionModel is differentiable end-to-end. This is acceptable — calibration (which uses camera.project()) does not need gradients.

2. **Rodrigues implementation choice**
   - What we know: AquaCal uses cv2.Rodrigues (CPU-only, NumPy). Pure-PyTorch Rodrigues requires handling edge cases at θ=0 and θ=π.
   - What's unclear: Will transforms.py be on the hot path or only at load-time?
   - Recommendation: Implement pure-PyTorch Rodrigues to be consistent with device-agnostic principle. Add edge-case guards. If edge cases cause issues, fall back to cv2.Rodrigues via numpy round-trip at load time only.

3. **TRI-03: Refractive triangulation**
   - What we know: The requirement says "handles refractive rays (rays with kink at interface)." Both AquaCal `triangulate_point()` and AquaMVS `triangulate_pair()` work by first casting refractive rays (origin on water surface, direction in water), then triangulating. The triangulation algorithm itself is the same — it doesn't need to know the rays are refractive.
   - What's unclear: Does TRI-03 require a new function, or just documentation that `triangulate_rays()` accepts refractive rays?
   - Recommendation: No new function needed. `triangulate_rays()` works with any rays. TRI-03 is satisfied by testing that refractive rays from `refraction.py` + `triangulate_rays()` together produce correct results (end-to-end test).

---

## Sources

### Primary (HIGH confidence — direct source code inspection)

- `C:/Users/tucke/PycharmProjects/AquaCal/src/aquacal/config/schema.py` — CameraIntrinsics, CameraExtrinsics, InterfaceParams, Vec2, Vec3, Mat3, exception hierarchy
- `C:/Users/tucke/PycharmProjects/AquaCal/src/aquacal/core/camera.py` — Camera, FisheyeCamera, create_camera, undistort_points
- `C:/Users/tucke/PycharmProjects/AquaCal/src/aquacal/core/refractive_geometry.py` — snells_law_3d, Newton-Raphson, ray tracing, complete implementation with docstrings
- `C:/Users/tucke/PycharmProjects/AquaCal/src/aquacal/core/interface_model.py` — Interface, ray_plane_intersection
- `C:/Users/tucke/PycharmProjects/AquaCal/src/aquacal/triangulation/triangulate.py` — triangulate_rays, point_to_ray_distance, triangulate_point
- `C:/Users/tucke/PycharmProjects/AquaCal/src/aquacal/utils/transforms.py` — rvec_to_matrix, matrix_to_rvec, compose_poses, invert_pose, camera_center
- `C:/Users/tucke/PycharmProjects/AquaMVS/src/aquamvs/projection/refractive.py` — RefractiveProjectionModel (PyTorch patterns)
- `C:/Users/tucke/PycharmProjects/AquaMVS/src/aquamvs/projection/protocol.py` — ProjectionModel protocol
- `C:/Users/tucke/PycharmProjects/AquaMVS/src/aquamvs/triangulation.py` — triangulate_rays (PyTorch), _triangulate_two_rays_batch
- `C:/Users/tucke/PycharmProjects/AquaMVS/src/aquamvs/calibration.py` — CameraData, CalibrationData, load_calibration_data
- `C:/Users/tucke/PycharmProjects/AquaMVS/tests/test_projection/test_refractive.py` — test patterns, device fixture, known-value tests
- `C:/Users/tucke/PycharmProjects/AquaMVS/tests/test_triangulation.py` — device parametrization pattern
- `C:/Users/tucke/PycharmProjects/AquaCal/tests/unit/test_refractive_geometry.py` — Snell's law known-value tests
- `C:/Users/tucke/PycharmProjects/AquaCal/tests/unit/test_transforms.py` — transform known-value tests

### Secondary (MEDIUM confidence)

- `.planning/research/aquacal-map.md`, `.planning/research/aquamvs-map.md`, `.planning/research/shared-patterns.md` — Pre-mapped cross-repo comparison; used as navigation guide before raw source inspection

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — pyproject.toml already declares all dependencies; both repos confirm usage
- Architecture: HIGH — implementations copied directly from source; patterns verified by inspection
- Pitfalls: HIGH — edge cases identified in actual source code (epsilon guards, dtype requirements, etc.)
- Test patterns: HIGH — AquaMVS test suite provides exact pytest patterns to follow

**Research date:** 2026-02-18
**Valid until:** 2026-08-18 (stable geometry domain — no fast-moving dependencies)
