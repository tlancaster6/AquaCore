"""Shared geometry types for all AquaCore modules.

Coordinate system: world origin at reference camera optical center.
+X right, +Y forward, +Z down into water.
Camera frame: OpenCV convention. Extrinsics: p_cam = R @ p_world + t.
Interface normal: [0, 0, -1] (upward from water surface).
"""

from dataclasses import dataclass
from typing import TypeAlias

import torch

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Vec2: TypeAlias = torch.Tensor
"""Shape (2,) or (N, 2), float32. 2D vector or batch of 2D vectors."""

Vec3: TypeAlias = torch.Tensor
"""Shape (3,) or (N, 3), float32. 3D vector or batch of 3D vectors."""

Mat3: TypeAlias = torch.Tensor
"""Shape (3, 3), float32. 3x3 matrix."""

# ---------------------------------------------------------------------------
# Coordinate system constants
# ---------------------------------------------------------------------------

INTERFACE_NORMAL: torch.Tensor = torch.tensor([0.0, 0.0, -1.0])
"""Unit normal of the flat air-water interface, pointing upward from water (shape (3,)).

In the world coordinate system (+Z down into water), this vector points from
water toward air.
"""

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CameraIntrinsics:
    """Intrinsic camera parameters.

    Attributes:
        K: Intrinsic matrix, shape (3, 3), float32.
        dist_coeffs: Distortion coefficients, shape (N,), float64.
            Pinhole: N=5 or N=8 (k1, k2, p1, p2, k3, ...).
            Fisheye: N=4 (k1, k2, k3, k4 equidistant model).
            Must be float64 — OpenCV requires float64 for distortion arrays.
        image_size: (width, height) in pixels.
        is_fisheye: True for OpenCV equidistant fisheye model (k1-k4).
            False for standard pinhole distortion model.
    """

    K: torch.Tensor  # (3, 3), float32
    dist_coeffs: torch.Tensor  # (N,), float64 — OpenCV requires float64
    image_size: tuple[int, int]  # (width, height)
    is_fisheye: bool = False


@dataclass
class CameraExtrinsics:
    """Extrinsic camera parameters (world-to-camera transform).

    Attributes:
        R: Rotation matrix (world to camera), shape (3, 3), float32.
        t: Translation vector (world to camera), shape (3,), float32.
            Transform: p_cam = R @ p_world + t.
    """

    R: torch.Tensor  # (3, 3), float32
    t: torch.Tensor  # (3,), float32

    @property
    def C(self) -> torch.Tensor:
        """Camera center in world coordinates.

        Returns:
            Camera center as -R.T @ t, shape (3,), float32.
        """
        return -self.R.T @ self.t


@dataclass
class InterfaceParams:
    """Refractive interface parameters for a flat air-water surface.

    Attributes:
        normal: Unit normal from water toward air, shape (3,), float32.
            Typically INTERFACE_NORMAL = [0, 0, -1].
        water_z: Z-coordinate of water surface in world frame (meters).
            In the +Z-down convention, water_z is positive (surface is below
            the camera).
        n_air: Refractive index of air (default 1.0).
        n_water: Refractive index of water (default 1.333).
    """

    normal: torch.Tensor  # (3,), float32
    water_z: float
    n_air: float = 1.0
    n_water: float = 1.333
