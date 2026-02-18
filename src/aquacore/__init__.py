"""AquaCore: refractive multi-camera geometry foundation for the Aqua ecosystem."""

from importlib.metadata import PackageNotFoundError, version

from .interface import ray_plane_intersection
from .transforms import (
    camera_center,
    compose_poses,
    invert_pose,
    matrix_to_rvec,
    rvec_to_matrix,
)
from .types import (
    INTERFACE_NORMAL,
    CameraExtrinsics,
    CameraIntrinsics,
    InterfaceParams,
    Mat3,
    Vec2,
    Vec3,
)

__all__ = [
    "INTERFACE_NORMAL",
    "CameraExtrinsics",
    "CameraIntrinsics",
    "InterfaceParams",
    "Mat3",
    "Vec2",
    "Vec3",
    "camera_center",
    "compose_poses",
    "invert_pose",
    "matrix_to_rvec",
    "ray_plane_intersection",
    "rvec_to_matrix",
]

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
