"""Camera models (pinhole and fisheye) with project/back-project operations.

Both models use OpenCV at the NumPy boundary for distortion. All project()
and pixel_to_ray() calls are NOT differentiable due to the OpenCV CPU round-trip.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .types import CameraExtrinsics, CameraIntrinsics

# ---------------------------------------------------------------------------
# Internal camera model implementations
# ---------------------------------------------------------------------------


class _PinholeCamera:
    """Pinhole camera model with OpenCV radial/tangential distortion.

    project() and pixel_to_ray() are NOT differentiable due to OpenCV CPU
    round-trip. All tensor inputs are moved to CPU for OpenCV calls and
    returned on the original device.

    Args:
        intrinsics: Camera intrinsic parameters (K, dist_coeffs, image_size).
        extrinsics: Camera extrinsic parameters (R, t world-to-camera).
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
    ) -> None:
        self._device = intrinsics.K.device
        self.K = intrinsics.K  # (3, 3) float32
        self.dist_coeffs = intrinsics.dist_coeffs  # (N,) float64
        self.image_size = intrinsics.image_size
        self.R = extrinsics.R  # (3, 3) float32
        self.t = extrinsics.t  # (3,) float32

    def project(
        self,
        points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project 3D world-frame points to 2D pixel coordinates.

        Args:
            points: World-frame 3D points, shape (N, 3), float32.

        Returns:
            Tuple of:
                pixels: Distorted pixel coordinates, shape (N, 2), float32.
                valid: Boolean mask, shape (N,). True where z_cam > 0.
        """
        # Transform to camera frame: p_cam = (R @ points.T).T + t
        p_cam = (self.R @ points.T).T + self.t  # (N, 3)

        # Valid mask: points in front of camera
        valid = p_cam[:, 2] > 0  # (N,)

        # OpenCV CPU boundary — convert to float64 numpy
        p_cam_np = p_cam.detach().cpu().numpy().astype(np.float64)  # (N, 3)
        K_np = self.K.detach().cpu().numpy().astype(np.float64)  # (3, 3)
        dist_np = self.dist_coeffs.detach().cpu().numpy().astype(np.float64)  # (N,)

        pixels_np, _ = cv2.projectPoints(
            p_cam_np,
            rvec=np.zeros(3, dtype=np.float64),
            tvec=np.zeros(3, dtype=np.float64),
            cameraMatrix=K_np,
            distCoeffs=dist_np,
        )
        # cv2.projectPoints returns shape (N, 1, 2) — squeeze and convert
        pixels_np = pixels_np.squeeze(1).astype(np.float32)  # (N, 2)
        pixels = torch.from_numpy(pixels_np).to(self._device)

        return pixels, valid

    def pixel_to_ray(self, pixels: torch.Tensor) -> torch.Tensor:
        """Back-project 2D pixel coordinates to 3D world-frame unit rays.

        Args:
            pixels: Pixel coordinates, shape (N, 2), float32.

        Returns:
            Unit direction vectors in world frame, shape (N, 3), float32.
        """
        # OpenCV CPU boundary — undistort pixels to normalized camera coords
        pixels_np = pixels.detach().cpu().numpy().astype(np.float64)  # (N, 2)
        K_np = self.K.detach().cpu().numpy().astype(np.float64)  # (3, 3)
        dist_np = self.dist_coeffs.detach().cpu().numpy().astype(np.float64)  # (N,)

        # Reshape to (N, 1, 2) as required by cv2.undistortPoints
        norm_pts_np = cv2.undistortPoints(
            pixels_np.reshape(-1, 1, 2),
            cameraMatrix=K_np,
            distCoeffs=dist_np,
        )  # (N, 1, 2)
        norm_pts_np = norm_pts_np.squeeze(1).astype(np.float32)  # (N, 2)

        # Construct 3D ray in camera frame: [x, y, 1], then normalize
        ones = np.ones((norm_pts_np.shape[0], 1), dtype=np.float32)
        rays_cam_np = np.concatenate([norm_pts_np, ones], axis=1)  # (N, 3)
        rays_cam = torch.from_numpy(rays_cam_np).to(self._device)

        # Normalize to unit vectors
        rays_cam = rays_cam / rays_cam.norm(dim=1, keepdim=True)

        # Rotate to world frame: rays_world = R.T @ rays_cam
        rays_world = (self.R.T @ rays_cam.T).T  # (N, 3)

        return rays_world


class _FisheyeCamera:
    """Fisheye camera model with OpenCV equidistant (k1-k4) distortion.

    project() and pixel_to_ray() are NOT differentiable due to OpenCV CPU
    round-trip. All tensor inputs are moved to CPU for OpenCV calls and
    returned on the original device.

    Args:
        intrinsics: Camera intrinsic parameters (K, dist_coeffs, image_size).
        extrinsics: Camera extrinsic parameters (R, t world-to-camera).
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
    ) -> None:
        self._device = intrinsics.K.device
        self.K = intrinsics.K  # (3, 3) float32
        self.dist_coeffs = intrinsics.dist_coeffs  # (4,) float64 for fisheye k1-k4
        self.image_size = intrinsics.image_size
        self.R = extrinsics.R  # (3, 3) float32
        self.t = extrinsics.t  # (3,) float32

    def project(
        self,
        points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project 3D world-frame points to 2D pixel coordinates.

        Args:
            points: World-frame 3D points, shape (N, 3), float32.

        Returns:
            Tuple of:
                pixels: Distorted pixel coordinates, shape (N, 2), float32.
                valid: Boolean mask, shape (N,). True where z_cam > 0.
        """
        # Transform to camera frame: p_cam = (R @ points.T).T + t
        p_cam = (self.R @ points.T).T + self.t  # (N, 3)

        # Valid mask: points in front of camera
        valid = p_cam[:, 2] > 0  # (N,)

        # OpenCV CPU boundary — convert to float64 numpy
        p_cam_np = p_cam.detach().cpu().numpy().astype(np.float64)  # (N, 3)
        K_np = self.K.detach().cpu().numpy().astype(np.float64)  # (3, 3)
        dist_np = self.dist_coeffs.detach().cpu().numpy().astype(np.float64)  # (4,)

        pixels_np, _ = cv2.fisheye.projectPoints(
            p_cam_np.reshape(-1, 1, 3),  # cv2.fisheye expects (N, 1, 3)
            rvec=np.zeros(3, dtype=np.float64),
            tvec=np.zeros(3, dtype=np.float64),
            K=K_np,
            D=dist_np,
        )
        # cv2.fisheye.projectPoints returns shape (N, 1, 2) — squeeze and convert
        pixels_np = pixels_np.squeeze(1).astype(np.float32)  # (N, 2)
        pixels = torch.from_numpy(pixels_np).to(self._device)

        return pixels, valid

    def pixel_to_ray(self, pixels: torch.Tensor) -> torch.Tensor:
        """Back-project 2D pixel coordinates to 3D world-frame unit rays.

        Args:
            pixels: Pixel coordinates, shape (N, 2), float32.

        Returns:
            Unit direction vectors in world frame, shape (N, 3), float32.
        """
        # OpenCV CPU boundary — undistort pixels to normalized camera coords
        pixels_np = pixels.detach().cpu().numpy().astype(np.float64)  # (N, 2)
        K_np = self.K.detach().cpu().numpy().astype(np.float64)  # (3, 3)
        dist_np = self.dist_coeffs.detach().cpu().numpy().astype(np.float64)  # (4,)

        # cv2.fisheye.undistortPoints expects (N, 1, 2)
        norm_pts_np = cv2.fisheye.undistortPoints(
            pixels_np.reshape(-1, 1, 2),
            K=K_np,
            D=dist_np,
        )  # (N, 1, 2)
        norm_pts_np = norm_pts_np.squeeze(1).astype(np.float32)  # (N, 2)

        # Construct 3D ray in camera frame: [x, y, 1], then normalize
        ones = np.ones((norm_pts_np.shape[0], 1), dtype=np.float32)
        rays_cam_np = np.concatenate([norm_pts_np, ones], axis=1)  # (N, 3)
        rays_cam = torch.from_numpy(rays_cam_np).to(self._device)

        # Normalize to unit vectors
        rays_cam = rays_cam / rays_cam.norm(dim=1, keepdim=True)

        # Rotate to world frame: rays_world = R.T @ rays_cam
        rays_world = (self.R.T @ rays_cam.T).T  # (N, 3)

        return rays_world


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_camera(
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
) -> _PinholeCamera | _FisheyeCamera:
    """Create a camera model from intrinsic and extrinsic parameters.

    This is the only public construction API for camera models. Validates
    all tensor shapes and device consistency before constructing the model.

    Args:
        intrinsics: Camera intrinsic parameters including K, dist_coeffs,
            image_size, and is_fisheye flag.
        extrinsics: Camera extrinsic parameters (R, t world-to-camera).

    Returns:
        _PinholeCamera if intrinsics.is_fisheye is False.
        _FisheyeCamera if intrinsics.is_fisheye is True.

    Raises:
        ValueError: If tensor devices do not all match, or if K/R/t/dist_coeffs
            have incorrect shapes.
    """
    # --- Device consistency check ---
    k_device = intrinsics.K.device
    r_device = extrinsics.R.device
    t_device = extrinsics.t.device

    if k_device != r_device or k_device != t_device:
        raise ValueError(
            f"All camera tensors must be on the same device. "
            f"Got K.device={k_device}, R.device={r_device}, t.device={t_device}."
        )

    # --- Shape checks ---
    if intrinsics.K.shape != (3, 3):
        raise ValueError(f"K must have shape (3, 3), got {tuple(intrinsics.K.shape)}.")
    if extrinsics.R.shape != (3, 3):
        raise ValueError(f"R must have shape (3, 3), got {tuple(extrinsics.R.shape)}.")
    if extrinsics.t.shape != (3,):
        raise ValueError(f"t must have shape (3,), got {tuple(extrinsics.t.shape)}.")
    if intrinsics.dist_coeffs.ndim != 1:
        raise ValueError(
            f"dist_coeffs must be 1D, got shape {tuple(intrinsics.dist_coeffs.shape)}."
        )

    # --- Dispatch ---
    if intrinsics.is_fisheye:
        return _FisheyeCamera(intrinsics, extrinsics)
    return _PinholeCamera(intrinsics, extrinsics)
