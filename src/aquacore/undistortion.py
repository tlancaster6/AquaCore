"""Undistortion map computation and image remapping for pinhole and fisheye cameras."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .calibration import CameraData


def compute_undistortion_maps(
    camera_data: CameraData,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute undistortion remap maps from calibration data.

    Dispatches to either the standard pinhole or fisheye OpenCV undistortion
    pipeline depending on ``camera_data.intrinsics.is_fisheye``.

    Fisheye path uses ``cv2.fisheye.estimateNewCameraMatrixForUndistortRectify``
    then ``cv2.fisheye.initUndistortRectifyMap``.
    Pinhole path uses ``cv2.getOptimalNewCameraMatrix`` (alpha=0) then
    ``cv2.initUndistortRectifyMap``.

    This is a non-differentiable OpenCV boundary operation. The returned maps
    are CPU NumPy arrays intended for use with :func:`undistort_image`.

    Args:
        camera_data: Typed calibration data for a single camera.

    Returns:
        Tuple ``(map_x, map_y)`` of ``np.float32`` arrays each with shape
        ``(height, width)`` suitable for ``cv2.remap``.
    """
    intr = camera_data.intrinsics
    image_size = intr.image_size  # (width, height)

    K = intr.K.cpu().numpy().astype(np.float64)
    dist = intr.dist_coeffs.cpu().numpy().astype(np.float64)

    if intr.is_fisheye:
        D = dist.reshape(4, 1)
        K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, image_size, np.eye(3)
        )
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K_new, image_size, cv2.CV_32FC1
        )
    else:
        K_new, _ = cv2.getOptimalNewCameraMatrix(
            K, dist, image_size, alpha=0, newImgSize=image_size
        )
        map_x, map_y = cv2.initUndistortRectifyMap(
            K, dist, None, K_new, image_size, cv2.CV_32FC1
        )

    return map_x, map_y


def undistort_image(
    image: torch.Tensor,
    maps: tuple[np.ndarray, np.ndarray],
) -> torch.Tensor:
    """Undistort an image tensor using precomputed remap maps.

    Converts the input PyTorch tensor to a NumPy array for ``cv2.remap``, then
    converts the result back to a PyTorch tensor on the original device.

    This is a non-differentiable OpenCV boundary operation.

    Args:
        image: Image tensor of shape ``(H, W, 3)`` or ``(H, W)``, dtype uint8.
            Any device is supported; the tensor is moved to CPU for remapping
            and the result is moved back to the original device.
        maps: ``(map_x, map_y)`` tuple as returned by
            :func:`compute_undistortion_maps`.

    Returns:
        Undistorted image tensor of the same shape and dtype as ``image``,
        on the same device as ``image``.
    """
    device = image.device
    map_x, map_y = maps

    img_np = image.detach().cpu().numpy()
    result = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR)
    return torch.from_numpy(result).to(device)
