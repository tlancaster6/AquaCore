"""Rotation and pose transformation utilities.

All functions are pure PyTorch — no NumPy or OpenCV dependencies — so they
are device-agnostic and support autograd.
"""

import torch


def rvec_to_matrix(rvec: torch.Tensor) -> torch.Tensor:
    """Convert a Rodrigues rotation vector to a rotation matrix.

    Uses the Rodrigues formula:
        R = cos(theta)*I + sin(theta)*K + (1 - cos(theta))*(k outer k)
    where theta = ||rvec|| is the rotation angle and k = rvec / theta is the
    unit rotation axis, and K is the skew-symmetric cross-product matrix of k.

    Edge cases:
        - theta < 1e-10: returns identity matrix.

    Args:
        rvec: Rodrigues rotation vector, shape (3,). The magnitude encodes the
            rotation angle in radians; the direction encodes the axis.

    Returns:
        R: Rotation matrix, shape (3, 3), same dtype and device as rvec.
    """
    theta = torch.linalg.norm(rvec)

    if theta < 1e-10:
        return torch.eye(3, dtype=rvec.dtype, device=rvec.device)

    k = rvec / theta  # unit rotation axis, shape (3,)

    # Skew-symmetric cross-product matrix K of k
    K = torch.zeros(3, 3, dtype=rvec.dtype, device=rvec.device)
    K[0, 1] = -k[2]
    K[0, 2] = k[1]
    K[1, 0] = k[2]
    K[1, 2] = -k[0]
    K[2, 0] = -k[1]
    K[2, 1] = k[0]

    cos_a = torch.cos(theta)
    sin_a = torch.sin(theta)
    eye3 = torch.eye(3, dtype=rvec.dtype, device=rvec.device)

    R = cos_a * eye3 + sin_a * K + (1 - cos_a) * torch.outer(k, k)
    return R


def matrix_to_rvec(R: torch.Tensor) -> torch.Tensor:
    """Convert a rotation matrix to a Rodrigues rotation vector.

    Inverse of rvec_to_matrix. Handles the degenerate cases at theta = 0 and
    theta = pi.

    Edge cases:
        - theta < 1e-10: returns zero vector.
        - theta > pi - 1e-6: uses the special-case formula
          k_i = sqrt((R[i,i] + 1) / 2) to extract the axis.

    Args:
        R: Rotation matrix, shape (3, 3). Must be a valid rotation matrix
            (orthogonal, determinant = +1).

    Returns:
        rvec: Rodrigues rotation vector, shape (3,), same dtype and device as R.
    """
    # theta = arccos(clamp((trace(R) - 1) / 2, -1, 1))
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    if theta < 1e-10:
        # Near-identity: no rotation
        return torch.zeros(3, dtype=R.dtype, device=R.device)

    if theta > torch.pi - 1e-6:
        # Near 180 degrees: standard formula is numerically unstable
        # because sin(pi) = 0. Use special-case formula instead.
        # k_i = sqrt((R[i,i] + 1) / 2), sign chosen so that k is consistent.
        k_sq = torch.clamp((torch.diag(R) + 1.0) / 2.0, min=0.0)  # (3,)
        k = torch.sqrt(k_sq)
        # Fix signs using off-diagonal entries:
        # R[0,1] + R[1,0] = 2*(1-cos)*k0*k1 > 0 if k0*k1 > 0
        if R[0, 1] + R[1, 0] < 0:
            k[1] = -k[1]
        if R[0, 2] + R[2, 0] < 0:
            k[2] = -k[2]
        # Normalize to unit length (floating-point rounding may break unit norm)
        k = k / torch.linalg.norm(k).clamp(min=1e-12)
        return theta * k

    # General case: extract axis from (R - R.T) / (2 * sin(theta))
    sin_theta = torch.sin(theta)
    skew = (R - R.T) / (2.0 * sin_theta)
    k = torch.stack([skew[2, 1], skew[0, 2], skew[1, 0]])
    return theta * k


def compose_poses(
    R1: torch.Tensor,
    t1: torch.Tensor,
    R2: torch.Tensor,
    t2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose two world-to-camera poses: (R2, t2) after (R1, t1).

    Given transformations:
        p_cam1 = R1 @ p_world + t1
        p_cam2 = R2 @ p_cam1  + t2

    The composed transformation is:
        p_cam2 = (R2 @ R1) @ p_world + (R2 @ t1 + t2)

    Args:
        R1: First rotation matrix, shape (3, 3), float32.
        t1: First translation vector, shape (3,), float32.
        R2: Second rotation matrix, shape (3, 3), float32.
        t2: Second translation vector, shape (3,), float32.

    Returns:
        R: Composed rotation matrix, shape (3, 3).
        t: Composed translation vector, shape (3,).
    """
    R = R2 @ R1
    t = R2 @ t1 + t2
    return R, t


def invert_pose(
    R: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert a world-to-camera pose to get the camera-to-world pose.

    Given p_cam = R @ p_world + t, the inverse gives:
        p_world = R.T @ p_cam - R.T @ t

    Args:
        R: Rotation matrix (world to camera), shape (3, 3), float32.
        t: Translation vector (world to camera), shape (3,), float32.

    Returns:
        R_inv: Inverse rotation matrix (camera to world), shape (3, 3).
        t_inv: Inverse translation vector (camera to world), shape (3,).
    """
    R_inv = R.T
    t_inv = -R.T @ t
    return R_inv, t_inv


def camera_center(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute the camera center in world coordinates.

    The camera center is the world-frame point that maps to the camera origin
    (0, 0, 0) in camera coordinates. From p_cam = R @ p_world + t:
        0 = R @ C + t  =>  C = -R.T @ t

    Args:
        R: Rotation matrix (world to camera), shape (3, 3), float32.
        t: Translation vector (world to camera), shape (3,), float32.

    Returns:
        C: Camera center in world coordinates, shape (3,), float32.
    """
    return -R.T @ t
