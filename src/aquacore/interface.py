"""Air-water plane model and ray-plane intersection geometry."""

import torch


def ray_plane_intersection(
    origins: torch.Tensor,
    directions: torch.Tensor,
    plane_normal: torch.Tensor,
    plane_d: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute intersections between rays and a plane.

    The plane is defined by: dot(point, plane_normal) = plane_d.
    For a flat horizontal air-water interface at world Z = water_z, use
    plane_normal = [0, 0, -1] and plane_d = -water_z (so that the equation
    dot(p, [0,0,-1]) = -water_z simplifies to p_z = water_z).

    Equivalently, pass plane_normal = [0, 0, 1] and plane_d = water_z.

    Args:
        origins: Ray origins, shape (N, 3).
        directions: Ray direction unit vectors, shape (N, 3).
        plane_normal: Plane normal vector, shape (3,). Does not need to be
            normalized; the dot product is used directly.
        plane_d: Signed distance parameter. The plane satisfies
            dot(p, plane_normal) = plane_d.

    Returns:
        points: Intersection points, shape (N, 3). Points are meaningful only
            where valid is True.
        valid: Boolean mask, shape (N,). False when the ray is parallel to the
            plane (abs(denom) < 1e-12) or when the intersection is behind the
            ray origin (t < 0).
    """
    # denom = dot(direction, plane_normal) for each ray
    denom = (directions * plane_normal).sum(dim=-1)  # (N,)

    # Numerator: plane_d - dot(origin, plane_normal)
    numer = plane_d - (origins * plane_normal).sum(dim=-1)  # (N,)

    # Ray is parallel to plane when abs(denom) < 1e-12
    parallel = denom.abs() < 1e-12

    # Compute t safely (avoid division by zero)
    safe_denom = torch.where(parallel, torch.ones_like(denom), denom)
    t = numer / safe_denom  # (N,)

    # Mark invalid: parallel rays or intersections behind the ray origin
    valid = ~parallel & (t >= 0.0)

    # Intersection points: origin + t * direction
    points = origins + t.unsqueeze(-1) * directions  # (N, 3)

    return points, valid
