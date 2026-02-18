"""ProjectionModel protocol defining the projection/back-projection interface."""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class ProjectionModel(Protocol):
    """Protocol for geometric projection models.

    Any class implementing ``project()`` and ``back_project()`` with the
    correct signatures satisfies this protocol structurally — no import of
    ``ProjectionModel`` is needed in the implementing class.

    The ``@runtime_checkable`` decorator enables both static type-checking
    (basedpyright structural subtyping) and runtime ``isinstance()`` checks.
    """

    def project(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project 3D world points to 2D pixel coordinates.

        Args:
            points: 3D points in world frame, shape (N, 3), float32.
                Points are assumed to be underwater (below the air-water
                interface). Points above the interface produce invalid results.

        Returns:
            pixels: 2D pixel coordinates (u, v), shape (N, 2), float32.
                Invalid pixels are NaN.
            valid: Boolean validity mask, shape (N,). False where the point
                is above the water surface or behind the camera.
        """
        ...

    def back_project(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cast refracted rays from pixel coordinates through air-water interface.

        Converts pixels to rays in air (via K_inv), finds where each ray
        intersects the water surface, then applies Snell's law to compute
        the refracted direction in water.

        Args:
            pixels: 2D pixel coordinates (u, v), shape (N, 2), float32.

        Returns:
            origins: Ray origin points on water surface, shape (N, 3), float32.
                Each origin lies on the air-water interface (z = water_z).
            directions: Unit ray direction vectors into water, shape (N, 3),
                float32. Normalized.
        """
        ...
