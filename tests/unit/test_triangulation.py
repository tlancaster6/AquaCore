"""Known-value tests for triangulation functions (TRI-01..03, QA-01, QA-02).

All tests parametrize over CPU and CUDA devices via the shared ``device``
fixture in tests/conftest.py. Ground-truth values are computed from known
geometric configurations — no AquaCal or AquaMVS imports are used.

TRI-03 integration test verifies that refractive rays from refraction.py
triangulate correctly end-to-end.
"""

from __future__ import annotations

import pytest
import torch

from aquakit import (
    InterfaceParams,
    point_to_ray_distance,
    triangulate_rays,
)
from aquakit.types import INTERFACE_NORMAL

N_AIR = 1.0
N_WATER = 1.333


def make_interface(device: torch.device, water_z: float = 0.0) -> InterfaceParams:
    """Return a standard flat air-water interface for testing."""
    return InterfaceParams(
        normal=INTERFACE_NORMAL.to(device=device, dtype=torch.float32),
        water_z=water_z,
        n_air=N_AIR,
        n_water=N_WATER,
    )


# ---------------------------------------------------------------------------
# triangulate_rays tests
# ---------------------------------------------------------------------------


class TestTriangulateRays:
    """Tests for triangulate_rays (TRI-01)."""

    def test_triangulate_two_rays_known_point(self, device: torch.device) -> None:
        """Two rays that converge at a known point — must recover that point."""
        target = torch.tensor([0.0, 0.0, 5.0], device=device)

        # Ray 1: from (1, 0, 0) toward target
        o1 = torch.tensor([1.0, 0.0, 0.0], device=device)
        d1 = target - o1
        d1 = d1 / d1.norm()

        # Ray 2: from (-1, 0, 0) toward target
        o2 = torch.tensor([-1.0, 0.0, 0.0], device=device)
        d2 = target - o2
        d2 = d2 / d2.norm()

        result = triangulate_rays([(o1, d1), (o2, d2)])

        torch.testing.assert_close(result, target, atol=1e-5, rtol=0)

    def test_triangulate_three_rays(self, device: torch.device) -> None:
        """Three rays converging at a known point — should recover it accurately."""
        target = torch.tensor([1.0, 2.0, 3.0], device=device)

        origins = [
            torch.tensor([0.0, 0.0, 0.0], device=device),
            torch.tensor([3.0, 0.0, 0.0], device=device),
            torch.tensor([0.0, 4.0, 0.0], device=device),
        ]

        rays = []
        for o in origins:
            d = target - o
            d = d / d.norm()
            rays.append((o, d))

        result = triangulate_rays(rays)
        torch.testing.assert_close(result, target, atol=1e-4, rtol=0)

    def test_triangulate_noisy_rays(self, device: torch.device) -> None:
        """Noisy ray directions — result should still be close to ground truth."""
        target = torch.tensor([0.0, 0.0, 5.0], device=device)

        torch.manual_seed(42)
        origins = [
            torch.tensor([1.0, 0.0, 0.0], device=device),
            torch.tensor([-1.0, 0.0, 0.0], device=device),
            torch.tensor([0.0, 1.0, 0.0], device=device),
            torch.tensor([0.0, -1.0, 0.0], device=device),
        ]

        noise_level = 0.005
        rays = []
        for o in origins:
            d = target - o
            d = d / d.norm()
            # Add small noise to direction then renormalize
            d = d + noise_level * torch.randn(3, device=device)
            d = d / d.norm()
            rays.append((o, d))

        result = triangulate_rays(rays)
        # With small noise, result should be within a reasonable bound
        dist = (result - target).norm().item()
        assert dist < 0.1, f"Noisy triangulation too far from target: dist={dist:.4f}"

    def test_triangulate_degenerate_parallel(self, device: torch.device) -> None:
        """Parallel rays have no unique intersection — must raise ValueError."""
        d = torch.tensor([0.0, 0.0, 1.0], device=device)  # all rays parallel

        o1 = torch.tensor([0.0, 0.0, 0.0], device=device)
        o2 = torch.tensor([1.0, 0.0, 0.0], device=device)

        with pytest.raises(ValueError, match=r"[Dd]egenerate"):
            triangulate_rays([(o1, d), (o2, d)])

    def test_triangulate_unnormalized_directions(self, device: torch.device) -> None:
        """Unnormalized directions should still produce correct result."""
        target = torch.tensor([0.0, 0.0, 4.0], device=device)

        o1 = torch.tensor([1.0, 0.0, 0.0], device=device)
        d1 = (target - o1) * 3.7  # scale up — triangulate_rays normalizes internally

        o2 = torch.tensor([-1.0, 0.0, 0.0], device=device)
        d2 = (target - o2) * 0.5  # scale down

        result = triangulate_rays([(o1, d1), (o2, d2)])
        torch.testing.assert_close(result, target, atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# point_to_ray_distance tests
# ---------------------------------------------------------------------------


class TestPointToRayDistance:
    """Tests for point_to_ray_distance (TRI-02)."""

    def test_point_on_ray(self, device: torch.device) -> None:
        """Point exactly on the ray — distance must be zero."""
        ray_origin = torch.tensor([0.0, 0.0, 0.0], device=device)
        ray_direction = torch.tensor([0.0, 0.0, 1.0], device=device)
        # Point along the ray at t=3
        point = torch.tensor([0.0, 0.0, 3.0], device=device)

        dist = point_to_ray_distance(point, ray_origin, ray_direction)

        torch.testing.assert_close(
            dist, torch.tensor(0.0, device=device), atol=1e-6, rtol=0
        )

    def test_point_off_ray_known_distance(self, device: torch.device) -> None:
        """Point at (1, 0, 0) with ray along Z-axis — distance must be 1.0."""
        ray_origin = torch.tensor([0.0, 0.0, 0.0], device=device)
        ray_direction = torch.tensor([0.0, 0.0, 1.0], device=device)
        point = torch.tensor([1.0, 0.0, 0.0], device=device)

        dist = point_to_ray_distance(point, ray_origin, ray_direction)

        torch.testing.assert_close(
            dist, torch.tensor(1.0, device=device), atol=1e-6, rtol=0
        )

    def test_point_off_ray_2d_offset(self, device: torch.device) -> None:
        """Point at (3, 4, 0) with ray along Z-axis — distance must be 5.0."""
        ray_origin = torch.tensor([0.0, 0.0, 0.0], device=device)
        ray_direction = torch.tensor([0.0, 0.0, 1.0], device=device)
        point = torch.tensor([3.0, 4.0, 0.0], device=device)

        dist = point_to_ray_distance(point, ray_origin, ray_direction)

        torch.testing.assert_close(
            dist, torch.tensor(5.0, device=device), atol=1e-5, rtol=0
        )

    def test_point_to_ray_batch_consistency(self, device: torch.device) -> None:
        """Triangulated point should be close to each input ray (distance < 0.01)."""
        target = torch.tensor([0.5, 0.3, 4.0], device=device)

        origins = [
            torch.tensor([1.0, 0.0, 0.0], device=device),
            torch.tensor([-1.0, 0.0, 0.0], device=device),
            torch.tensor([0.0, 1.0, 0.0], device=device),
        ]

        rays = []
        for o in origins:
            d = target - o
            d = d / d.norm()
            rays.append((o, d))

        p = triangulate_rays(rays)

        for o, d in rays:
            dist = point_to_ray_distance(p, o, d)
            assert dist.item() < 0.01, (
                f"Triangulated point distance to ray: {dist.item():.6f} (expected < 0.01)"
            )


# ---------------------------------------------------------------------------
# TRI-03 Integration: refractive triangulation
# ---------------------------------------------------------------------------


class TestRefractiveTriangulationIntegration:
    """TRI-03: Refractive rays from refraction.py + triangulate_rays end-to-end."""

    def test_refractive_triangulation_integration(self, device: torch.device) -> None:
        """Known underwater point must be recovered by refractive triangulation.

        Setup:
        - Water surface at z=0 (world +Z-down convention)
        - Known underwater point Q at (0.5, 0.5, 1.5)
        - Two cameras in air at (-1, 0, -1) and (1, 0, -1)
        - For each camera: use refractive_project to find the interface point P
          that satisfies Snell's law (camera → P → Q). The water-side ray is
          (origin=P, direction=normalize(Q - P)).
        - Triangulate the two refractive rays → must recover Q within tolerance.

        This validates the full refraction + triangulation pipeline (TRI-03).
        """
        from aquakit import refractive_project

        interface = make_interface(device, water_z=0.0)
        known_point = torch.tensor([0.5, 0.5, 1.5], device=device)

        cam_positions = [
            torch.tensor([-1.0, 0.0, -1.0], device=device),
            torch.tensor([1.0, 0.0, -1.0], device=device),
        ]

        refractive_rays: list[tuple[torch.Tensor, torch.Tensor]] = []

        for cam_pos in cam_positions:
            # Find the interface point P satisfying Snell's law for camera → Q
            ipts, valid = refractive_project(
                known_point.unsqueeze(0),  # (1, 3) underwater target
                cam_pos,  # (3,) camera center
                interface,
            )
            assert valid[0].item(), (
                f"refractive_project should succeed for cam {cam_pos.tolist()}"
            )

            # Water-side ray: origin at interface point P, direction toward Q
            interface_pt = ipts[0]  # (3,)
            water_dir = known_point - interface_pt
            water_dir = water_dir / water_dir.norm()

            refractive_rays.append((interface_pt, water_dir))

        # Triangulate the two refractive rays — should recover Q
        result = triangulate_rays(refractive_rays)

        dist = (result - known_point).norm().item()
        assert dist < 0.01, (
            f"Refractive triangulation error: {dist:.6f} (expected < 0.01)\n"
            f"Result: {result.tolist()}\n"
            f"Expected: {known_point.tolist()}"
        )
