"""Known-value tests for refraction functions (REF-01..07, QA-01, QA-02).

All tests parametrize over CPU and CUDA devices via the shared ``device``
fixture in tests/conftest.py. Ground-truth values are computed analytically
from Snell's law — no AquaCal or AquaMVS imports are used.
"""

from __future__ import annotations

import math

import torch

from aquakit import (
    InterfaceParams,
    refractive_back_project,
    refractive_project,
    snells_law_3d,
    trace_ray_air_to_water,
    trace_ray_water_to_air,
)
from aquakit.types import INTERFACE_NORMAL

# ---------------------------------------------------------------------------
# Helper: build a standard air-water interface at z=0
# ---------------------------------------------------------------------------

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
# Snell's law tests
# ---------------------------------------------------------------------------


class TestSnellsLaw3D:
    """Tests for snells_law_3d (REF-01, REF-02)."""

    def test_normal_incidence(self, device: torch.device) -> None:
        """Ray hitting surface at 0° — direction must be unchanged after refraction."""
        # Ray travelling in +Z direction, normal pointing -Z (upward from water)
        incident = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        normal = torch.tensor([0.0, 0.0, -1.0], device=device)
        n_ratio = N_AIR / N_WATER

        dirs, valid = snells_law_3d(incident, normal, n_ratio)

        assert valid[0].item(), "Normal incidence should be valid (no TIR)"
        expected = torch.tensor([0.0, 0.0, 1.0], device=device)
        torch.testing.assert_close(dirs[0], expected, atol=1e-6, rtol=0)

    def test_known_angle_30deg(self, device: torch.device) -> None:
        """30° air incidence → known refracted angle in water via Snell's law."""
        # sin(30°) = 0.5; sin(θ_water) = sin(30°) / 1.333 ≈ 0.3751
        theta_air = math.radians(30.0)
        incident = torch.tensor(
            [[math.sin(theta_air), 0.0, math.cos(theta_air)]], device=device
        )
        normal = torch.tensor([0.0, 0.0, -1.0], device=device)
        n_ratio = N_AIR / N_WATER

        dirs, valid = snells_law_3d(incident, normal, n_ratio)

        assert valid[0].item(), "30° incidence from air is valid"
        # X-component of refracted ray = sin(θ_water)
        sin_theta_water = dirs[0, 0].item()
        expected_sin = math.sin(theta_air) / N_WATER
        assert abs(sin_theta_water - expected_sin) < 1e-5, (
            f"sin(θ_water)={sin_theta_water:.6f}, expected {expected_sin:.6f}"
        )

    def test_known_angle_45deg(self, device: torch.device) -> None:
        """45° air incidence → known refracted angle in water."""
        # sin(45°) / 1.333 ≈ 0.5303
        theta_air = math.radians(45.0)
        incident = torch.tensor(
            [[math.sin(theta_air), 0.0, math.cos(theta_air)]], device=device
        )
        normal = torch.tensor([0.0, 0.0, -1.0], device=device)
        n_ratio = N_AIR / N_WATER

        dirs, valid = snells_law_3d(incident, normal, n_ratio)

        assert valid[0].item(), "45° incidence from air is valid"
        sin_theta_water = dirs[0, 0].item()
        expected_sin = math.sin(theta_air) / N_WATER
        assert abs(sin_theta_water - expected_sin) < 1e-5

    def test_total_internal_reflection(self, device: torch.device) -> None:
        """Water-to-air at steep angle (>critical) — must return valid=False."""
        # Critical angle for n=1.333: sin(θ_c) = 1/1.333 ≈ 0.7502 → θ_c ≈ 48.6°
        # Use 60° > critical angle → TIR
        theta_water = math.radians(60.0)
        incident = torch.tensor(
            [[math.sin(theta_water), 0.0, -math.cos(theta_water)]], device=device
        )
        # Ray going up (-Z) from water, normal pointing -Z (up out of water)
        normal = torch.tensor([0.0, 0.0, -1.0], device=device)
        n_ratio = N_WATER / N_AIR  # water → air

        dirs, valid = snells_law_3d(incident, normal, n_ratio)

        assert not valid[0].item(), "60° water-to-air is total internal reflection"
        # TIR rows should be zeros
        torch.testing.assert_close(
            dirs[0], torch.zeros(3, device=device), atol=1e-6, rtol=0
        )

    def test_critical_angle_boundary(self, device: torch.device) -> None:
        """Ray at exactly the critical angle — must return valid=True (just barely)."""
        # sin(θ_c) = n_air / n_water = 1 / 1.333
        sin_theta_c = N_AIR / N_WATER  # 0.7502...
        cos_theta_c = math.sqrt(1.0 - sin_theta_c**2)

        # Water-to-air: ray going upward (-Z direction)
        incident = torch.tensor(
            [[sin_theta_c, 0.0, -cos_theta_c]], device=device, dtype=torch.float32
        )
        normal = torch.tensor([0.0, 0.0, -1.0], device=device)
        n_ratio = N_WATER / N_AIR

        _dirs, valid = snells_law_3d(incident, normal, n_ratio)

        assert valid[0].item(), (
            "Critical angle boundary should be valid (sin_t_sq == 1)"
        )

    def test_batch_mixed_validity(self, device: torch.device) -> None:
        """Batch of 3 rays: one normal incidence, one partial refraction, one TIR."""
        # Ray 0: 0° incidence air→water — valid
        # Ray 1: 30° incidence air→water — valid
        # Ray 2: 60° incidence water→air — TIR (invalid)
        r0 = [0.0, 0.0, 1.0]
        theta30 = math.radians(30.0)
        r1 = [math.sin(theta30), 0.0, math.cos(theta30)]
        theta60 = math.radians(60.0)
        r2 = [
            math.sin(theta60),
            0.0,
            math.cos(theta60),
        ]  # going +Z (into surface from below)

        # We'll use air→water for rays 0 and 1, then test TIR separately for water→air
        # Instead, test all three with water→air n_ratio to get mixed validity
        n_ratio = N_WATER / N_AIR  # water → air
        normal = torch.tensor([0.0, 0.0, -1.0], device=device)
        incident = torch.tensor([r0, r1, r2], device=device, dtype=torch.float32)
        # Flip so rays go upward (toward the surface) for water→air scenario
        incident = incident * torch.tensor([[1.0, 1.0, -1.0]], device=device)

        _dirs, valid = snells_law_3d(incident, normal, n_ratio)

        # n_water * sin(0°) / n_air = 0 → valid
        assert valid[0].item(), "0° water-to-air should be valid"
        # n_water * sin(30°) / n_air = 1.333 * 0.5 = 0.667 < 1 → valid
        assert valid[1].item(), "30° water-to-air should be valid"
        # n_water * sin(60°) / n_air = 1.333 * 0.866 ≈ 1.154 > 1 → TIR
        assert not valid[2].item(), "60° water-to-air should be TIR"

    def test_snells_law_satisfies_ratio(self, device: torch.device) -> None:
        """Physics invariant: n1 * sin(θ1) == n2 * sin(θ2) for all valid refractions."""
        # Test multiple angles from air into water
        angles = [10.0, 20.0, 30.0, 40.0]
        for angle_deg in angles:
            theta = math.radians(angle_deg)
            incident = torch.tensor(
                [[math.sin(theta), 0.0, math.cos(theta)]], device=device
            )
            normal = torch.tensor([0.0, 0.0, -1.0], device=device)
            n_ratio = N_AIR / N_WATER

            dirs, valid = snells_law_3d(incident, normal, n_ratio)

            assert valid[0].item(), f"{angle_deg}° incidence should be valid"
            # n1 * sin(θ1) = n2 * sin(θ2)
            # sin(θ1) = incident X-component, sin(θ2) = refracted X-component
            n1_sin_theta1 = N_AIR * math.sin(theta)
            n2_sin_theta2 = N_WATER * dirs[0, 0].item()
            assert abs(n1_sin_theta1 - n2_sin_theta2) < 1e-5, (
                f"Snell's ratio violated at {angle_deg}°: "
                f"n1*sin1={n1_sin_theta1:.6f}, n2*sin2={n2_sin_theta2:.6f}"
            )


# ---------------------------------------------------------------------------
# Ray tracing tests
# ---------------------------------------------------------------------------


class TestTraceRayAirToWater:
    """Tests for trace_ray_air_to_water (REF-03)."""

    def test_trace_air_to_water_vertical(self, device: torch.device) -> None:
        """Vertical ray at normal incidence — no bending, interface point at z=water_z."""
        interface = make_interface(device, water_z=0.0)
        # Camera above water at (0, 0, -1), ray going straight down (+Z)
        origins = torch.tensor([[0.0, 0.0, -1.0]], device=device)
        directions = torch.tensor([[0.0, 0.0, 1.0]], device=device)

        ipts, rdirs, valid = trace_ray_air_to_water(origins, directions, interface)

        assert valid[0].item(), "Vertical air-to-water ray should be valid"
        # Interface point should be at z=0 directly below origin
        expected_pt = torch.tensor([0.0, 0.0, 0.0], device=device)
        torch.testing.assert_close(ipts[0], expected_pt, atol=1e-5, rtol=0)
        # No refraction at normal incidence — direction unchanged
        expected_dir = torch.tensor([0.0, 0.0, 1.0], device=device)
        torch.testing.assert_close(rdirs[0], expected_dir, atol=1e-5, rtol=0)

    def test_trace_air_to_water_angled(self, device: torch.device) -> None:
        """Angled ray — interface point lies on water surface; Snell's law holds."""
        interface = make_interface(device, water_z=0.0)
        # Camera at (0, 0, -1), ray at 30° from vertical (+X, +Z)
        theta = math.radians(30.0)
        origins = torch.tensor([[0.0, 0.0, -1.0]], device=device)
        directions = torch.tensor(
            [[math.sin(theta), 0.0, math.cos(theta)]], device=device
        )

        ipts, rdirs, valid = trace_ray_air_to_water(origins, directions, interface)

        assert valid[0].item(), "Angled air-to-water ray should be valid"

        # Interface point Z-coordinate must equal water_z
        assert abs(ipts[0, 2].item() - interface.water_z) < 1e-5, (
            f"Interface point z={ipts[0, 2].item()}, expected {interface.water_z}"
        )

        # Refracted direction satisfies Snell's law: n1*sin(θ1) = n2*sin(θ2)
        n1_sin_theta1 = N_AIR * math.sin(theta)
        n2_sin_theta2 = N_WATER * rdirs[0, 0].item()
        assert abs(n1_sin_theta1 - n2_sin_theta2) < 1e-5

    def test_trace_air_to_water_interface_on_surface(
        self, device: torch.device
    ) -> None:
        """Interface point z-coordinate must always equal water_z."""
        interface = make_interface(device, water_z=0.5)  # non-zero surface
        origins = torch.tensor([[1.0, 0.5, 0.0]], device=device)
        directions = torch.tensor([[0.0, 0.0, 1.0]], device=device)

        ipts, _rdirs, valid = trace_ray_air_to_water(origins, directions, interface)

        assert valid[0].item()
        assert abs(ipts[0, 2].item() - 0.5) < 1e-5


class TestTraceRayWaterToAir:
    """Tests for trace_ray_water_to_air (REF-04)."""

    def test_trace_water_to_air_vertical(self, device: torch.device) -> None:
        """Vertical ray from underwater — no bending at normal incidence."""
        interface = make_interface(device, water_z=0.0)
        # Point underwater at (0, 0, 1), ray going straight up (-Z)
        origins = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        directions = torch.tensor([[0.0, 0.0, -1.0]], device=device)

        ipts, rdirs, valid = trace_ray_water_to_air(origins, directions, interface)

        assert valid[0].item(), "Vertical water-to-air ray should be valid"
        # Interface point at (0, 0, 0)
        expected_pt = torch.tensor([0.0, 0.0, 0.0], device=device)
        torch.testing.assert_close(ipts[0], expected_pt, atol=1e-5, rtol=0)
        # Direction unchanged at normal incidence
        expected_dir = torch.tensor([0.0, 0.0, -1.0], device=device)
        torch.testing.assert_close(rdirs[0], expected_dir, atol=1e-5, rtol=0)

    def test_trace_water_to_air_snell_holds(self, device: torch.device) -> None:
        """Snell's law must hold for water-to-air refraction at valid angles."""
        interface = make_interface(device, water_z=0.0)
        # 20° from vertical upward direction in water (below critical ~48.6°)
        theta_water = math.radians(20.0)
        origins = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        directions = torch.tensor(
            [[math.sin(theta_water), 0.0, -math.cos(theta_water)]], device=device
        )

        _ipts, rdirs, valid = trace_ray_water_to_air(origins, directions, interface)

        assert valid[0].item(), "20° water-to-air should not be TIR"
        # n_water * sin(θ_water) = n_air * sin(θ_air)
        n_water_sin = N_WATER * math.sin(theta_water)
        n_air_sin = N_AIR * rdirs[0, 0].item()
        assert abs(n_water_sin - n_air_sin) < 1e-5

    def test_trace_water_to_air_tir_invalid(self, device: torch.device) -> None:
        """Steep water-to-air ray returns valid=False (TIR)."""
        interface = make_interface(device, water_z=0.0)
        theta_water = math.radians(60.0)  # > critical angle ≈ 48.6°
        origins = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        directions = torch.tensor(
            [[math.sin(theta_water), 0.0, -math.cos(theta_water)]], device=device
        )

        _ipts, _rdirs, valid = trace_ray_water_to_air(origins, directions, interface)

        assert not valid[0].item(), "60° water-to-air should be TIR"


# ---------------------------------------------------------------------------
# Refractive projection tests
# ---------------------------------------------------------------------------


class TestRefractiveProject:
    """Tests for refractive_project (REF-06)."""

    def test_refractive_project_on_axis(self, device: torch.device) -> None:
        """Point directly below camera — interface point has same XY as camera."""
        interface = make_interface(device, water_z=0.0)
        camera_center = torch.tensor([0.0, 0.0, -1.0], device=device)
        # Point directly below camera at depth 2m below surface
        points = torch.tensor([[0.0, 0.0, 2.0]], device=device)

        ipts, valid = refractive_project(points, camera_center, interface)

        assert valid[0].item()
        # Interface point should be at (0, 0, water_z) — no horizontal displacement
        torch.testing.assert_close(
            ipts[0, 0], torch.tensor(0.0, device=device), atol=1e-4, rtol=0
        )
        torch.testing.assert_close(
            ipts[0, 1], torch.tensor(0.0, device=device), atol=1e-4, rtol=0
        )
        assert abs(ipts[0, 2].item() - interface.water_z) < 1e-5

    def test_refractive_project_off_axis(self, device: torch.device) -> None:
        """Point offset horizontally — interface point lies between camera and target."""
        interface = make_interface(device, water_z=0.0)
        camera_center = torch.tensor([0.0, 0.0, -1.0], device=device)
        # Underwater point at (0.5, 0, 1.5)
        points = torch.tensor([[0.5, 0.0, 1.5]], device=device)

        ipts, valid = refractive_project(points, camera_center, interface)

        assert valid[0].item()
        # Interface point Z must equal water_z
        assert abs(ipts[0, 2].item() - interface.water_z) < 1e-5

        # Interface point X must be between camera X (0) and point X (0.5)
        ix = ipts[0, 0].item()
        assert 0.0 <= ix <= 0.5, f"Interface X={ix} not between 0 and 0.5"

    def test_refractive_project_convergence(self, device: torch.device) -> None:
        """Newton-Raphson residual at found interface point must be < 1e-5."""
        interface = make_interface(device, water_z=0.0)
        camera_center = torch.tensor([0.0, 0.0, -1.0], device=device)
        points = torch.tensor([[0.3, 0.2, 1.0]], device=device)

        ipts, valid = refractive_project(points, camera_center, interface)

        assert valid[0].item()

        # Compute Snell's law residual at the found interface point
        ip = ipts[0]
        h_c = abs(camera_center[2].item() - interface.water_z)
        h_q = abs(points[0, 2].item() - interface.water_z)

        # Camera horizontal offset to interface point
        cam_to_ip = torch.tensor(
            [
                ip[0].item() - camera_center[0].item(),
                ip[1].item() - camera_center[1].item(),
            ],
            device=device,
        )
        r_p = torch.linalg.norm(cam_to_ip).item()

        # Interface point to underwater point
        ip_to_q = torch.tensor(
            [points[0, 0].item() - ip[0].item(), points[0, 1].item() - ip[1].item()],
            device=device,
        )
        r_diff = torch.linalg.norm(ip_to_q).item()

        d_air = math.sqrt(r_p**2 + h_c**2)
        d_water = math.sqrt(r_diff**2 + h_q**2)

        if d_air < 1e-10 or d_water < 1e-10:
            return  # on-axis case: residual trivially 0

        sin_air = r_p / d_air
        sin_water = r_diff / d_water
        residual = abs(N_AIR * sin_air - N_WATER * sin_water)

        assert residual < 1e-5, (
            f"Newton-Raphson did not converge: residual={residual:.2e}"
        )

    def test_refractive_project_surface_z(self, device: torch.device) -> None:
        """Interface point Z must always equal water_z (for non-zero water_z)."""
        interface = make_interface(device, water_z=0.3)
        camera_center = torch.tensor([0.0, 0.0, 0.0], device=device)
        points = torch.tensor([[0.4, 0.1, 1.5]], device=device)

        ipts, valid = refractive_project(points, camera_center, interface)

        assert valid[0].item()
        assert abs(ipts[0, 2].item() - 0.3) < 1e-5, (
            f"Interface Z={ipts[0, 2].item()}, expected 0.3"
        )


# ---------------------------------------------------------------------------
# Refractive back-projection tests
# ---------------------------------------------------------------------------


class TestRefractiveBackProject:
    """Tests for refractive_back_project (REF-07)."""

    def test_refractive_back_project_vertical(self, device: torch.device) -> None:
        """Vertical pixel ray — normal incidence, no bending."""
        interface = make_interface(device, water_z=0.0)
        camera_centers = torch.tensor([0.0, 0.0, -1.0], device=device)  # (3,)
        pixel_rays = torch.tensor([[0.0, 0.0, 1.0]], device=device)  # straight down

        ipts, water_dirs, valid = refractive_back_project(
            pixel_rays, camera_centers, interface
        )

        assert valid[0].item()
        # Interface point at z=0
        assert abs(ipts[0, 2].item() - 0.0) < 1e-5
        # Direction in water same as in air (normal incidence)
        expected_dir = torch.tensor([0.0, 0.0, 1.0], device=device)
        torch.testing.assert_close(water_dirs[0], expected_dir, atol=1e-5, rtol=0)

    def test_refractive_back_project_consistency(self, device: torch.device) -> None:
        """Back-project a ray, then reverse from water to air — interface XY must agree."""
        interface = make_interface(device, water_z=0.0)
        camera_centers = torch.tensor([0.0, 0.0, -1.0], device=device)
        # Angled pixel ray at 20° from vertical
        theta = math.radians(20.0)
        pixel_rays = torch.tensor(
            [[math.sin(theta), 0.0, math.cos(theta)]], device=device
        )

        # Forward: cast from camera through air into water
        ipts_fwd, water_dirs, valid_fwd = refractive_back_project(
            pixel_rays, camera_centers, interface
        )

        assert valid_fwd[0].item()

        # Reverse: trace upward from just below the interface using the *negated*
        # water direction (going back toward the surface from below).
        # The reversed ray should intersect the surface at the same interface XY.
        origins_rev = ipts_fwd + torch.tensor([[0.0, 0.0, 0.001]], device=device)
        reverse_dirs = -water_dirs  # flip: now pointing upward (toward surface)

        ipts_rev, _air_dirs, valid_rev = trace_ray_water_to_air(
            origins_rev, reverse_dirs, interface
        )

        assert valid_rev[0].item(), "Reversed water ray should reach the surface"

        # Both forward and reverse interface points should agree (up to offset)
        torch.testing.assert_close(ipts_fwd[0, :2], ipts_rev[0, :2], atol=1e-3, rtol=0)

    def test_refractive_back_project_batch_camera(self, device: torch.device) -> None:
        """Per-ray camera centers (N, 3) shape — all rays valid."""
        interface = make_interface(device, water_z=0.0)
        # Two cameras at different positions, one ray each
        camera_centers = torch.tensor(
            [[0.0, 0.0, -1.0], [-0.5, 0.0, -1.0]], device=device
        )
        pixel_rays = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device=device)

        ipts, _water_dirs, valid = refractive_back_project(
            pixel_rays, camera_centers, interface
        )

        assert valid.all().item()
        # Both interface points should be at z=0
        assert (ipts[:, 2].abs() < 1e-5).all().item()
