# AquaKit

## What This Is

Shared foundation library for the Aqua ecosystem providing refractive multi-camera geometry, calibration loading, synchronized I/O, and (later) synthetic data generation. Consumed by AquaCal (calibration), AquaMVS (3D surface reconstruction), and AquaPose (3D pose estimation). Extracts duplicated geometry code from AquaCal (NumPy) and AquaMVS (PyTorch) into a single PyTorch-first implementation before AquaPose development begins.

## Core Value

Correct, tested PyTorch implementations of refractive multi-camera geometry (Snell's law, projection, triangulation, transforms) that all Aqua consumers share instead of duplicating.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

(None yet — ship to validate)

### Active

- [ ] Shared types (CameraIntrinsics, CameraExtrinsics, InterfaceParams, Vec2, Vec3, Mat3)
- [ ] Camera models (Camera, FisheyeCamera, create_camera) — PyTorch
- [ ] Interface model (air-water plane, ray-plane intersection)
- [ ] Refraction (snells_law_3d, trace_ray_air_to_water, refractive_project, refractive_back_project)
- [ ] Transforms (rvec_to_matrix, matrix_to_rvec, compose_poses, invert_pose)
- [ ] Triangulation (triangulate_rays, triangulate_point, point_to_ray_distance — batched PyTorch)
- [ ] Projection protocol (ProjectionModel) and RefractiveProjectionModel (Newton-Raphson)
- [ ] Calibration loader (CalibrationData, CameraData, load from AquaCal JSON)
- [ ] Undistortion (compute_undistortion_maps, undistort_image)
- [ ] I/O (FrameSet protocol, VideoSet, ImageSet)
- [ ] Standalone test suite with known-value tests (no AquaCal dependency)
- [ ] Device-parametrized tests (CPU + CUDA)
- [ ] CI (GitHub Actions: lint, test, typecheck — based on existing workflows)
- [ ] PyPI publishing via existing publish workflow
- [ ] Rewiring guide: old-import → new-import mapping table for AquaCal and AquaMVS

### Out of Scope

- Datasets/synthetic data module — deferred to v2 milestone (large scope, not needed for AquaPose kickoff)
- Rewiring AquaCal/AquaMVS source code — separate project, guided by the import mapping
- Cross-validation tests importing AquaCal — standalone tests only, no test-time dependency
- NumPy API wrappers — consumers handle their own conversion at boundaries
- Mobile/embedded targets — desktop Python only
- Photorealistic rendering — v2 with datasets module

## Context

- AquaCal geometry is NumPy; AquaMVS projection/triangulation is already PyTorch. AquaKit consolidates into PyTorch-only with NumPy at serialization boundaries only.
- Coordinate system: World origin at reference camera optical center (+X right, +Y forward, +Z down into water). Camera frame: OpenCV convention. Extrinsics: `p_cam = R @ p_world + t`. Interface normal: `[0, 0, -1]`. Depth: ray depth, not world Z.
- Device convention: follow input tensor device. No explicit `device` param, no hardcoded `.cuda()`.
- Project skeleton already scaffolded: Hatch build system, Ruff linter, basedpyright type checker, GH Actions (test matrix: ubuntu+windows, Python 3.11-3.13), PyPI trusted publishing, semantic-release, Sphinx docs, Codecov.
- Source references: AquaCal at `../AquaCal/src/aquacal/`, AquaMVS at `../AquaMVS/src/aquamvs/`.

## Constraints

- **Python**: >=3.11 (modern syntax: `X | Y` unions, `match` statements)
- **Tensor library**: PyTorch for all math. NumPy only at serialization boundaries (JSON, OpenCV calls).
- **Dependencies**: PyTorch, OpenCV (undistortion, image I/O), kornia (optional, image ops). No heavy ML deps (LightGlue, RoMa, Open3D stay in consumers).
- **Build system**: Hatch (already configured in pyproject.toml)
- **Type checker**: basedpyright in basic mode
- **Packaging**: PyPI from start via existing trusted publishing workflow

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| PyTorch-first, no NumPy math | One implementation, no duplication. Conversion cost negligible for calibration workloads. | — Pending |
| Standalone tests only (no AquaCal oracle) | Removes test-time coupling. Known-value tests are more reliable long-term. | — Pending |
| Datasets deferred to v2 | Keeps v1 focused on geometry foundation. AquaPose can start without datasets. | — Pending |
| Rewiring guide, not rewiring | AquaKit ships independently. Consumer changes are a separate project. | — Pending |
| Device-follows-input convention | Low-level math follows tensor device. Consumers pass device from their config. | — Pending |

---
*Last updated: 2026-02-18 after initialization*
