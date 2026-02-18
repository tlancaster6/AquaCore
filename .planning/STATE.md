# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-18)

**Core value:** Correct, tested PyTorch implementations of refractive multi-camera geometry that all Aqua consumers share instead of duplicating.
**Current focus:** Phase 1 - Foundation and Physics Math

## Current Position

Phase: 1 of 5 (Foundation and Physics Math)
Plan: 2 of TBD in current phase
Status: In progress
Last activity: 2026-02-18 — Completed Plan 02 (camera models: pinhole, fisheye, create_camera factory)

Progress: [██░░░░░░░░] ~10%

## Performance Metrics

**Velocity:**
- Total plans completed: 2
- Average duration: 27.5 min
- Total execution time: 0.9 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation-and-physics-math | 2 | 55 min | 27.5 min |

**Recent Trend:**
- Last 5 plans: 01-01 (25 min), 01-02 (30 min)
- Trend: stable

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Init]: PyTorch-first, no NumPy math — one implementation, no duplication
- [Init]: Standalone tests only (no AquaCal oracle) — known-value tests, no test-time coupling
- [Init]: Datasets deferred to v2 — keeps v1 focused on geometry foundation
- [Init]: Rewiring guide, not rewiring — AquaCore ships independently
- [01-01]: Pure-PyTorch Rodrigues (not cv2.Rodrigues) — device-agnostic, autograd-compatible, handles theta=0 and theta=pi edge cases
- [01-01]: dist_coeffs stored as float64 (OpenCV requirement); K as float32 (AquaMVS convention)
- [01-01]: (output, valid_mask) return pattern for functions that can fail on individual elements (no NaN, no None)
- [01-01]: conftest.py at tests/ root (not tests/unit/) — shared by all test subdirectories
- [01-02]: OpenCV boundary: always cpu().numpy() before cv2 calls, .to(device) after — documented as non-differentiable in class docstrings
- [01-02]: atan2(|cross|, dot) for round-trip angle tests — float32 acos gives ~4.88e-4 rad noise near 1.0 even for bit-identical rays; atan2 returns exact 0.0
- [01-02]: create_camera() is sole public constructor — _PinholeCamera/_FisheyeCamera prefixed _ and not re-exported

### Pending Todos

None.

### Blockers/Concerns

- [Phase 3]: AquaCal JSON schema field names, shape variants (t: (3,) vs (3,1)), and optional fields must be extracted from AquaCal source before Phase 3 task planning — highest-probability integration risk
- [Phase 1]: CUDA CI runner availability must be confirmed; device-mismatch and autograd pitfalls only surface reliably on CUDA
- [Phase 1]: Glass thickness parameter resolved — simplified air-to-water model chosen (no glass layer)
- [Phase 1]: Two pre-existing test failures in scaffolded test_refraction.py and test_triangulation.py — to be fixed in plans 03/04

## Session Continuity

Last session: 2026-02-18
Stopped at: Completed Phase 1, Plan 02 (01-02-PLAN.md)
Resume file: .planning/phases/01-foundation-and-physics-math/01-03-PLAN.md
