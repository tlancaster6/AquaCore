# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-18)

**Core value:** Correct, tested PyTorch implementations of refractive multi-camera geometry that all Aqua consumers share instead of duplicating.
**Current focus:** Phase 1 - Foundation and Physics Math

## Current Position

Phase: 1 of 5 (Foundation and Physics Math)
Plan: 1 of TBD in current phase
Status: In progress
Last activity: 2026-02-18 — Completed Plan 01 (types, transforms, interface, conftest)

Progress: [█░░░░░░░░░] ~5%

## Performance Metrics

**Velocity:**
- Total plans completed: 1
- Average duration: 25 min
- Total execution time: 0.4 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation-and-physics-math | 1 | 25 min | 25 min |

**Recent Trend:**
- Last 5 plans: 01-01 (25 min)
- Trend: baseline established

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

### Pending Todos

None.

### Blockers/Concerns

- [Phase 3]: AquaCal JSON schema field names, shape variants (t: (3,) vs (3,1)), and optional fields must be extracted from AquaCal source before Phase 3 task planning — highest-probability integration risk
- [Phase 1]: CUDA CI runner availability must be confirmed; device-mismatch and autograd pitfalls only surface reliably on CUDA
- [Phase 1]: Glass thickness parameter resolved — simplified air-to-water model chosen (no glass layer)

## Session Continuity

Last session: 2026-02-18
Stopped at: Completed Phase 1, Plan 01 (01-01-PLAN.md)
Resume file: .planning/phases/01-foundation-and-physics-math/01-02-PLAN.md
