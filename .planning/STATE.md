# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-18)

**Core value:** Correct, tested PyTorch implementations of refractive multi-camera geometry that all Aqua consumers share instead of duplicating.
**Current focus:** Phase 1 - Foundation and Physics Math

## Current Position

Phase: 1 of 5 (Foundation and Physics Math)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-02-18 — Roadmap created; ready to plan Phase 1

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: -
- Trend: -

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Init]: PyTorch-first, no NumPy math — one implementation, no duplication
- [Init]: Standalone tests only (no AquaCal oracle) — known-value tests, no test-time coupling
- [Init]: Datasets deferred to v2 — keeps v1 focused on geometry foundation
- [Init]: Rewiring guide, not rewiring — AquaCore ships independently

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 3]: AquaCal JSON schema field names, shape variants (t: (3,) vs (3,1)), and optional fields must be extracted from AquaCal source before Phase 3 task planning — highest-probability integration risk
- [Phase 1]: CUDA CI runner availability must be confirmed; device-mismatch and autograd pitfalls only surface reliably on CUDA
- [Phase 1]: Glass thickness parameter for InterfaceParams (full air→glass→water chain) needs default and valid range decided before implementation

## Session Continuity

Last session: 2026-02-18
Stopped at: Roadmap created, STATE.md initialized — next step is /gsd:plan-phase 1
Resume file: None
