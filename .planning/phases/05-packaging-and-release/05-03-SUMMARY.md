---
phase: 05-packaging-and-release
plan: 03
subsystem: docs
tags: [rewiring-guide, migration, aquacal, aquamvs, import-mapping]

# Dependency graph
requires:
  - phase: 01-foundation-and-physics-math
    provides: types, refraction, transforms, triangulation exports
  - phase: 02-projection-protocol
    provides: ProjectionModel, RefractiveProjectionModel exports
  - phase: 03-calibration-and-undistortion
    provides: calibration loader, undistortion exports
  - phase: 04-i-o-layer
    provides: FrameSet, ImageSet, VideoSet, create_frameset exports
provides:
  - Complete import migration reference for AquaCal consumers (21 ported symbols)
  - Complete import migration reference for AquaMVS consumers (8 ported symbols)
  - Documented intentional gaps: 12 AquaCal-only modules, 11 AquaMVS-only modules
  - Signature change docs for 6 non-trivial API differences
  - New-in-AquaKit section for 5 additions with no prior equivalent
affects: [AquaCal migration, AquaMVS migration, external consumers]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Rewiring guide lives in .planning/rewiring/ (dev doc, not shipped with package)"
    - "Guide structured by consumer (AquaCal section / AquaMVS section)"

key-files:
  created:
    - .planning/rewiring/REWIRING.md
  modified: []

key-decisions:
  - "Guide structured by consumer, not by module — allows AquaCal and AquaMVS teams to navigate directly to their section"
  - "Signature change depth: examples only for non-obvious changes (TIR pattern, two-step refractive_project, triangulate_rays list API, create_camera dataclass args, load_calibration name change)"
  - "AquaMVS section has no Signature Changes subsection — all AquaMVS → AquaKit ports are identical APIs, pure path migration"

patterns-established:
  - "Import migration guide: find old path in left column, use new path in right column, check Notes column for caveats"
  - "(output, valid_mask) pattern documented as the standard return for fallible geometry functions"

# Metrics
duration: 20min
completed: 2026-02-18
---

# Phase 5 Plan 3: Rewiring Guide Summary

**Import migration reference mapping 21 AquaCal and 8 AquaMVS symbols to aquakit, with signature diffs for 6 non-trivial API changes and NOT PORTED documentation for 23 intentional gaps**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-02-18
- **Completed:** 2026-02-18
- **Tasks:** 2 (Task 1: API verification; Task 2: write guide)
- **Files modified:** 1 created

## Accomplishments

- Verified all 34 symbols in `aquakit.__all__` are covered in the guide (either as migration targets or as new additions)
- Wrote complete AquaCal import table (21 ported symbols) with signature change sections for the 6 non-obvious API differences
- Wrote complete AquaMVS import table (8 ported symbols) with NOT PORTED documentation for 11 pipeline-specific modules
- Documented `create_camera` signature difference (confirmed from source: no `name` arg, takes `CameraIntrinsics`/`CameraExtrinsics` dataclasses directly)
- Added "New in AquaKit" section for 5 new symbols with no prior equivalent (`trace_ray_water_to_air`, `back_project_multi`, `project_multi`, `point_to_ray_distance`, `create_frameset`)

## Task Commits

1. **Task 1: Verify AquaKit public API against research mapping** - verification only, no commit (findings applied directly to Task 2)
2. **Task 2: Write the rewiring guide** - `537aad6` (feat)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified

- `.planning/rewiring/REWIRING.md` - Complete import migration guide (315 lines): AquaCal section, AquaMVS section, prerequisite instructions, new-in-aquakit section

## Decisions Made

- AquaMVS section has no Signature Changes subsection — all 8 AquaMVS ports are pure path migrations (identical function signatures), so a code example section would add noise without value
- Deprecated AquaCal shims (`refractive_project_fast`, `refractive_project_fast_batch`) are listed as "removed" rather than omitted — explicitly flagging removed symbols prevents confusion for users who see them in old AquaCal code
- `create_camera` signature section added even though it was an open question in research — reading `camera.py` confirmed the exact signature

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Committed pre-existing test.yml change from plan 05-01**
- **Found during:** Task 2 (staging files for commit)
- **Issue:** `.github/workflows/test.yml` had uncommitted changes from plan 05-01 (PyTorch CPU install step). Left unstaged when 05-01 was committed as `d461139`.
- **Fix:** Committed `test.yml` separately as `fix(05-01)` before committing the rewiring guide, to keep commits atomic and correctly attributed.
- **Files modified:** `.github/workflows/test.yml`
- **Verification:** Git status clean after both commits
- **Committed in:** `703bd77` (separate fix commit)

---

**Total deviations:** 1 (1 blocking pre-existing uncommitted change)
**Impact on plan:** No scope change; resolved a housekeeping gap from the previous plan without affecting the rewiring guide work.

## Issues Encountered

None — research mapping was comprehensive and accurate. The only open question from research (create_camera signature) was resolved by reading `camera.py` directly.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Rewiring guide is complete. Consumer teams (AquaCal, AquaMVS) can migrate imports using the find-and-replace tables.
- Phase 5 remaining work: Plan 05-02 (GitHub CI/branch protection configuration) — a human-action plan requiring GitHub repository settings and PyPI trusted publisher setup.
- All AquaKit source code is complete and packaged. The rewiring guide is the final dev artifact before release.

---
*Phase: 05-packaging-and-release*
*Completed: 2026-02-18*
