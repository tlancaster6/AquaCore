---
phase: 04-i-o-layer
plan: 01
subsystem: io
tags: [opencv, torch, protocol, structural-typing, image-io, frameset]

# Dependency graph
requires:
  - phase: 02-projection-protocol
    provides: ProjectionModel runtime_checkable Protocol pattern
provides:
  - FrameSet runtime_checkable Protocol (5 methods, context manager)
  - ImageSet class satisfying FrameSet structurally
  - aquacore.io public API (FrameSet, ImageSet)
  - 15 ImageSet unit tests with synthetic image fixtures
affects:
  - 04-02 (VideoSet and create_frameset factory will extend aquacore.io)
  - downstream consumers (AquaMVS, AquaPose) using FrameSet protocol

# Tech tracking
tech-stack:
  added: []
  patterns:
    - runtime_checkable Protocol for multi-camera frame access (mirrors ProjectionModel)
    - case-insensitive glob deduplication via seen-dict (setdefault by filename)
    - BGR-to-(C,H,W)-float32-[0,1] conversion: bgr[..., ::-1].copy() + permute + float / 255.0
    - warnings.warn (stacklevel=2) for init-time data quality; stacklevel=3 for read helpers
    - logging.info for successful init (frame count, camera count)

key-files:
  created:
    - src/aquacore/io/frameset.py
    - tests/unit/test_io/__init__.py
    - tests/unit/test_io/test_imageset.py
  modified:
    - src/aquacore/io/images.py
    - src/aquacore/io/__init__.py

key-decisions:
  - "FrameSet is runtime_checkable Protocol with 5 methods: ImageSet does NOT inherit — structural typing only"
  - "BGR-to-RGB conversion via bgr[..., ::-1].copy(): .copy() required for torch.from_numpy (negative stride)"
  - "Glob deduplication via seen-dict by filename: prevents double-counting on case-insensitive filesystems (Windows)"
  - "Frame count mismatch warning path unreachable in practice (filename matching enforces equal counts)"

patterns-established:
  - "Protocol pattern: @runtime_checkable class with all 5 dunder methods; no abstract base class"
  - "Image glob: both-case extensions + seen-dict dedup by name for cross-platform correctness"
  - "Tensor conversion chain: bgr[..., ::-1].copy() -> torch.from_numpy() -> permute(2,0,1).float() / 255.0"

# Metrics
duration: 25min
completed: 2026-02-18
---

# Phase 4 Plan 01: I/O Layer (FrameSet + ImageSet) Summary

**FrameSet runtime_checkable Protocol and ImageSet class delivering (C,H,W) float32 [0,1] RGB tensors from sorted per-camera image directories, with 15 tests and case-insensitive glob deduplication**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-02-18
- **Completed:** 2026-02-18
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Defined `FrameSet` as a `@runtime_checkable` Protocol with `__getitem__`, `__len__`, `__iter__`, `__enter__`, and `__exit__` — the I/O equivalent of the `ProjectionModel` pattern from Phase 2
- Implemented `ImageSet` with directory validation, sorted filename matching, BGR-to-RGB conversion, and (C, H, W) float32 [0, 1] tensor output; `isinstance(ImageSet(...), FrameSet)` returns True
- Fixed case-insensitive glob deduplication bug (Windows returns both `*.png` and `*.PNG` for the same files) — without this fix frame counts were doubled, breaking all downstream tests
- Wrote 15 unit tests covering construction, tensor format, BGR-to-RGB correctness, iteration semantics, context manager, protocol compliance, all error paths, and memory independence

## Task Commits

Each task was committed atomically:

1. **Task 1: FrameSet Protocol and ImageSet implementation** - `d92ee75` (feat)
2. **Task 2: ImageSet tests and glob deduplication fix** - `fb440cc` (feat)

## Files Created/Modified

- `src/aquacore/io/frameset.py` - FrameSet runtime_checkable Protocol with 5 methods and Google-style docstrings
- `src/aquacore/io/images.py` - ImageSet class: validation, globbing with dedup, BGR->RGB conversion, full Protocol API
- `src/aquacore/io/__init__.py` - Exports FrameSet and ImageSet from aquacore.io
- `tests/unit/test_io/__init__.py` - Empty package init for test_io subpackage
- `tests/unit/test_io/test_imageset.py` - 15 tests with synthetic image fixtures using cv2.imwrite + tmp_path

## Decisions Made

- `FrameSet` is a `runtime_checkable` Protocol — `ImageSet` does NOT inherit from it; structural typing only (same as `ProjectionModel`/`RefractiveProjectionModel` pattern)
- BGR-to-(C,H,W)-float32-[0,1] conversion path: `bgr[..., ::-1].copy()` then `torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0` — `.copy()` is required because negative-stride arrays cannot be wrapped by `torch.from_numpy`
- Frame count mismatch warning path is unreachable through normal construction (filename matching enforces equal counts before count check runs) — tested via subclass override to verify the branch logic

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed case-insensitive glob doubling on Windows**

- **Found during:** Task 2 (ImageSet tests — `test_imageset_construction` failed with `len == 10` instead of 5)
- **Issue:** On Windows, `pathlib.Path.glob("*.png")` and `pathlib.Path.glob("*.PNG")` both match the same files (case-insensitive filesystem), doubling the frame count. The plan's extension list included both cases for Linux portability, but the deduplication step was missing.
- **Fix:** Replaced the simple `files.extend(cam_dir.glob(ext))` loop with a `seen: dict[str, Path]` keyed by filename, using `setdefault` to keep only the first match per name. Produces a deduplicated, sorted file list on both Windows (case-insensitive) and Linux (case-sensitive).
- **Files modified:** `src/aquacore/io/images.py`
- **Verification:** `test_imageset_construction` passes with `len == 5`; all 15 tests pass; `hatch run test` shows 207 passed
- **Committed in:** `fb440cc` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Required for correctness on Windows. Without the fix, all frame count tests would fail. No scope creep.

## Issues Encountered

- `test_imageset_frame_count_mismatch_warns`: The frame count mismatch warning branch in `_validate_and_index` is unreachable through normal construction, because filename validation (which raises `ValueError` on mismatch) necessarily ensures equal counts when it passes. The test uses a subclass override of `_validate_and_index` to inject the mismatch directly, verifying the warning logic is correct even if the natural trigger path does not exist. This is documented in the test docstring.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `aquacore.io` is ready with `FrameSet` and `ImageSet` exported
- Plan 02 will add `VideoSet`, `create_frameset`, and top-level `aquacore/__init__.py` exports
- No blockers; design validated by 15 passing tests

## Self-Check: PASSED

| Check | Result |
|-------|--------|
| `src/aquacore/io/frameset.py` | FOUND |
| `src/aquacore/io/images.py` | FOUND |
| `src/aquacore/io/__init__.py` | FOUND |
| `tests/unit/test_io/__init__.py` | FOUND |
| `tests/unit/test_io/test_imageset.py` | FOUND |
| `.planning/phases/04-i-o-layer/04-01-SUMMARY.md` | FOUND |
| Commit `d92ee75` | FOUND |
| Commit `fb440cc` | FOUND |

---
*Phase: 04-i-o-layer*
*Completed: 2026-02-18*
