---
phase: 05-packaging-and-release
plan: 01
subsystem: infra
tags: [basedpyright, ci, github-actions, pytorch, typecheck]

# Dependency graph
requires:
  - phase: 04-i-o-layer
    provides: complete codebase with 226 passing tests
provides:
  - basedpyright at "standard" strictness with 0 errors
  - CI test workflow with PyTorch CPU install step
  - CI publish workflow with PyTorch CPU install step in test job
  - validated release.yml with infinite loop guard confirmed
affects:
  - 05-02 (branch protection depends on CI passing cleanly)
  - future releases (publish workflow correctness)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "hatch run pip install for dev-only deps not in pyproject.toml (PyTorch)"
    - "hatch env create then hatch run pip install for CI env setup"

key-files:
  created: []
  modified:
    - pyproject.toml
    - .github/workflows/test.yml
    - .github/workflows/publish.yml

key-decisions:
  - "basedpyright standard mode: 0 errors with no source changes needed - existing codebase already compliant"
  - "PyTorch installed into hatch env via hatch run pip install (not system pip) - hatch venvs are isolated and don't inherit system packages"
  - "typecheck CI job also needs PyTorch: basedpyright resolves torch types from the Python env it uses"
  - "release.yml: no changes needed - infinite loop guard (!startsWith chore(release):) already present"
  - "publish.yml build/publish jobs: no changes needed - python -m build + OIDC trusted publishing already correct"

patterns-established:
  - "CI PyTorch pattern: hatch env create then hatch run pip install torch torchvision --index-url CPU wheel"

# Metrics
duration: 5min
completed: 2026-02-18
---

# Phase 5 Plan 01: basedpyright Standard + CI Validation Summary

**basedpyright bumped to "standard" strictness (0 errors, no source changes) and PyTorch CI installation gap fixed across test and publish workflows**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-02-18T~14:18Z
- **Completed:** 2026-02-18T~14:23Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Bumped basedpyright from "basic" to "standard" in pyproject.toml — 0 errors, 0 warnings, 0 notes immediately; existing codebase already fully standard-compliant
- Identified and fixed the critical CI gap: torch is imported at module load time but was not installed in fresh hatch environments; added `hatch run pip install torch torchvision --index-url CPU wheel` step to test.yml (test job + typecheck job) and publish.yml (test job)
- Verified all three workflow files are syntactically valid (YAML check passed via pre-commit hook); release.yml confirmed correct with infinite loop guard in place

## Task Commits

Each task was committed atomically:

1. **Task 1: Bump basedpyright to standard and fix errors** - `d461139` (chore)
2. **Task 2: Validate CI workflows and fix gaps** - `e6c3cfb` (ci)

## Files Created/Modified

- `pyproject.toml` - Changed `typeCheckingMode = "basic"` to `typeCheckingMode = "standard"`
- `.github/workflows/test.yml` - Added `hatch env create` + `hatch run pip install torch torchvision --index-url CPU` steps to both test and typecheck jobs
- `.github/workflows/publish.yml` - Added `hatch env create` + `hatch run pip install torch torchvision --index-url CPU` step to test job

## Decisions Made

- **PyTorch install via hatch run, not system pip:** `hatch env create` creates an isolated virtualenv that does not inherit system site-packages. Installing torch via `pip install` (system) then running `hatch run test` would fail because hatch uses its own venv. The correct pattern is `hatch run pip install torch torchvision` which targets the hatch env directly.
- **typecheck job also needs PyTorch:** basedpyright resolves types from the Python interpreter in its virtualenv. Without torch installed in the hatch env, basedpyright would produce errors for `import torch` usages across all source files.
- **No source code changes needed:** The standard mode's 5 additional rules (reportFunctionMemberAccess, reportIncompatibleMethodOverride, reportIncompatibleVariableOverride, reportOverlappingOverload, reportPossiblyUnboundVariable) found zero violations — codebase was already compliant.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected PyTorch install placement in workflow**
- **Found during:** Task 2 (Validate CI workflows and fix gaps)
- **Issue:** Plan specified `pip install torch` before `hatch env create`, but hatch virtualenvs are isolated — system pip installs are not visible inside hatch env. Running tests would still fail with ModuleNotFoundError.
- **Fix:** Used `hatch run pip install torch torchvision --index-url CPU` AFTER `hatch env create` to install directly into the hatch virtualenv. Also added explicit `hatch env create` step to typecheck job (was missing; typecheck previously relied on hatch auto-creating env).
- **Files modified:** .github/workflows/test.yml, .github/workflows/publish.yml
- **Verification:** Confirmed locally that torch is installed in hatch env site-packages; all 226 tests pass
- **Committed in:** e6c3cfb (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug in plan's pip install approach)
**Impact on plan:** Essential correction — the original approach would have silently failed in CI. No scope creep.

## Issues Encountered

None beyond the pip install approach correction noted above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- basedpyright standard mode confirmed clean — ready for branch protection configuration
- CI workflows are correctly structured for PyTorch — pushing to main/dev will trigger tests that can actually pass
- publish.yml and release.yml are correct — ready for first release workflow
- Next: 05-02 (configure branch protection and rewiring guide)

---
*Phase: 05-packaging-and-release*
*Completed: 2026-02-18*

## Self-Check: PASSED

- FOUND: pyproject.toml (typeCheckingMode = "standard")
- FOUND: .github/workflows/test.yml (PyTorch CPU steps in test + typecheck jobs)
- FOUND: .github/workflows/publish.yml (PyTorch CPU step in test job)
- FOUND: .planning/phases/05-packaging-and-release/05-01-SUMMARY.md
- FOUND commit d461139: chore(05-01): bump basedpyright to standard strictness
- FOUND commit e6c3cfb: ci(05-01): add PyTorch CPU install step to CI test and typecheck jobs
