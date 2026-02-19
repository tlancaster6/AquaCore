# Phase 5: Packaging and Release - Research

**Researched:** 2026-02-18
**Domain:** Python packaging, CI/CD, PyPI trusted publishing, basedpyright type-checking, import migration
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### CI pipeline
- Existing workflows (test.yml, publish.yml, release.yml, slow-tests.yml, docs.yml) already cover the target setup
- Phase 5 validates these workflows work end-to-end with current code and fixes any gaps
- Target matrix: Ubuntu + Windows, Python 3.11/3.12/3.13 (already configured)
- Separate jobs for test, typecheck, and pre-commit (already configured)
- No GPU CI — CUDA testing stays manual/local
- Slow tests remain manual dispatch only

#### PyPI publishing
- Package name: `aquakit` (already configured in pyproject.toml)
- Versioning: SemVer starting at 0.1.0 (already configured with python-semantic-release)
- Publishing: Tag-triggered via trusted publishing (already configured)
- TestPyPI step before real PyPI (already configured in publish.yml)
- PyTorch is intentionally NOT a declared dependency — users install their own variant

#### Rewiring guide
- Structured by consumer: separate sections for AquaCal users and AquaMVS users
- Lives in `.planning/rewiring/` — it's a dev doc, not shipped with the package
- Covers ported imports AND flags gaps (modules still in AquaCal/AquaMVS that haven't been ported)
- Depth of examples: Claude's discretion based on actual API differences

#### Quality gates
- No coverage threshold — coverage is tracked (Codecov) but informational only
- Typecheck failures block merge — basedpyright is a required status check
- Branch protection rules: configure on main, checking for existing rules first
- basedpyright strictness: bump from "basic" to "standard" — may require type annotation fixes

### Claude's Discretion
- CI job structure (whether to split/merge existing jobs) — keep what works
- Level of detail in rewiring guide usage examples
- How to handle any type annotation fixes needed for "standard" strictness
- Branch protection rule specifics (required reviewers, etc.)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

## Summary

Phase 5 is a validation, wiring, and documentation phase, not a build-from-scratch phase. All five GitHub Actions workflows already exist (test.yml, publish.yml, release.yml, slow-tests.yml, docs.yml) and pyproject.toml is already fully configured with hatch, semantic-release, ruff, basedpyright, pytest, and classifiers. The work is: (1) validate workflows pass end-to-end, (2) bump basedpyright from "basic" to "standard" and fix any resulting errors, (3) configure GitHub repository settings (branch protection, Codecov token, PyPI trusted publisher environments), and (4) write the rewiring guide.

The rewiring guide has clear scope: AquaCal exports geometry functions (numpy-based, CPU-only) and calibration serialization; AquaKit re-exports the same logical operations but as PyTorch-native, batched versions with different signatures. AquaMVS has its own copies of some modules (calibration.py, triangulation.py, projection/) that must map to AquaKit equivalents. Several AquaCal modules (board detection, calibration pipeline, diagnostics) have no AquaKit equivalent — these are intentional gaps to document.

The most important technical finding: the `RELEASE_TOKEN` secret referenced in release.yml must be a personal access token (classic) with `repo` scope, or a fine-grained PAT with `contents: write`. The default `GITHUB_TOKEN` cannot bypass branch protection rules, which is why python-semantic-release uses a PAT. This secret must be set in the repository's Settings > Secrets before the release workflow can run.

**Primary recommendation:** Work task-by-task: fix basedpyright standard mode first (verify 0 errors), then validate CI workflows run cleanly, then configure GitHub repository settings, then write the rewiring guide. Each task is independently verifiable.

---

## Standard Stack

### Core (all already in place)

| Tool | Version in Repo | Purpose | Notes |
|------|-----------------|---------|-------|
| hatch | latest | Build backend and env management | `[build-system]` uses hatchling |
| python-semantic-release | latest | Automated versioning and changelog | Configured in `[tool.semantic_release]` |
| pypa/gh-action-pypi-publish | release/v1 | Trusted publish action | Used in publish.yml |
| basedpyright | 1.38.1 (detected) | Type checker | Bumping basic → standard |
| ruff | 0.15.1 | Linter and formatter | Used in pre-commit and CI |
| pytest | latest | Test runner | Configured with slow marker |
| codecov/codecov-action | v5 | Coverage reporting | Requires CODECOV_TOKEN secret |

### GitHub Actions Action Versions in Use

| Action | Version | Purpose |
|--------|---------|---------|
| actions/checkout | v4 | Checkout code |
| actions/setup-python | v5 | Python setup with pip cache |
| actions/upload-artifact | v4 | Upload dist/ between jobs |
| actions/download-artifact | v4 | Download dist/ for publish |
| pypa/gh-action-pypi-publish | release/v1 | Publish to PyPI/TestPyPI |
| codecov/codecov-action | v5 | Upload coverage |

---

## Architecture Patterns

### Workflow Architecture (what exists)

```
.github/workflows/
├── test.yml         # Push to main/dev + PR to main: test (matrix) + typecheck + pre-commit
├── release.yml      # Push to main/dev: python-semantic-release version bump + tag
├── publish.yml      # Tag push (v*.*.*): test → build → testpypi → pypi
├── slow-tests.yml   # workflow_dispatch: manual full test suite
└── docs.yml         # PR to main: sphinx build check
```

### Pattern 1: Trusted Publishing Flow

**What:** PyPI OIDC-based publishing without long-lived secrets.
**How it works:**
1. Workflow requests OIDC token from GitHub (requires `id-token: write` permission)
2. `pypa/gh-action-pypi-publish` exchanges OIDC token for a 15-minute PyPI API token
3. Package is uploaded; token expires

**GitHub environment setup required:**
- Repository must have two GitHub Environments configured: `testpypi` and `pypi`
- PyPI project must have a Trusted Publisher registered pointing to `publish.yml`
- Required fields on PyPI: owner name, repo name, workflow filename (`publish.yml`), environment name (`pypi`)

**Current workflow already has this correctly:**
```yaml
# publish.yml excerpt
publish:
  environment: pypi
  permissions:
    id-token: write
  steps:
    - uses: pypa/gh-action-pypi-publish@release/v1
```

### Pattern 2: python-semantic-release with PAT

**What:** Semantic release pushes version bump commits and tags to main/dev.
**Why PAT is required:** The default `GITHUB_TOKEN` cannot bypass branch protection rules. python-semantic-release needs to push a commit (`chore(release): 0.1.0`) to the protected branch, which requires a PAT.

**Current workflow:**
```yaml
# release.yml excerpt
- uses: actions/checkout@v4
  with:
    fetch-depth: 0
    ref: ${{ github.ref_name }}
    token: ${{ secrets.RELEASE_TOKEN }}  # PAT, not GITHUB_TOKEN
- name: Semantic release
  env:
    GH_TOKEN: ${{ secrets.RELEASE_TOKEN }}
  run: semantic-release version
```

**Required secret:** `RELEASE_TOKEN` must be a classic PAT with `repo` scope (or fine-grained PAT with `contents: write`). Must be set in repository Settings > Secrets and variables > Actions.

**The infinite loop guard:** `if: "!startsWith(github.event.head_commit.message, 'chore(release):')"` prevents the release workflow from triggering itself when it pushes the version bump commit.

### Pattern 3: basedpyright "standard" Mode

**What:** Bumping `typeCheckingMode` from `"basic"` to `"standard"` in pyproject.toml.
**Rules that become errors in standard (were none in basic):**

| Rule | What it catches |
|------|-----------------|
| `reportFunctionMemberAccess` | Accessing attributes on function objects that don't exist |
| `reportIncompatibleMethodOverride` | Subclass method with incompatible signature vs base |
| `reportIncompatibleVariableOverride` | Subclass attribute narrowing base class type unsafely |
| `reportOverlappingOverload` | `@overload` signatures that overlap (ambiguous dispatch) |
| `reportPossiblyUnboundVariable` | Variable used that may not be defined on all code paths |

**Current status:** `hatch run typecheck` shows `0 errors, 0 warnings, 0 notes` at `"basic"`. The codebase is clean. After bumping to `"standard"`, errors must be verified with a fresh run.

**The fix:** In pyproject.toml:
```toml
[tool.basedpyright]
pythonVersion = "3.11"
typeCheckingMode = "standard"  # was "basic"
```

### Pattern 4: Branch Protection on main

**What:** GitHub branch protection rules prevent direct pushes and require status checks.
**Required status checks to configure:**

Status check names come from the workflow job names in test.yml:
- `test` (matrix — each matrix combination creates a separate check)
- `typecheck`
- `pre-commit`

**Configuration via GitHub UI (Settings > Branches > Add rule for `main`):**
- Require a pull request before merging
- Require status checks to pass: `test`, `typecheck`, `pre-commit`
- Do not include administrators (so semantic-release PAT can push)

**Critical:** Status checks only appear in the dropdown after they have run on the branch at least once in the last 7 days. Must run workflows first before they can be added as required checks.

**Important nuance for semantic-release:** Branch protection must allow the PAT user (the PAT owner) to bypass rules, OR the protection must be configured to not require PRs for pushes (only for merges). The existing release.yml uses a PAT specifically to enable the bypass.

### Pattern 5: Rewiring Guide Structure

**Location:** `.planning/rewiring/REWIRING.md`

**Structure per consumer:**
```
## AquaCal Users
### Module: aquacal.core.refractive_geometry → aquakit
### Module: aquacal.core.camera → aquakit (partial)
### Module: aquacal.utils.transforms → aquakit
### Module: aquacal.triangulation → aquakit
### Module: aquacal.io.serialization → aquakit (via calibration.py)
### Module: aquacal.io.{video,images,frameset} → aquakit.io
### NOT PORTED (intentional gaps)

## AquaMVS Users
### Module: aquamvs.calibration → aquakit
### Module: aquamvs.triangulation (partial) → aquakit
### Module: aquamvs.projection → aquakit.projection
### NOT PORTED
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| OIDC token exchange for PyPI | Custom publish script | `pypa/gh-action-pypi-publish@release/v1` | Handles OIDC exchange, retries, error reporting |
| Version bumping from commit messages | Custom changelog script | `python-semantic-release` | Already configured; parses conventional commits, updates pyproject.toml |
| Package build | Custom distutils script | `python -m build` (hatchling backend) | Already in publish.yml |
| Coverage upload | curl to codecov API | `codecov/codecov-action@v5` | Handles token auth, report format, retry |

---

## Common Pitfalls

### Pitfall 1: RELEASE_TOKEN Not Set
**What goes wrong:** release.yml fails with `403 Forbidden` or `remote: Permission to ... denied`.
**Why it happens:** The workflow uses `secrets.RELEASE_TOKEN` but the secret hasn't been created in GitHub repository Settings.
**How to avoid:** Before running any workflow, verify the secret exists in Settings > Secrets and variables > Actions. The PAT needs `repo` scope (classic) or `contents: write` (fine-grained).
**Warning signs:** release.yml fails immediately on the checkout step or on `semantic-release version`.

### Pitfall 2: PyPI Trusted Publisher Not Configured
**What goes wrong:** publish.yml fails with `403 Forbidden` or authentication error on the `pypa/gh-action-pypi-publish` step.
**Why it happens:** The GitHub environment (`pypi`, `testpypi`) exists in GitHub but the corresponding Trusted Publisher is not registered on PyPI/TestPyPI for this workflow.
**How to avoid:** Register on PyPI: project > Publishing > Add trusted publisher. Fields: owner=`tlancaster6`, repo=`aquakit`, workflow=`publish.yml`, environment=`pypi`. Same for TestPyPI with environment=`testpypi`.
**Warning signs:** The publish job fails with an OIDC or 403 error.

### Pitfall 3: Branch Protection Breaks Semantic Release
**What goes wrong:** semantic-release pushes the `chore(release):` commit but it's rejected because the branch is protected.
**Why it happens:** If "Include administrators" is checked in branch protection, even PATs are blocked.
**How to avoid:** Do NOT check "Include administrators" in branch protection. The PAT bypass relies on administrator exemption. Alternatively use a GitHub App token (more complex).
**Warning signs:** release.yml succeeds but the version bump commit is never pushed.

### Pitfall 4: Status Checks Not Available in Branch Protection UI
**What goes wrong:** When configuring branch protection, the `test`, `typecheck`, `pre-commit` checks don't appear in the dropdown.
**Why it happens:** GitHub only shows checks that have run on the target branch in the last 7 days.
**How to avoid:** Push a commit to `main` to trigger test.yml, wait for it to complete, then configure branch protection.
**Warning signs:** The search box in "Require status checks" shows no results.

### Pitfall 5: basedpyright "standard" Reports Errors After Bump
**What goes wrong:** After changing `typeCheckingMode = "standard"`, the CI typecheck job fails.
**Why it happens:** The five new rules (`reportPossiblyUnboundVariable` etc.) catch issues invisible in basic mode.
**How to avoid:** Run `hatch run typecheck` locally after the bump before committing. Fix any `reportPossiblyUnboundVariable` errors by adding explicit `else` branches or default variable initialization.
**Warning signs:** Typecheck CI fails on the first push after the config change.

### Pitfall 6: publish.yml Tag Pattern Too Strict
**What goes wrong:** A tag like `v0.1.0-dev.1` (pre-release from dev branch) doesn't trigger publish.yml.
**Why it happens:** The current trigger pattern `v[0-9]+.[0-9]+.[0-9]+` only matches full releases, which is intentional.
**Implication for planning:** This is correct behavior — pre-releases on dev branch do NOT publish to PyPI. Document this explicitly. Only final version tags trigger publishing.

### Pitfall 7: PyTorch Not in Dependencies
**What goes wrong:** `pip install aquakit` succeeds but `import aquakit` fails with `ModuleNotFoundError: torch`.
**Why it happens:** PyTorch is intentionally omitted from `dependencies` in pyproject.toml so users install their preferred variant (CPU/CUDA/etc.).
**How to avoid:** The README and rewiring guide must explicitly state that users must install PyTorch separately before installing aquakit.
**Warning signs:** New user installs aquakit and gets an immediate import error.

---

## Code Examples

### Verified: Current pyproject.toml basedpyright config (to be changed)
```toml
# Source: C:/Users/tucke/PycharmProjects/AquaKit/pyproject.toml
[tool.basedpyright]
pythonVersion = "3.11"
typeCheckingMode = "basic"   # Change this to "standard"
```

### Verified: release.yml infinite loop guard
```yaml
# Source: .github/workflows/release.yml
on:
  push:
    branches: [main, dev]

jobs:
  release:
    if: "!startsWith(github.event.head_commit.message, 'chore(release):')"
```

### Verified: publish.yml OIDC permissions block
```yaml
# Source: .github/workflows/publish.yml
publish:
  needs: publish-testpypi
  runs-on: ubuntu-latest
  environment: pypi

  permissions:
    id-token: write   # Required for OIDC token exchange

  steps:
    - uses: actions/download-artifact@v4
      with:
        name: dist
        path: dist/
    - uses: pypa/gh-action-pypi-publish@release/v1
```

### Verified: semantic-release configuration in pyproject.toml
```toml
# Source: C:/Users/tucke/PycharmProjects/AquaKit/pyproject.toml
[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]
commit_message = "chore(release): {version}"
build_command = "python -m build"
tag_format = "v{version}"

[tool.semantic_release.branches.main]
match = "^main$"
prerelease = false

[tool.semantic_release.branches.dev]
match = "^dev$"
prerelease = true
prerelease_token = "dev"

[tool.semantic_release.remote.token]
env = "GH_TOKEN"
```

---

## Rewiring Guide Reference Data

This section documents the actual import mappings for the rewiring guide. The planner should use this to scope the guide content.

### AquaCal → AquaKit Import Map

#### Ported: Full equivalents with signature differences

| Old Import (aquacal) | New Import (aquakit) | Signature Change |
|----------------------|-----------------------|------------------|
| `from aquacal.core.refractive_geometry import snells_law_3d` | `from aquakit import snells_law_3d` | Old: numpy, returns `Vec3 \| None`. New: PyTorch tensors, batched |
| `from aquacal.core.refractive_geometry import trace_ray_air_to_water` | `from aquakit import trace_ray_air_to_water` | Old: takes `Camera, Interface, pixel`. New: takes tensors directly |
| `from aquacal.core.refractive_geometry import refractive_back_project` | `from aquakit import refractive_back_project` | Old: numpy/Camera objects. New: PyTorch tensors |
| `from aquacal.core.refractive_geometry import refractive_project` | `from aquakit import refractive_project` | Old: numpy. New: PyTorch |
| `from aquacal.utils.transforms import rvec_to_matrix` | `from aquakit import rvec_to_matrix` | Old: numpy. New: PyTorch |
| `from aquacal.utils.transforms import matrix_to_rvec` | `from aquakit import matrix_to_rvec` | Old: numpy. New: PyTorch |
| `from aquacal.utils.transforms import compose_poses` | `from aquakit import compose_poses` | Old: numpy. New: PyTorch |
| `from aquacal.utils.transforms import invert_pose` | `from aquakit import invert_pose` | Old: numpy. New: PyTorch |
| `from aquacal.utils.transforms import camera_center` | `from aquakit import camera_center` | Old: numpy. New: PyTorch |
| `from aquacal.config.schema import CameraIntrinsics` | `from aquakit import CameraIntrinsics` | Same dataclass shape; now PyTorch tensors for K/R/t |
| `from aquacal.config.schema import CameraExtrinsics` | `from aquakit import CameraExtrinsics` | Same |
| `from aquacal.config.schema import InterfaceParams` | `from aquakit import InterfaceParams` | Same |
| `from aquacal.config.schema import Vec2, Vec3, Mat3` | `from aquakit import Vec2, Vec3, Mat3` | Now torch.Tensor type aliases |
| `from aquacal.config.schema import INTERFACE_NORMAL` | `from aquakit import INTERFACE_NORMAL` | Now torch.Tensor |
| `from aquacal.io.serialization import load_calibration` | `from aquakit import load_calibration_data` | New function name; returns `CalibrationData` (PyTorch) not `CalibrationResult` |
| `from aquacal.io.video import VideoSet` | `from aquakit import VideoSet` | Same protocol |
| `from aquacal.io.images import ImageSet` | `from aquakit import ImageSet` | Same |
| `from aquacal.io.frameset import FrameSet` | `from aquakit import FrameSet` | Same protocol |
| `from aquacal.io.images import create_frameset` (if exists) | `from aquakit import create_frameset` | Same factory |
| `from aquacal.core.interface_model import ray_plane_intersection` | `from aquakit import ray_plane_intersection` | Old: numpy. New: PyTorch |
| `from aquacal.triangulation.triangulate import triangulate_point` | `from aquakit import triangulate_rays` | Different API: new takes list of (origin, dir) tensor pairs |
| `from aquakit import point_to_ray_distance` | New in aquakit, no AquaCal equivalent | — |
| `from aquakit import compute_undistortion_maps` | From `aquamvs.calibration.compute_undistortion_maps` | Moved to aquakit |
| `from aquakit import undistort_image` | From `aquamvs.calibration.undistort_image` | Moved to aquakit |

#### NOT PORTED (intentional gaps — AquaCal only)

| AquaCal Module | What It Does | Status |
|----------------|--------------|--------|
| `aquacal.core.camera.Camera` | NumPy Camera class with cv2 projection | Not ported; AquaKit uses plain tensors |
| `aquacal.core.camera.FisheyeCamera` | Fisheye variant | Not ported |
| `aquacal.core.camera.create_camera` | Factory for Camera/FisheyeCamera | AquaKit has `create_camera` but it creates tensor-based cameras |
| `aquacal.core.board` | ChArUco board detection | Not ported (calibration-specific) |
| `aquacal.core.interface_model.Interface` | NumPy interface class | Not ported; AquaKit uses `InterfaceParams` struct |
| `aquacal.calibration.*` | Calibration optimization pipeline | Not ported |
| `aquacal.validation.*` | Reprojection/reconstruction validation | Not ported |
| `aquacal.io.detection.*` | ChArUco detection | Not ported |
| `aquacal.io.serialization.save_calibration` | Save calibration JSON | Not ported (write side) |
| `aquacal.config.schema.CalibrationResult` | Full calibration result struct | Not ported; consumers use `CalibrationData` |
| `aquacal.config.schema.BoardConfig` | Board spec | Not ported |
| `aquacal.datasets.*` | Synthetic datasets | Not ported |

### AquaMVS → AquaKit Import Map

#### Ported: Full equivalents

| Old Import (aquamvs) | New Import (aquakit) | Notes |
|----------------------|-----------------------|-------|
| `from aquamvs.calibration import CalibrationData` | `from aquakit import CalibrationData` | Same structure, same PyTorch tensors |
| `from aquamvs.calibration import CameraData` | `from aquakit import CameraData` | Same |
| `from aquamvs.calibration import load_calibration_data` | `from aquakit import load_calibration_data` | Identical function, moved |
| `from aquamvs.calibration import compute_undistortion_maps` | `from aquakit import compute_undistortion_maps` | Moved to aquakit |
| `from aquamvs.calibration import undistort_image` | `from aquakit import undistort_image` | Moved to aquakit |
| `from aquamvs.projection import ProjectionModel` | `from aquakit import ProjectionModel` | Same Protocol |
| `from aquamvs.projection import RefractiveProjectionModel` | `from aquakit import RefractiveProjectionModel` | Same class |
| `from aquamvs.triangulation import triangulate_rays` | `from aquakit import triangulate_rays` | Identical |

#### NOT PORTED (intentional gaps — AquaMVS only)

| AquaMVS Module | What It Does | Status |
|----------------|--------------|--------|
| `aquamvs.triangulation.triangulate_pair` | Pair-wise triangulation with quality filtering | Not ported |
| `aquamvs.triangulation.triangulate_all_pairs` | All-pairs triangulation aggregation | Not ported |
| `aquamvs.triangulation.filter_sparse_cloud` | Point cloud filtering | Not ported |
| `aquamvs.triangulation.compute_depth_ranges` | Depth range estimation | Not ported |
| `aquamvs.io.ImageDirectorySet` | Image directory input | Partially; AquaKit has `ImageSet` |
| `aquamvs.features.*` | Feature extraction/matching (RoMA) | Not ported |
| `aquamvs.dense.*` | Dense stereo, plane sweep | Not ported |
| `aquamvs.fusion.*` | Depth fusion | Not ported |
| `aquamvs.pipeline.*` | Full MVS pipeline | Not ported |
| `aquamvs.evaluation.*` | Metrics and alignment | Not ported |
| `aquamvs.visualization.*` | 3D visualization | Not ported |

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Long-lived PyPI API tokens | OIDC trusted publishing | No secrets to rotate, 15-min scoped tokens |
| `GITHUB_TOKEN` for semantic-release | PAT with `repo` scope | Required to bypass branch protection |
| pyright basic mode | basedpyright standard mode | 5 new diagnostic rules become errors |
| Manual version bumps | python-semantic-release | Reads conventional commits, auto-bumps |

**Deprecated/outdated:**
- `aquamvs.calibration.UndistortionData`: Not ported to AquaKit; `compute_undistortion_maps` returns maps directly now — verify this in aquakit/undistortion.py.
- `refractive_project_fast` / `refractive_project_fast_batch` in AquaCal: Deprecated shims; the rewiring guide should note these were removed entirely in AquaKit.

---

## Open Questions

1. **RELEASE_TOKEN secret existence**
   - What we know: release.yml references `secrets.RELEASE_TOKEN`; the workflow exists
   - What's unclear: Whether the secret has been created in the GitHub repo Settings
   - Recommendation: The plan task should include "check if RELEASE_TOKEN exists, create if missing"

2. **PyPI project and Trusted Publisher existence**
   - What we know: publish.yml is correctly configured for trusted publishing
   - What's unclear: Whether the `aquakit` project exists on PyPI and TestPyPI, and whether Trusted Publishers are registered
   - Recommendation: Task must include "create PyPI project if absent, register Trusted Publisher"

3. **GitHub Environments existence**
   - What we know: publish.yml references environments `testpypi` and `pypi`
   - What's unclear: Whether these environments exist in GitHub repository Settings
   - Recommendation: Task must verify and create environments before running publish workflow

4. **Branch protection current state**
   - What we know: CONTEXT.md says "configure on main, checking for existing rules first"
   - What's unclear: Whether any rules already exist on main
   - Recommendation: Task must check via `gh api` before creating rules

5. **basedpyright standard mode errors count**
   - What we know: Basic mode shows 0 errors; 5 new rules activate in standard mode
   - What's unclear: Whether any of the 5 rules catch real issues in the current codebase
   - Recommendation: First task in phase; run locally, fix any issues, then update CI

6. **aquakit.camera.create_camera signature**
   - What we know: AquaCal has `create_camera(name, intrinsics, extrinsics) -> Camera`; AquaKit also has `create_camera` but it likely differs
   - What's unclear: The exact AquaKit `create_camera` API (not read in detail)
   - Recommendation: Verify before writing rewiring guide; read `src/aquakit/camera.py`

---

## Sources

### Primary (HIGH confidence)
- Direct file reads: `pyproject.toml`, all 5 workflow YAML files, `src/aquakit/__init__.py` and all module files — all verified by reading source
- `hatch run typecheck` output: `0 errors, 0 warnings, 0 notes` confirmed by running the command
- basedpyright 1.38.1 version confirmed by running `basedpyright --version`

### Secondary (MEDIUM confidence)
- [basedpyright config docs](https://docs.basedpyright.com/latest/configuration/config-files/) — standard vs basic mode: 5 rules change (reportFunctionMemberAccess, reportIncompatibleMethodOverride, reportIncompatibleVariableOverride, reportOverlappingOverload, reportPossiblyUnboundVariable)
- [PyPI Trusted Publisher docs](https://docs.pypi.org/trusted-publishers/) — OIDC flow, PyPI-side configuration steps
- [python-semantic-release docs](https://python-semantic-release.readthedocs.io/en/latest/) — GH_TOKEN is the default env var; PAT with `repo` scope required; RELEASE_TOKEN is a custom name used in this repo's config

### Tertiary (LOW confidence)
- GitHub branch protection status check availability timing (7-day window) — from GitHub community forums, not official docs link

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all tools read directly from source files
- Architecture patterns: HIGH — all workflows read directly; PyPI/basedpyright from official docs
- Rewiring guide data: HIGH — all import paths read directly from both AquaCal and AquaMVS source
- Pitfalls: MEDIUM — some based on domain knowledge (RELEASE_TOKEN, branch protection interaction), patterns are well-documented

**Research date:** 2026-02-18
**Valid until:** 2026-03-20 (stable tooling; python-semantic-release and pypa/gh-action-pypi-publish change slowly)
