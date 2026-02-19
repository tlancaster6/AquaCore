# Phase 5: Packaging and Release - Context

**Gathered:** 2026-02-18
**Status:** Ready for planning

<domain>
## Phase Boundary

The library installs from PyPI, CI enforces quality on every push, and consumer teams have an import migration guide. Workflows and pyproject.toml already exist — this phase validates, fixes, and completes them.

</domain>

<decisions>
## Implementation Decisions

### CI pipeline
- Existing workflows (test.yml, publish.yml, release.yml, slow-tests.yml, docs.yml) already cover the target setup
- Phase 5 validates these workflows work end-to-end with current code and fixes any gaps
- Target matrix: Ubuntu + Windows, Python 3.11/3.12/3.13 (already configured)
- Separate jobs for test, typecheck, and pre-commit (already configured)
- No GPU CI — CUDA testing stays manual/local
- Slow tests remain manual dispatch only

### PyPI publishing
- Package name: `aquacore` (already configured in pyproject.toml)
- Versioning: SemVer starting at 0.1.0 (already configured with python-semantic-release)
- Publishing: Tag-triggered via trusted publishing (already configured)
- TestPyPI step before real PyPI (already configured in publish.yml)
- PyTorch is intentionally NOT a declared dependency — users install their own variant

### Rewiring guide
- Structured by consumer: separate sections for AquaCal users and AquaMVS users
- Lives in `.planning/rewiring/` — it's a dev doc, not shipped with the package
- Covers ported imports AND flags gaps (modules still in AquaCal/AquaMVS that haven't been ported)
- Depth of examples: Claude's discretion based on actual API differences

### Quality gates
- No coverage threshold — coverage is tracked (Codecov) but informational only
- Typecheck failures block merge — basedpyright is a required status check
- Branch protection rules: configure on main, checking for existing rules first
- basedpyright strictness: bump from "basic" to "standard" — may require type annotation fixes

### Claude's Discretion
- CI job structure (whether to split/merge existing jobs) — keep what works
- Level of detail in rewiring guide usage examples
- How to handle any type annotation fixes needed for "standard" strictness
- Branch protection rule specifics (required reviewers, etc.)

</decisions>

<specifics>
## Specific Ideas

- Existing workflows are the starting point — validate and fix, not rewrite
- Rewiring guide should help teams plan their migration by showing what's available and what's still missing
- pyproject.toml already well-configured with semantic-release, hatch build, classifiers, etc.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-packaging-and-release*
*Context gathered: 2026-02-18*
