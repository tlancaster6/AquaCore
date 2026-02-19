# Phase 3: Calibration and Undistortion - Context

**Gathered:** 2026-02-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Load AquaCal JSON calibration files into typed Python objects and compute undistortion maps for images — all without requiring AquaCal as a dependency. CalibrationData composes existing AquaKit types (CameraIntrinsics, CameraExtrinsics, InterfaceParams). Creating calibrations, refining calibrations, and video/image I/O are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Loader API surface
- `load_calibration_data()` accepts a file path (str | Path) OR a pre-parsed dict — flexible for testing and pipelines
- Returns `CalibrationData` with a single global `water_z` (world-frame property of the rig, not per-camera)
- `CalibrationData.cameras` is `dict[str, CameraData]` keyed by camera name
- Ordered list property for index-based iteration (e.g., `calibration.camera_list`)
- `CameraData.is_auxiliary` flag; `CalibrationData.core_cameras()` and `CalibrationData.auxiliary_cameras()` helper methods
- `CameraData.name` stores the string identifier from the JSON key

### Schema tolerance
- Strict validation on required fields (cameras, interface); silently ignore optional sections (board, diagnostics, metadata)
- Normalize `t` shape from (3,1) to (3,) silently — known AquaCal quirk, not a user error
- Check `version` field; warn on unknown version but attempt to load anyway
- If a camera entry is missing a required field: skip that camera with a warning, don't fail the entire load

### Undistortion pipeline
- `compute_undistortion_maps(camera_data: CameraData)` accepts CameraData object directly
- Returns NumPy `(map_x, map_y)` tuple — maps stay in NumPy since cv2.remap requires it
- No UndistortionData wrapper dataclass — just the map tuple
- `undistort_image()` accepts and returns PyTorch tensors — converts to NumPy internally for cv2.remap
- Fisheye vs pinhole dispatch based on `is_fisheye` flag (same as AquaCal/AquaMVS)

### Type mapping
- `CameraData` composes existing Phase 1 types: `intrinsics: CameraIntrinsics`, `extrinsics: CameraExtrinsics`
- `CalibrationData` stores `interface: InterfaceParams` (reuses Phase 1 type)
- `CameraData` and `CalibrationData` live in `calibration.py` alongside the loader function
- `CameraData.name: str` carries the JSON key for logging and dict reconstruction

### Claude's Discretion
- Exact warning/logging mechanism (Python warnings module vs logging)
- How `camera_list` property orders cameras (insertion order from JSON, alphabetical, etc.)
- Internal conversion details for tensor ↔ NumPy at cv2 boundaries
- Error message wording for validation failures

</decisions>

<specifics>
## Specific Ideas

- water_z is a global rig property in world coordinates — camera height differences come from extrinsics, not per-camera water_z
- Non-auxiliary cameras are called "core cameras" (not "ring cameras" as in AquaMVS)
- Follow AquaMVS's existing `load_calibration_data()` pattern as the reference implementation

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-calibration-and-undistortion*
*Context gathered: 2026-02-18*
