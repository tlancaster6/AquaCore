# Phase 4: I/O Layer - Context

**Gathered:** 2026-02-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Synchronized multi-camera frame access from video files and image directories via a common FrameSet protocol. Users construct a VideoSet or ImageSet with explicit camera-to-path mappings and iterate or index into synchronized frames returned as PyTorch tensors. A factory function auto-detects input type. Creating calibration data, undistortion, or projection are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Frame tensor format
- Layout: **(C, H, W) float32** — PyTorch convention, works natively with torchvision and conv layers
- Color order: **RGB** — BGR-to-RGB conversion happens inside the I/O layer; consumers get standard RGB
- Value range: Claude's discretion (see below)
- Output type: **PyTorch tensors only** — no NumPy return option; consumers call `.numpy()` if needed
- Independent copies: tensors must be `.clone()`d from OpenCV buffers to prevent silent overwrite on next read

### Access & iteration API
- FrameSet protocol defines **`__getitem__`**, **`__len__`**, and **`__iter__`**
- `__getitem__(idx)` returns `dict[str, Tensor]` — camera name keys, (C, H, W) float32 tensor values
- `__len__` returns total frame count
- `__iter__` yields `(frame_idx, dict[str, Tensor])` tuples — sequential, frame-exact
- **VideoSet caveat:** `__getitem__` uses cv2 seek (approximate for compressed video — fine for locating window starts); `__iter__` reads sequentially and is frame-exact
- ImageSet: all access is exact (file-based)
- FrameSet protocol **requires context manager** (`__enter__`/`__exit__`); ImageSet is a no-op, VideoSet releases cv2.VideoCapture handles

### Camera-to-path mapping
- Constructor takes **`dict[str, str | Path]`** — explicit camera name to file/directory path
- Internally converts all paths to `Path` objects
- One video file per camera (no multi-camera-in-one-file support)
- **Factory function** `create_frameset(camera_map)` auto-detects images vs video from paths and returns the appropriate concrete class
- No CalibrationData coupling — camera-to-path mapping is established by consumer repos (AquaCal init, AquaMVS init), not AquaCore

### Frame mismatch & errors
- **Mismatched frame counts:** warn and use minimum count (not ValueError like AquaMVS)
- **Corrupt/unreadable frames:** warn + omit that camera from the returned dict (AquaMVS pattern); consumer checks if expected cameras are present
- **ImageSet filename matching:** require matching filenames across all camera dirs (sorted order), raise ValueError on mismatch — catches data alignment issues early
- **Missing directories/files:** raise ValueError at init (same as AquaMVS)

### Claude's Discretion
- Float32 value range ([0, 1] vs [0, 255]) — pick based on typical consumer patterns
- Exact image extensions supported (png, jpg, tiff, etc.)
- Logging verbosity and format
- Internal buffering strategy for VideoSet sequential reads

</decisions>

<specifics>
## Specific Ideas

- VideoSet `__getitem__` is approximate (keyframe-based seeking) — acceptable for use cases like locating temporal-median window starts. Sequential `__iter__` is frame-exact for precision-critical workflows like AquaPose.
- AquaMVS `ImageDirectorySet` is the primary reference implementation (same pattern: dict-based, sorted filenames, cv2.imread)
- `detect_input_type()` from AquaMVS is the reference for the factory function logic (checks dirs vs files, falls back to extension detection)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-i-o-layer*
*Context gathered: 2026-02-18*
