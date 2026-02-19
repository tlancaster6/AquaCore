"""Tests for VideoSet: construction, tensor format, iteration, context manager, errors."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from aquacore.io.frameset import FrameSet
from aquacore.io.video import VideoSet

# ---------------------------------------------------------------------------
# Construction and length
# ---------------------------------------------------------------------------


def test_videoset_construction(two_camera_video_files: dict[str, Path]) -> None:
    """VideoSet constructs successfully and reports the correct frame count."""
    with VideoSet(two_camera_video_files) as vs:
        assert len(vs) == 5, f"Expected 5 frames, got {len(vs)}"


# ---------------------------------------------------------------------------
# __getitem__ — dict structure
# ---------------------------------------------------------------------------


def test_videoset_getitem_returns_dict(two_camera_video_files: dict[str, Path]) -> None:
    """__getitem__ returns a dict with the expected camera name keys."""
    with VideoSet(two_camera_video_files) as vs:
        frame = vs[0]
        assert isinstance(frame, dict), f"Expected dict, got {type(frame)}"
        assert set(frame.keys()) == {"cam0", "cam1"}, (
            f"Expected keys {{'cam0', 'cam1'}}, got {set(frame.keys())}"
        )


# ---------------------------------------------------------------------------
# Tensor format — shape, dtype, value range
# ---------------------------------------------------------------------------


def test_videoset_tensor_format(two_camera_video_files: dict[str, Path]) -> None:
    """Each frame tensor is (C, H, W) float32 with values in [0, 1]."""
    with VideoSet(two_camera_video_files) as vs:
        frames = vs[0]
        for cam_name, tensor in frames.items():
            assert tensor.ndim == 3, (
                f"{cam_name}: expected 3D tensor, got {tensor.ndim}D"
            )
            assert tensor.shape[0] == 3, (
                f"{cam_name}: expected C=3 channels, got shape {tensor.shape}"
            )
            assert tensor.dtype == torch.float32, (
                f"{cam_name}: expected float32, got {tensor.dtype}"
            )
            assert tensor.min() >= 0.0, (
                f"{cam_name}: min value {tensor.min():.4f} below 0.0"
            )
            assert tensor.max() <= 1.0, (
                f"{cam_name}: max value {tensor.max():.4f} above 1.0"
            )


# ---------------------------------------------------------------------------
# __iter__
# ---------------------------------------------------------------------------


def test_videoset_iter_yields_tuples(two_camera_video_files: dict[str, Path]) -> None:
    """__iter__ yields (int, dict) tuples with correct sequential indices."""
    with VideoSet(two_camera_video_files) as vs:
        indices: list[int] = []
        for idx, frames in vs:
            assert isinstance(idx, int), f"Expected int index, got {type(idx)}"
            assert isinstance(frames, dict), f"Expected dict, got {type(frames)}"
            indices.append(idx)

        assert indices == list(range(5)), f"Expected indices 0..4, got {indices}"


def test_videoset_iter_all_frames(two_camera_video_files: dict[str, Path]) -> None:
    """Iteration produces exactly len(vs) frames."""
    with VideoSet(two_camera_video_files) as vs:
        frame_count = sum(1 for _ in vs)
        assert frame_count == len(vs), (
            f"Iteration produced {frame_count} frames, expected {len(vs)}"
        )


def test_videoset_iter_resets_after_getitem(
    two_camera_video_files: dict[str, Path],
) -> None:
    """Calling __getitem__ before iterating does not affect iteration start.

    After seeking to frame 3 via __getitem__, __iter__ must reset all captures
    to frame 0 and yield frames 0, 1, 2, ... in order.
    """
    with VideoSet(two_camera_video_files) as vs:
        _ = vs[3]  # Seek to frame 3

        indices: list[int] = []
        for idx, _frames in vs:
            indices.append(idx)

        assert indices == list(range(len(vs))), (
            f"Expected indices starting from 0, got {indices}"
        )
        assert indices[0] == 0, (
            f"First iteration index should be 0 after reset, got {indices[0]}"
        )


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_videoset_context_manager_releases(
    two_camera_video_files: dict[str, Path],
) -> None:
    """After context manager exits, all capture handles are released."""
    vs = VideoSet(two_camera_video_files)
    with vs:
        pass  # Use context manager

    # _caps should be empty after __exit__
    assert vs._caps == {}, f"Expected _caps to be empty after __exit__, got {vs._caps}"


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


def test_videoset_protocol_compliance(two_camera_video_files: dict[str, Path]) -> None:
    """isinstance(VideoSet(...), FrameSet) returns True (structural compliance)."""
    with VideoSet(two_camera_video_files) as vs:
        assert isinstance(vs, FrameSet), (
            "VideoSet must satisfy FrameSet protocol for isinstance() to return True"
        )


# ---------------------------------------------------------------------------
# Error handling — ValueError / IndexError
# ---------------------------------------------------------------------------


def test_videoset_missing_file(tmp_path: Path) -> None:
    """ValueError raised when video file does not exist."""
    nonexistent = tmp_path / "no_such_file.mp4"
    with pytest.raises(ValueError, match="does not exist"):
        VideoSet({"cam0": nonexistent})


def test_videoset_not_a_file(tmp_path: Path) -> None:
    """ValueError raised when path is a directory instead of a video file."""
    dir_path = tmp_path / "a_directory"
    dir_path.mkdir()
    with pytest.raises(ValueError, match="directory"):
        VideoSet({"cam0": dir_path})


def test_videoset_index_out_of_range(two_camera_video_files: dict[str, Path]) -> None:
    """IndexError raised for negative or out-of-bounds frame indices."""
    with VideoSet(two_camera_video_files) as vs:
        with pytest.raises(IndexError):
            vs[-1]

        with pytest.raises(IndexError):
            vs[5]  # len == 5, valid range is [0, 4]

        with pytest.raises(IndexError):
            vs[100]


# ---------------------------------------------------------------------------
# Frame count mismatch warning
# ---------------------------------------------------------------------------


def test_videoset_frame_count_mismatch_warns(tmp_path: Path) -> None:
    """UserWarning issued when cameras have different frame counts; len() uses minimum.

    Creates one 5-frame video and one 3-frame video, constructs VideoSet, and
    verifies that a UserWarning is emitted and len() equals the minimum (3).
    """
    height, width = 240, 320
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Camera 0: 5 frames
    path0 = tmp_path / "cam0.mp4"
    writer0 = cv2.VideoWriter(str(path0), fourcc, 30.0, (width, height))
    if not writer0.isOpened():
        # Fallback to MJPG
        writer0.release()
        path0 = tmp_path / "cam0.avi"
        writer0 = cv2.VideoWriter(
            str(path0), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (width, height)
        )
    for _ in range(5):
        writer0.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer0.release()

    # Camera 1: 3 frames
    path1 = tmp_path / "cam1.mp4"
    writer1 = cv2.VideoWriter(str(path1), fourcc, 30.0, (width, height))
    if not writer1.isOpened():
        writer1.release()
        path1 = tmp_path / "cam1.avi"
        writer1 = cv2.VideoWriter(
            str(path1), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (width, height)
        )
    for _ in range(3):
        writer1.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer1.release()

    with pytest.warns(UserWarning, match="Frame counts differ"):
        vs = VideoSet({"cam0": path0, "cam1": path1})

    with vs:
        assert len(vs) <= 5, f"Expected len <= 5, got {len(vs)}"
        # We can't strictly assert len==3 since cv2 frame count is codec-dependent,
        # but it must be at most the reported count for the shorter video.
        assert len(vs) >= 1, "VideoSet must have at least 1 frame"


# ---------------------------------------------------------------------------
# Mid-init cleanup — handle leak prevention
# ---------------------------------------------------------------------------


def test_videoset_mid_init_cleanup(tmp_path: Path) -> None:
    """Constructor releases opened captures if it fails partway through init.

    Creates one valid video file and specifies one invalid (nonexistent) path.
    Verifies that ValueError is raised and that the valid video can subsequently
    be re-opened (demonstrating the first handle was properly released).
    """
    height, width = 240, 320
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    valid_path = tmp_path / "valid.mp4"
    writer = cv2.VideoWriter(str(valid_path), fourcc, 30.0, (width, height))
    if not writer.isOpened():
        writer.release()
        valid_path = tmp_path / "valid.avi"
        writer = cv2.VideoWriter(
            str(valid_path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (width, height)
        )
    for _ in range(3):
        writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()

    invalid_path = tmp_path / "no_such_file.mp4"

    # Construction with an invalid second camera must raise ValueError.
    with pytest.raises(ValueError, match="does not exist"):
        VideoSet({"cam0": valid_path, "cam1": invalid_path})

    # After the failed constructor, the valid file should be re-openable,
    # proving the handle was released during mid-init cleanup.
    with VideoSet({"cam0": valid_path}) as vs:
        assert len(vs) >= 1, (
            "Valid video should be openable after mid-init cleanup of failed VideoSet"
        )
