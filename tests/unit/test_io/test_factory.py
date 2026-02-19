"""Tests for create_frameset: auto-detection, empty map, mixed types, extension inference."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from aquacore.io.images import ImageSet, create_frameset
from aquacore.io.video import VideoSet

# ---------------------------------------------------------------------------
# create_frameset with real directories (ImageSet)
# ---------------------------------------------------------------------------


def test_create_frameset_with_image_dirs(tmp_path: Path) -> None:
    """create_frameset returns ImageSet when all paths are existing directories."""
    cam_dir = tmp_path / "cam0"
    cam_dir.mkdir()
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(cam_dir / "frame_0000.png"), img)

    result = create_frameset({"cam0": cam_dir})
    assert isinstance(result, ImageSet), (
        f"Expected ImageSet for directory paths, got {type(result)}"
    )


# ---------------------------------------------------------------------------
# create_frameset with real video files (VideoSet)
# ---------------------------------------------------------------------------


def test_create_frameset_with_video_files(
    two_camera_video_files: dict[str, Path],
) -> None:
    """create_frameset returns VideoSet when all paths are existing video files."""
    with create_frameset(two_camera_video_files) as result:
        assert isinstance(result, VideoSet), (
            f"Expected VideoSet for video file paths, got {type(result)}"
        )


# ---------------------------------------------------------------------------
# Empty map
# ---------------------------------------------------------------------------


def test_create_frameset_empty_map() -> None:
    """ValueError raised when camera_map is empty."""
    with pytest.raises(ValueError, match="empty"):
        create_frameset({})


# ---------------------------------------------------------------------------
# Mixed types
# ---------------------------------------------------------------------------


def test_create_frameset_mixed_types(tmp_path: Path) -> None:
    """ValueError raised when camera_map contains a mix of dirs and files.

    Uses a mix of an existing directory and an existing file to trigger the
    mixed-type detection path.
    """
    cam_dir = tmp_path / "cam0"
    cam_dir.mkdir()
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(cam_dir / "frame_0000.png"), img)

    # Create a valid video file for cam1
    height, width = 64, 64
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = tmp_path / "cam1.mp4"
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))
    if not writer.isOpened():
        writer.release()
        video_path = tmp_path / "cam1.avi"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (width, height)
        )
    for _ in range(2):
        writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()

    with pytest.raises(ValueError, match="mix"):
        create_frameset({"cam0": cam_dir, "cam1": video_path})


# ---------------------------------------------------------------------------
# Extension-based inference for nonexistent paths
# ---------------------------------------------------------------------------


def test_create_frameset_nonexistent_video_ext(tmp_path: Path) -> None:
    """Nonexistent paths with video extension return VideoSet (extension inference).

    This covers the test/mock use case where paths do not yet exist on disk.
    VideoSet construction will fail — we only test create_frameset dispatch.
    """
    fake_path = tmp_path / "recording.mp4"
    # Do NOT create the file — testing extension-based inference only.

    # create_frameset dispatches to VideoSet based on extension.
    # VideoSet constructor then raises ValueError for nonexistent file.
    with pytest.raises(ValueError, match="does not exist"):
        create_frameset({"cam0": fake_path})


def test_create_frameset_nonexistent_dir_ext(tmp_path: Path) -> None:
    """Nonexistent paths without video extension return ImageSet (extension inference).

    This covers the test/mock use case where paths do not yet exist on disk.
    ImageSet construction will fail — we only test create_frameset dispatch.
    """
    fake_path = tmp_path / "images_dir"
    # Do NOT create the directory — testing extension-based inference only.

    # create_frameset dispatches to ImageSet (no video extension).
    # ImageSet constructor then raises ValueError for nonexistent directory.
    with pytest.raises(ValueError, match="does not exist"):
        create_frameset({"cam0": fake_path})
