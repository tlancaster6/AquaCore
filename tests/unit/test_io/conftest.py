"""Shared fixtures for test_io: synthetic video and image data."""

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def two_camera_video_files(tmp_path: Path) -> dict[str, Path]:
    """Create two synthetic video files with 5 frames each.

    Each video has 480x640 frames with a distinctive per-frame color
    (blue channel = frame_idx * 50 in BGR) so pixel values can be
    verified. Falls back from mp4v to MJPG/.avi if the mp4v codec is
    unavailable on the current platform.

    Returns:
        Mapping from camera name (``"cam0"``, ``"cam1"``) to video file path.
    """
    cam_names = ["cam0", "cam1"]
    camera_map: dict[str, Path] = {}
    n_frames = 5
    height, width = 480, 640

    for cam_name in cam_names:
        # Try mp4v first; fall back to MJPG if unavailable.
        fourcc_mp4 = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = tmp_path / f"{cam_name}.mp4"
        writer = cv2.VideoWriter(str(out_path), fourcc_mp4, 30.0, (width, height))

        if not writer.isOpened():
            # Fallback to MJPG + .avi
            writer.release()
            fourcc_mjpg = cv2.VideoWriter_fourcc(*"MJPG")
            out_path = tmp_path / f"{cam_name}.avi"
            writer = cv2.VideoWriter(str(out_path), fourcc_mjpg, 30.0, (width, height))

        for i in range(n_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = i * 50  # Blue channel in BGR
            writer.write(frame)
        writer.release()

        camera_map[cam_name] = out_path

    return camera_map
