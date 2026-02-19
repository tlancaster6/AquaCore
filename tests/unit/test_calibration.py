"""Tests for calibration.py: CameraData, CalibrationData, and load_calibration_data."""

import copy
import json
import warnings

import pytest
import torch

from aquakit import CalibrationData, CameraData, load_calibration_data
from aquakit.types import CameraExtrinsics, CameraIntrinsics, InterfaceParams

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_cal_dict() -> dict:
    """Valid calibration dict matching the AquaCal JSON schema."""
    return {
        "version": "1.0",
        "cameras": {
            "cam0": {
                "intrinsics": {
                    "K": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                    "dist_coeffs": [0.1, -0.2, 0.001, 0.002, 0.03],
                    "image_size": [640, 480],
                },
                "extrinsics": {
                    "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "t": [0.0, 0.0, 0.0],
                },
                "water_z": 0.15,
            },
            "cam1": {
                "intrinsics": {
                    "K": [[600, 0, 320], [0, 600, 240], [0, 0, 1]],
                    "dist_coeffs": [0.05, -0.1, 0.0, 0.0, 0.01],
                    "image_size": [640, 480],
                },
                "extrinsics": {
                    "R": [[0.9, 0.0, 0.436], [0.0, 1.0, 0.0], [-0.436, 0.0, 0.9]],
                    "t": [0.5, 0.0, 0.0],
                },
                "water_z": 0.15,
                "is_auxiliary": True,
            },
        },
        "interface": {"normal": [0, 0, -1], "n_air": 1.0, "n_water": 1.333},
    }


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_load_from_dict(valid_cal_dict: dict) -> None:
    """Loading from a dict returns CalibrationData with 2 camera entries."""
    cal = load_calibration_data(valid_cal_dict)
    assert isinstance(cal, CalibrationData)
    assert len(cal.cameras) == 2
    assert "cam0" in cal.cameras
    assert "cam1" in cal.cameras
    assert isinstance(cal.cameras["cam0"], CameraData)


def test_load_from_file(valid_cal_dict: dict, tmp_path) -> None:
    """Loading from a Path returns the same result as loading from a dict."""
    json_file = tmp_path / "calibration.json"
    json_file.write_text(json.dumps(valid_cal_dict), encoding="utf-8")

    cal = load_calibration_data(json_file)
    assert isinstance(cal, CalibrationData)
    assert len(cal.cameras) == 2


def test_load_from_str_path(valid_cal_dict: dict, tmp_path) -> None:
    """Loading from a str path (not Path object) works correctly."""
    json_file = tmp_path / "calibration.json"
    json_file.write_text(json.dumps(valid_cal_dict), encoding="utf-8")

    cal = load_calibration_data(str(json_file))
    assert isinstance(cal, CalibrationData)
    assert len(cal.cameras) == 2


def test_camera_data_composes_phase1_types(valid_cal_dict: dict) -> None:
    """CameraData.intrinsics is CameraIntrinsics; extrinsics is CameraExtrinsics."""
    cal = load_calibration_data(valid_cal_dict)
    cam = cal.cameras["cam0"]
    assert isinstance(cam.intrinsics, CameraIntrinsics)
    assert isinstance(cam.extrinsics, CameraExtrinsics)


def test_camera_data_name(valid_cal_dict: dict) -> None:
    """CameraData.name matches the JSON key used to look it up."""
    cal = load_calibration_data(valid_cal_dict)
    assert cal.cameras["cam0"].name == "cam0"
    assert cal.cameras["cam1"].name == "cam1"


def test_K_dtype_float32(valid_cal_dict: dict) -> None:
    """K tensor is float32."""
    cal = load_calibration_data(valid_cal_dict)
    assert cal.cameras["cam0"].intrinsics.K.dtype == torch.float32


def test_dist_coeffs_dtype_float64(valid_cal_dict: dict) -> None:
    """dist_coeffs tensor is float64 (required by OpenCV)."""
    cal = load_calibration_data(valid_cal_dict)
    assert cal.cameras["cam0"].intrinsics.dist_coeffs.dtype == torch.float64


def test_R_dtype_float32(valid_cal_dict: dict) -> None:
    """R and t tensors are float32."""
    cal = load_calibration_data(valid_cal_dict)
    cam = cal.cameras["cam0"]
    assert cam.extrinsics.R.dtype == torch.float32
    assert cam.extrinsics.t.dtype == torch.float32


def test_t_shape_normalization(valid_cal_dict: dict) -> None:
    """t provided as (3,1) nested list is silently normalised to shape (3,)."""
    data = copy.deepcopy(valid_cal_dict)
    data["cameras"]["cam0"]["extrinsics"]["t"] = [[0.1], [0.2], [0.3]]

    cal = load_calibration_data(data)
    t = cal.cameras["cam0"].extrinsics.t
    assert t.shape == (3,), f"Expected (3,), got {t.shape}"


def test_interface_params(valid_cal_dict: dict) -> None:
    """interface is an InterfaceParams with the correct normal, n_air, n_water."""
    cal = load_calibration_data(valid_cal_dict)
    assert isinstance(cal.interface, InterfaceParams)
    assert torch.allclose(cal.interface.normal, torch.tensor([0.0, 0.0, -1.0]))
    assert cal.interface.n_air == pytest.approx(1.0)
    assert cal.interface.n_water == pytest.approx(1.333)


def test_water_z_from_first_camera(valid_cal_dict: dict) -> None:
    """interface.water_z equals the first camera's water_z value."""
    cal = load_calibration_data(valid_cal_dict)
    assert cal.interface.water_z == pytest.approx(0.15)


def test_is_fisheye_default_false(valid_cal_dict: dict) -> None:
    """is_fisheye defaults to False when omitted from the JSON."""
    cal = load_calibration_data(valid_cal_dict)
    assert cal.cameras["cam0"].intrinsics.is_fisheye is False


def test_is_fisheye_true(valid_cal_dict: dict) -> None:
    """is_fisheye=True with 4 dist_coeffs is loaded correctly."""
    data = copy.deepcopy(valid_cal_dict)
    data["cameras"]["cam0"]["intrinsics"]["is_fisheye"] = True
    data["cameras"]["cam0"]["intrinsics"]["dist_coeffs"] = [0.01, 0.02, 0.03, 0.04]

    cal = load_calibration_data(data)
    assert cal.cameras["cam0"].intrinsics.is_fisheye is True
    assert cal.cameras["cam0"].intrinsics.dist_coeffs.shape == (4,)


def test_is_auxiliary(valid_cal_dict: dict) -> None:
    """cam1 is auxiliary; cam0 is not."""
    cal = load_calibration_data(valid_cal_dict)
    assert cal.cameras["cam0"].is_auxiliary is False
    assert cal.cameras["cam1"].is_auxiliary is True


def test_core_cameras(valid_cal_dict: dict) -> None:
    """core_cameras() returns only non-auxiliary cameras."""
    cal = load_calibration_data(valid_cal_dict)
    core = cal.core_cameras()
    assert "cam0" in core
    assert "cam1" not in core


def test_auxiliary_cameras(valid_cal_dict: dict) -> None:
    """auxiliary_cameras() returns only auxiliary cameras."""
    cal = load_calibration_data(valid_cal_dict)
    aux = cal.auxiliary_cameras()
    assert "cam1" in aux
    assert "cam0" not in aux


def test_camera_list(valid_cal_dict: dict) -> None:
    """camera_list returns CameraData objects in insertion order."""
    cal = load_calibration_data(valid_cal_dict)
    cam_list = cal.camera_list
    assert len(cam_list) == 2
    assert all(isinstance(c, CameraData) for c in cam_list)
    # Insertion order: cam0 first, cam1 second
    assert cam_list[0].name == "cam0"
    assert cam_list[1].name == "cam1"


def test_backward_compat_interface_distance(valid_cal_dict: dict) -> None:
    """interface_distance is accepted as an alias for water_z."""
    data = copy.deepcopy(valid_cal_dict)
    for cam in data["cameras"].values():
        cam["interface_distance"] = cam.pop("water_z")

    cal = load_calibration_data(data)
    assert cal.interface.water_z == pytest.approx(0.15)


def test_unknown_version_warns(valid_cal_dict: dict) -> None:
    """Unknown version emits UserWarning but load succeeds."""
    data = copy.deepcopy(valid_cal_dict)
    data["version"] = "2.0"

    with pytest.warns(UserWarning, match="Unknown calibration version"):
        cal = load_calibration_data(data)
    assert len(cal.cameras) == 2


def test_missing_camera_field_skips_with_warning(valid_cal_dict: dict) -> None:
    """Camera entry missing 'extrinsics' is skipped with a warning; other cameras load fine."""
    data = copy.deepcopy(valid_cal_dict)
    del data["cameras"]["cam1"]["extrinsics"]

    with pytest.warns(UserWarning, match="cam1"):
        cal = load_calibration_data(data)
    assert "cam0" in cal.cameras
    assert "cam1" not in cal.cameras


def _load_suppressing_warnings(data: dict) -> None:
    """Helper: load calibration while suppressing UserWarnings (for error path tests)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        load_calibration_data(data)


def test_all_cameras_invalid_raises(valid_cal_dict: dict) -> None:
    """All cameras missing required fields raises ValueError."""
    data = copy.deepcopy(valid_cal_dict)
    del data["cameras"]["cam0"]["extrinsics"]
    del data["cameras"]["cam1"]["extrinsics"]

    with pytest.raises(ValueError, match="No valid camera entries"):
        _load_suppressing_warnings(data)


def test_ignores_optional_sections(valid_cal_dict: dict) -> None:
    """Optional top-level sections (board, diagnostics, metadata) are silently ignored."""
    data = copy.deepcopy(valid_cal_dict)
    data["board"] = {"type": "charuco", "squares": [7, 5]}
    data["diagnostics"] = {"reprojection_error": 0.3}
    data["metadata"] = {"captured_by": "lab_session_1"}

    cal = load_calibration_data(data)
    assert len(cal.cameras) == 2


def test_missing_cameras_section_raises() -> None:
    """Missing 'cameras' key in JSON raises ValueError."""
    data = {
        "version": "1.0",
        "interface": {"normal": [0, 0, -1], "n_air": 1.0, "n_water": 1.333},
    }
    with pytest.raises(ValueError, match="cameras"):
        load_calibration_data(data)


def test_missing_interface_section_raises(valid_cal_dict: dict) -> None:
    """Missing 'interface' key in JSON raises ValueError."""
    data = copy.deepcopy(valid_cal_dict)
    del data["interface"]

    with pytest.raises(ValueError, match="interface"):
        load_calibration_data(data)


def test_file_not_found(tmp_path) -> None:
    """Loading from a nonexistent path raises FileNotFoundError."""
    nonexistent = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        load_calibration_data(nonexistent)
