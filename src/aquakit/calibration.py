"""Calibration data loading from AquaCal JSON format."""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

from .types import CameraExtrinsics, CameraIntrinsics, InterfaceParams

# Known supported version; warn on others but still attempt load
_KNOWN_VERSION = "1.0"


@dataclass
class CameraData:
    """Typed representation of a single camera from a calibration file.

    Attributes:
        name: Camera identifier (the JSON key for this camera entry).
        intrinsics: Intrinsic camera parameters (K, dist_coeffs, image_size, is_fisheye).
        extrinsics: Extrinsic camera parameters (R, t) in world-to-camera convention.
        is_auxiliary: True if this camera is marked as auxiliary in the calibration.
    """

    name: str
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    is_auxiliary: bool = False


@dataclass
class CalibrationData:
    """Typed representation of a full AquaCal calibration file.

    Attributes:
        cameras: All cameras keyed by name, in JSON insertion order.
        interface: Refractive interface parameters including normal, water_z,
            n_air, and n_water.
    """

    cameras: dict[str, CameraData]
    interface: InterfaceParams

    @property
    def camera_list(self) -> list[CameraData]:
        """All cameras as an ordered list (insertion order from JSON).

        Returns:
            List of CameraData in the order they appear in the source JSON.
        """
        return list(self.cameras.values())

    def core_cameras(self) -> dict[str, CameraData]:
        """Return only non-auxiliary cameras.

        Returns:
            Subset of cameras dict where is_auxiliary is False.
        """
        return {k: v for k, v in self.cameras.items() if not v.is_auxiliary}

    def auxiliary_cameras(self) -> dict[str, CameraData]:
        """Return only auxiliary cameras.

        Returns:
            Subset of cameras dict where is_auxiliary is True.
        """
        return {k: v for k, v in self.cameras.items() if v.is_auxiliary}


def _parse_intrinsics(cam_name: str, intr_dict: dict) -> CameraIntrinsics:
    """Parse the intrinsics sub-dict into CameraIntrinsics.

    Args:
        cam_name: Camera name (used in error messages).
        intr_dict: Raw dict with K, dist_coeffs, image_size, optionally is_fisheye.

    Returns:
        CameraIntrinsics with K as float32, dist_coeffs as float64.

    Raises:
        KeyError: If a required field is missing.
    """
    K = torch.tensor(intr_dict["K"], dtype=torch.float32)
    dist_coeffs = torch.tensor(intr_dict["dist_coeffs"], dtype=torch.float64)
    w, h = intr_dict["image_size"]
    image_size = (int(w), int(h))
    is_fisheye = bool(intr_dict.get("is_fisheye", False))
    return CameraIntrinsics(
        K=K, dist_coeffs=dist_coeffs, image_size=image_size, is_fisheye=is_fisheye
    )


def _parse_extrinsics(cam_name: str, extr_dict: dict) -> CameraExtrinsics:
    """Parse the extrinsics sub-dict into CameraExtrinsics.

    Normalises t from shape (3, 1) to (3,) silently (known AquaCal quirk).

    Args:
        cam_name: Camera name (used in error messages).
        extr_dict: Raw dict with R and t.

    Returns:
        CameraExtrinsics with R as float32 (3, 3) and t as float32 (3,).

    Raises:
        KeyError: If a required field is missing.
    """
    R = torch.tensor(extr_dict["R"], dtype=torch.float32)
    t = torch.tensor(extr_dict["t"], dtype=torch.float32)
    # Normalise (3, 1) → (3,)
    if t.ndim == 2 and t.shape == (3, 1):
        t = t.squeeze(1)
    return CameraExtrinsics(R=R, t=t)


def _parse_camera(name: str, entry: dict) -> CameraData:
    """Parse one camera JSON entry into CameraData.

    Args:
        name: The JSON key identifying this camera.
        entry: Raw dict for the camera entry.

    Returns:
        CameraData composed of Phase 1 types.

    Raises:
        KeyError: If any required sub-key is missing.
        ValueError: If data shapes are invalid.
    """
    intrinsics = _parse_intrinsics(name, entry["intrinsics"])
    extrinsics = _parse_extrinsics(name, entry["extrinsics"])
    # Require water_z / backward compat alias interface_distance
    if "water_z" not in entry and "interface_distance" not in entry:
        raise KeyError("water_z")
    is_auxiliary = bool(entry.get("is_auxiliary", False))
    return CameraData(
        name=name,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        is_auxiliary=is_auxiliary,
    )


def _extract_water_z(entry: dict) -> float:
    """Extract water_z from a raw camera entry (supports backward compat).

    Args:
        entry: Raw camera dict that contains water_z or interface_distance.

    Returns:
        water_z as float.
    """
    if "water_z" in entry:
        return float(entry["water_z"])
    return float(entry["interface_distance"])


def _parse_interface(iface_dict: dict, water_z: float) -> InterfaceParams:
    """Parse the interface section into InterfaceParams.

    Args:
        iface_dict: Raw interface dict with normal, n_air, n_water.
        water_z: Water surface Z-coordinate extracted from camera entries.

    Returns:
        InterfaceParams with normal as float32 tensor.
    """
    normal = torch.tensor(iface_dict["normal"], dtype=torch.float32)
    n_air = float(iface_dict.get("n_air", 1.0))
    n_water = float(iface_dict.get("n_water", 1.333))
    return InterfaceParams(normal=normal, water_z=water_z, n_air=n_air, n_water=n_water)


def load_calibration_data(source: str | Path | dict) -> CalibrationData:
    """Load a calibration file and return a typed CalibrationData object.

    Accepts either a file path (str or Path) or a pre-parsed dict that mirrors
    the AquaCal JSON schema.  No AquaCal dependency is required.

    Behaviour:
    - Unknown ``version`` values produce a :class:`UserWarning` but load proceeds.
    - Camera entries with missing required fields are skipped with a
      :class:`UserWarning` rather than raising; all other cameras still load.
    - If *all* camera entries are invalid a :class:`ValueError` is raised.
    - Optional top-level sections (``board``, ``diagnostics``, ``metadata``) are
      silently ignored.
    - ``t`` vectors with shape ``(3, 1)`` are silently normalised to ``(3,)``.
    - ``interface_distance`` is accepted as a backward-compatible alias for
      ``water_z`` in camera entries.

    Args:
        source: File path (``str`` or :class:`~pathlib.Path`) or pre-parsed
            ``dict`` matching the AquaCal JSON schema.

    Returns:
        CalibrationData containing typed camera data and interface parameters.

    Raises:
        FileNotFoundError: If ``source`` is a path that does not exist.
        ValueError: If the JSON has no ``cameras`` section, no ``interface``
            section, or every camera entry is invalid.
    """
    if isinstance(source, dict):
        raw: dict = source
    else:
        path = Path(source)
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

    # Version check
    version = raw.get("version")
    if version is not None and version != _KNOWN_VERSION:
        warnings.warn(
            f"Unknown calibration version {version!r}; expected {_KNOWN_VERSION!r}. "
            "Attempting to load anyway.",
            UserWarning,
            stacklevel=2,
        )

    # Validate required top-level sections
    if "cameras" not in raw:
        raise ValueError("Calibration JSON missing required 'cameras' section.")
    if "interface" not in raw:
        raise ValueError("Calibration JSON missing required 'interface' section.")

    cameras_raw: dict = raw["cameras"]
    cameras: dict[str, CameraData] = {}
    first_water_z: float | None = None

    for cam_name, cam_entry in cameras_raw.items():
        try:
            cam_data = _parse_camera(cam_name, cam_entry)
            water_z = _extract_water_z(cam_entry)
            if first_water_z is None:
                first_water_z = water_z
            cameras[cam_name] = cam_data
        except (KeyError, TypeError, ValueError) as exc:
            warnings.warn(
                f"Camera {cam_name!r} skipped due to missing or invalid field: {exc}. "
                "Continuing with remaining cameras.",
                UserWarning,
                stacklevel=2,
            )

    if not cameras:
        raise ValueError(
            "No valid camera entries found in calibration data. "
            "All camera entries were skipped due to missing required fields."
        )

    # Fallback water_z if somehow none was found (should not happen after cameras check)
    water_z_final: float = first_water_z if first_water_z is not None else 0.0

    interface = _parse_interface(raw["interface"], water_z_final)

    return CalibrationData(cameras=cameras, interface=interface)
