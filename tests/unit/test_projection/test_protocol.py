"""Protocol compliance tests for ProjectionModel (structural subtyping, isinstance).

Tests verify that:
- Real ``RefractiveProjectionModel`` satisfies the protocol (positive).
- A dummy class with both methods satisfies the protocol (positive).
- Classes missing either required method do NOT satisfy the protocol (negative).
"""

from __future__ import annotations

import torch

from aquakit.projection import ProjectionModel, RefractiveProjectionModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_refractive_model() -> RefractiveProjectionModel:
    """Return a valid RefractiveProjectionModel with known-good params."""
    K = torch.tensor(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    R = torch.eye(3, dtype=torch.float32)
    t = torch.zeros(3, dtype=torch.float32)
    normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    return RefractiveProjectionModel(
        K=K,
        R=R,
        t=t,
        water_z=1.0,
        normal=normal,
        n_air=1.0,
        n_water=1.333,
    )


# ---------------------------------------------------------------------------
# Protocol compliance tests (SC-4)
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    """Protocol compliance tests — positive and negative cases."""

    def test_protocol_compliance_refractive_model(self) -> None:
        """Real RefractiveProjectionModel must satisfy ProjectionModel protocol."""
        model = _make_refractive_model()
        assert isinstance(model, ProjectionModel), (
            "RefractiveProjectionModel must satisfy ProjectionModel protocol via "
            "isinstance() runtime check."
        )

    def test_protocol_compliance_dummy_class(self) -> None:
        """A class with project() and back_project() satisfies protocol without import."""

        class _DummyProjectionModel:
            """Dummy model that satisfies ProjectionModel structurally."""

            def project(
                self, points: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                n = points.shape[0]
                return torch.zeros(n, 2), torch.ones(n, dtype=torch.bool)

            def back_project(
                self, pixels: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                n = pixels.shape[0]
                return torch.zeros(n, 3), torch.zeros(n, 3)

        dummy = _DummyProjectionModel()
        assert isinstance(dummy, ProjectionModel), (
            "A class with project() and back_project() must satisfy ProjectionModel "
            "via structural subtyping — no import of ProjectionModel needed."
        )

    def test_protocol_compliance_missing_back_project(self) -> None:
        """A class with only project() must NOT satisfy ProjectionModel."""

        class _OnlyProject:
            """Has project() but not back_project()."""

            def project(
                self, points: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                n = points.shape[0]
                return torch.zeros(n, 2), torch.ones(n, dtype=torch.bool)

        obj = _OnlyProject()
        assert not isinstance(obj, ProjectionModel), (
            "A class missing back_project() must not satisfy ProjectionModel."
        )

    def test_protocol_compliance_missing_project(self) -> None:
        """A class with only back_project() must NOT satisfy ProjectionModel."""

        class _OnlyBackProject:
            """Has back_project() but not project()."""

            def back_project(
                self, pixels: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                n = pixels.shape[0]
                return torch.zeros(n, 3), torch.zeros(n, 3)

        obj = _OnlyBackProject()
        assert not isinstance(obj, ProjectionModel), (
            "A class missing project() must not satisfy ProjectionModel."
        )
