"""Tests for dev_rdkit_version.py."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import dev_rdkit_version
import pytest

if TYPE_CHECKING:
    from pathlib import Path


def test_get_dev_rdkit_version_extracts_the_pinned_version(tmp_path: Path) -> None:
    pyproject_toml = tmp_path / "pyproject.toml"
    pyproject_toml.write_text('dependencies = ["joblib", "rdkit==2026.3.5", "scipy"]\n')
    with patch.object(dev_rdkit_version, "PYPROJECT_TOML", pyproject_toml):
        assert dev_rdkit_version.get_dev_rdkit_version() == "2026.3.5"


def test_get_dev_rdkit_version_raises_if_the_pin_is_missing(tmp_path: Path) -> None:
    pyproject_toml = tmp_path / "pyproject.toml"
    pyproject_toml.write_text("nothing to see here\n")
    with (
        patch.object(dev_rdkit_version, "PYPROJECT_TOML", pyproject_toml),
        pytest.raises(ValueError, match="Could not find a pinned rdkit=="),
    ):
        dev_rdkit_version.get_dev_rdkit_version()


def test_main_prints_the_version_and_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("dev_rdkit_version.get_dev_rdkit_version", return_value="2026.3.5"):
        assert dev_rdkit_version.main() == 0

    assert capsys.readouterr().out.strip() == "2026.3.5"


def test_main_prints_the_error_to_stderr_and_returns_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("dev_rdkit_version.get_dev_rdkit_version", side_effect=ValueError("no pin found")):
        assert dev_rdkit_version.main() == 1

    assert "no pin found" in capsys.readouterr().err
