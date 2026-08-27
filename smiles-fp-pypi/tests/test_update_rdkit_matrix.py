"""Tests for update_rdkit_matrix.py."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import update_rdkit_matrix
from packaging.specifiers import SpecifierSet
from packaging.version import Version

if TYPE_CHECKING:
    from pathlib import Path


def mock_response(json_data: object, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_data
    return response


def test_get_cargo_version_reads_the_package_version(tmp_path: Path) -> None:
    cargo_toml = tmp_path / "Cargo.toml"
    cargo_toml.write_text('[package]\nversion = "0.2.0"\n')
    with patch.object(update_rdkit_matrix, "CARGO_TOML", cargo_toml):
        assert update_rdkit_matrix.get_cargo_version() == "0.2.0"


def test_get_requires_python_reads_the_specifier(tmp_path: Path) -> None:
    pyproject_toml = tmp_path / "pyproject.toml"
    pyproject_toml.write_text('[project]\nrequires-python = ">=3.10,<3.15"\n')
    with patch.object(update_rdkit_matrix, "PYPROJECT_TOML", pyproject_toml):
        specifier = update_rdkit_matrix.get_requires_python()

    assert Version("3.10") in specifier
    assert Version("3.14") in specifier
    assert Version("3.15") not in specifier
    assert Version("3.9") not in specifier


def test_fetch_latest_patches_keeps_only_the_newest_per_series_since_2024() -> None:
    releases: dict[str, list[dict[str, object]]] = {
        "2023.9.6": [{"yanked": False}],
        "2024.3.1": [{"yanked": False}],
        "2024.3.6": [{"yanked": False}],
        # older than 2024.3.6 above, so it must not overwrite the already-recorded latest
        "2024.3.2": [{"yanked": False}],
        "2024.9.3": [{"yanked": True}],
        "2025.3.2": [{"yanked": False}],
    }
    with patch(
        "update_rdkit_matrix.requests.get", return_value=mock_response({"releases": releases})
    ):
        result = update_rdkit_matrix.fetch_latest_patches()

    assert result == [Version("2024.3.6"), Version("2025.3.2")]


def test_get_published_rdkit_versions_returns_empty_set_for_unpublished_package() -> None:
    with patch("update_rdkit_matrix.requests.get", return_value=mock_response({}, status_code=404)):
        assert update_rdkit_matrix.get_published_rdkit_versions("0.2.0") == set()


def test_get_published_rdkit_versions_strips_the_cargo_version_prefix() -> None:
    releases: dict[str, list[object]] = {
        "0.2.0.2024.3.6": [],
        "0.2.0.2026.3.5": [],
        "0.1.0.2024.3.6": [],  # different smiles-fp version, must be excluded
        "0.20.0.2024.3.6": [],  # prefix-like but not a real "0.2.0." match, must be excluded
    }
    with patch(
        "update_rdkit_matrix.requests.get", return_value=mock_response({"releases": releases})
    ):
        result = update_rdkit_matrix.get_published_rdkit_versions("0.2.0")

    assert result == {Version("2024.3.6"), Version("2026.3.5")}


def test_compute_release_matrix_unions_latest_and_already_published() -> None:
    with (
        patch("update_rdkit_matrix.get_cargo_version", return_value="0.2.0"),
        patch(
            "update_rdkit_matrix.fetch_latest_patches",
            return_value=[Version("2024.3.6"), Version("2026.3.5")],
        ),
        patch(
            "update_rdkit_matrix.get_published_rdkit_versions",
            return_value={Version("2026.3.4")},
        ) as mock_published,
    ):
        result = update_rdkit_matrix.compute_release_matrix()

    mock_published.assert_called_once_with("0.2.0")
    assert result == [Version("2024.3.6"), Version("2026.3.4"), Version("2026.3.5")]


def test_diff_new_versions_excludes_already_published_versions() -> None:
    with (
        patch("update_rdkit_matrix.get_cargo_version", return_value="0.2.0"),
        patch(
            "update_rdkit_matrix.fetch_latest_patches",
            return_value=[Version("2024.3.6"), Version("2026.3.5")],
        ),
        patch(
            "update_rdkit_matrix.get_published_rdkit_versions",
            return_value={Version("2024.3.6")},
        ),
    ):
        result = update_rdkit_matrix.diff_new_versions()

    assert result == [Version("2026.3.5")]


def test_diff_new_versions_returns_everything_for_a_fresh_cargo_version() -> None:
    with (
        patch("update_rdkit_matrix.get_cargo_version", return_value="0.3.0"),
        patch(
            "update_rdkit_matrix.fetch_latest_patches",
            return_value=[Version("2024.3.6"), Version("2026.3.5")],
        ),
        patch("update_rdkit_matrix.get_published_rdkit_versions", return_value=set()),
    ):
        result = update_rdkit_matrix.diff_new_versions()

    assert result == [Version("2024.3.6"), Version("2026.3.5")]


def test_compute_ci_matrix_excludes_unsupported_python_rdkit_pairs() -> None:
    def fake_get_supported_pythons(rdkit_ver: Version) -> list[Version]:
        if str(rdkit_ver) == "2024.3.6":
            return [Version("3.10"), Version("3.11")]
        # 3.9 is outside requires-python below and must be excluded from the pairs.
        return [Version("3.9"), Version("3.10"), Version("3.14")]

    with (
        patch(
            "update_rdkit_matrix.compute_release_matrix",
            return_value=[Version("2024.3.6"), Version("2026.3.5")],
        ),
        patch("update_rdkit_matrix.get_requires_python", return_value=SpecifierSet(">=3.10,<3.15")),
        patch("update_rdkit_matrix.get_supported_pythons", side_effect=fake_get_supported_pythons),
    ):
        pairs = update_rdkit_matrix.compute_ci_matrix()

    assert pairs == [
        {"python-version": "3.10", "rdkit_version": "2024.3.6"},
        {"python-version": "3.11", "rdkit_version": "2024.3.6"},
        {"python-version": "3.10", "rdkit_version": "2026.3.5"},
        {"python-version": "3.14", "rdkit_version": "2026.3.5"},
    ]


def test_latest_overall_returns_the_max_fetched_version() -> None:
    with patch(
        "update_rdkit_matrix.fetch_latest_patches",
        return_value=[Version("2024.3.6"), Version("2026.3.5"), Version("2025.9.6")],
    ):
        assert update_rdkit_matrix.latest_overall() == Version("2026.3.5")


def test_bump_dev_pin_rewrites_pyproject_toml(tmp_path: Path) -> None:
    pyproject_toml = tmp_path / "pyproject.toml"
    pyproject_toml.write_text('dependencies = ["joblib", "numpy", "rdkit==2026.3.2", "scipy"]\n')
    with patch.object(update_rdkit_matrix, "PYPROJECT_TOML", pyproject_toml):
        changed = update_rdkit_matrix.bump_dev_pin(Version("2026.3.5"))

    assert changed is True
    assert '"rdkit==2026.3.5"' in pyproject_toml.read_text()


def test_bump_dev_pin_returns_false_when_already_at_that_version(tmp_path: Path) -> None:
    pyproject_toml = tmp_path / "pyproject.toml"
    pyproject_toml.write_text('dependencies = ["rdkit==2026.3.5"]\n')
    with patch.object(update_rdkit_matrix, "PYPROJECT_TOML", pyproject_toml):
        changed = update_rdkit_matrix.bump_dev_pin(Version("2026.3.5"))

    assert changed is False


def test_bump_dev_pin_raises_if_the_pin_is_not_found(tmp_path: Path) -> None:
    pyproject_toml = tmp_path / "pyproject.toml"
    pyproject_toml.write_text("nothing to see here\n")
    with (
        patch.object(update_rdkit_matrix, "PYPROJECT_TOML", pyproject_toml),
        pytest.raises(ValueError, match="Could not find the rdkit== pin"),
    ):
        update_rdkit_matrix.bump_dev_pin(Version("2026.3.5"))


def test_release_matrix_prints_the_computed_versions_as_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch(
        "update_rdkit_matrix.compute_release_matrix",
        return_value=[Version("2024.3.6"), Version("2026.3.5")],
    ):
        update_rdkit_matrix.release_matrix()

    assert capsys.readouterr().out.strip() == '["2024.3.6", "2026.3.5"]'


def test_diff_prints_the_new_versions_as_json(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("update_rdkit_matrix.diff_new_versions", return_value=[Version("2026.3.5")]):
        update_rdkit_matrix.diff()

    assert capsys.readouterr().out.strip() == '["2026.3.5"]'


def test_ci_matrix_prints_the_computed_pairs_as_json(capsys: pytest.CaptureFixture[str]) -> None:
    pairs = [{"python-version": "3.12", "rdkit_version": "2026.3.5"}]
    with patch("update_rdkit_matrix.compute_ci_matrix", return_value=pairs):
        update_rdkit_matrix.ci_matrix()

    assert capsys.readouterr().out.strip() == json.dumps(pairs)


def test_bump_pin_reports_changed(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch("update_rdkit_matrix.latest_overall", return_value=Version("2026.3.5")),
        patch("update_rdkit_matrix.bump_dev_pin", return_value=True) as mock_bump,
    ):
        update_rdkit_matrix.bump_pin()

    mock_bump.assert_called_once_with(Version("2026.3.5"))
    assert capsys.readouterr().out.strip() == "changed"


def test_bump_pin_reports_unchanged(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch("update_rdkit_matrix.latest_overall", return_value=Version("2026.3.5")),
        patch("update_rdkit_matrix.bump_dev_pin", return_value=False),
    ):
        update_rdkit_matrix.bump_pin()

    assert capsys.readouterr().out.strip() == "unchanged"


def test_main_registers_all_commands_and_runs_the_app() -> None:
    mock_app = MagicMock()
    with patch("update_rdkit_matrix.doctyper.DocTyper", return_value=mock_app):
        update_rdkit_matrix.main()

    registered = [c.args[0] for c in mock_app.command.return_value.call_args_list]
    assert registered == [
        update_rdkit_matrix.release_matrix,
        update_rdkit_matrix.diff,
        update_rdkit_matrix.ci_matrix,
        update_rdkit_matrix.bump_pin,
    ]
    mock_app.assert_called_once_with()
