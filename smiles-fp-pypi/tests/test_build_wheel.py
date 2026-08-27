"""Tests for build_wheel.py."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import build_wheel
import pytest
import requests
from packaging.version import Version


def test_git_cmd_runs_git_and_returns_stripped_output() -> None:
    with patch("build_wheel.subprocess.check_output", return_value="deadbeef\n") as mock_output:
        assert build_wheel.git_cmd("rev-parse", "HEAD") == "deadbeef"
    mock_output.assert_called_once_with(
        ["git", "rev-parse", "HEAD"], cwd=build_wheel.REPO_ROOT, text=True
    )

    with patch("build_wheel.subprocess.check_output", return_value="ok\n") as mock_output:
        build_wheel.git_cmd("status", use_cwd=True)
    mock_output.assert_called_once_with(["git", "status"], cwd=None, text=True)


def test_get_cargo_version_raises_if_cargo_toml_is_missing(tmp_path: Path) -> None:
    with (
        patch.object(build_wheel, "CARGO_TOML", tmp_path / "missing.toml"),
        pytest.raises(FileNotFoundError, match="Missing"),
    ):
        build_wheel.get_cargo_version()


def test_get_cargo_version_raises_if_version_key_is_missing(tmp_path: Path) -> None:
    cargo_toml = tmp_path / "Cargo.toml"
    cargo_toml.write_text('[package]\nname = "smiles_fp"\n')
    with (
        patch.object(build_wheel, "CARGO_TOML", cargo_toml),
        pytest.raises(ValueError, match=r"\[package\] -> 'version'"),
    ):
        build_wheel.get_cargo_version()


def test_get_cargo_version_reads_the_package_version(tmp_path: Path) -> None:
    cargo_toml = tmp_path / "Cargo.toml"
    cargo_toml.write_text('[package]\nname = "smiles_fp"\nversion = "0.2.0"\n')
    with patch.object(build_wheel, "CARGO_TOML", cargo_toml):
        assert build_wheel.get_cargo_version() == Version("0.2.0")


def mock_response(json_data: object) -> MagicMock:
    response = MagicMock()
    response.json.return_value = json_data
    return response


def test_get_supported_pythons_parses_cpython_wheel_tags() -> None:
    urls = [
        {"packagetype": "sdist", "python_version": "source"},
        {"packagetype": "bdist_wheel", "python_version": "cp310"},
        {"packagetype": "bdist_wheel", "python_version": "cp312"},
        {"packagetype": "bdist_wheel", "python_version": "py3"},
    ]
    with patch("build_wheel.requests.get", return_value=mock_response({"urls": urls})):
        assert build_wheel.get_supported_pythons(Version("2026.3.2")) == [
            Version("3.10"),
            Version("3.12"),
        ]


def test_get_supported_pythons_raises_on_request_failure() -> None:
    with (
        patch("build_wheel.requests.get", side_effect=requests.RequestException("boom")),
        pytest.raises(RuntimeError, match="Failed to fetch RDKit"),
    ):
        build_wheel.get_supported_pythons(Version("2026.3.2"))


def test_sync_worktree_reuses_a_dirty_snapshot_and_resets_an_existing_worktree(
    tmp_path: Path,
) -> None:
    worktree_root = tmp_path / "worktrees"
    worktree_dir = worktree_root / "rdkit-2026.3.2"
    worktree_dir.mkdir(parents=True)

    with (
        patch.object(build_wheel, "WORKTREE_ROOT", worktree_root),
        patch("build_wheel.git_cmd", return_value="deadbeef") as mock_git,
    ):
        result = build_wheel.sync_worktree(Version("2026.3.2"))

    assert result == worktree_dir
    mock_git.assert_any_call("stash", "create")
    mock_git.assert_any_call("-C", str(worktree_dir), "reset", "--hard", "deadbeef", use_cwd=True)


def test_sync_worktree_falls_back_to_head_and_creates_a_new_worktree(tmp_path: Path) -> None:
    worktree_root = tmp_path / "worktrees"

    def fake_git_cmd(*args: str, **unused_kwargs: object) -> str:
        return "" if args == ("stash", "create") else "cafebabe"

    with (
        patch.object(build_wheel, "WORKTREE_ROOT", worktree_root),
        patch("build_wheel.git_cmd", side_effect=fake_git_cmd) as mock_git,
    ):
        result = build_wheel.sync_worktree(Version("2026.3.2"))

    assert result == worktree_root / "rdkit-2026.3.2"
    assert worktree_root.is_dir()
    mock_git.assert_any_call("worktree", "add", "--detach", str(result), "cafebabe")


def test_patch_pyproject_pins_version_python_range_and_pinned_rdkit(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        '[project]\nversion = "0.0.0"\nrequires-python = ">=3.10"\n'
        'dependencies = ["joblib", "rdkit==2026.3.2", "scipy"]\n'
    )

    build_wheel.patch_pyproject(path, "0.2.0.2026.3.5", ">=3.10,<=3.13", Version("2026.3.5"))

    content = path.read_text()
    assert 'version = "0.2.0.2026.3.5"' in content
    assert 'requires-python = ">=3.10,<=3.13"' in content
    assert '"rdkit==2026.3.5"' in content


def test_patch_pyproject_pins_an_unpinned_rdkit_dependency(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text('[project]\nversion = "0.0.0"\ndependencies = ["rdkit"]\n')

    build_wheel.patch_pyproject(path, "0.2.0.2026.3.5", ">=3.10,<=3.13", Version("2026.3.5"))

    assert '"rdkit==2026.3.5"' in path.read_text()


def test_repair_linux_wheel_invokes_auditwheel_with_excludes(tmp_path: Path) -> None:
    wheel_path = tmp_path / "smiles_fp-0.2.0-cp312-linux.whl"
    with (
        patch.object(build_wheel, "WHEEL_DIR", tmp_path / "wheels"),
        patch("build_wheel.subprocess.check_call") as mock_call,
    ):
        build_wheel.repair_linux_wheel(wheel_path)

    mock_call.assert_called_once_with(
        [
            "auditwheel",
            "repair",
            str(wheel_path),
            "--wheel-dir",
            str(tmp_path / "wheels"),
            "--exclude",
            "libRDKit*",
            "--exclude",
            "libboost_python*",
        ]
    )


def test_build_wheel_raises_if_rdkit_has_no_supported_pythons() -> None:
    with (
        patch("build_wheel.get_cargo_version", return_value=Version("0.2.0")),
        patch("build_wheel.get_supported_pythons", return_value=[Version("3.8")]),
        pytest.raises(ValueError, match="No Python wheels found"),
    ):
        build_wheel.build_wheel("2026.3.2")


def test_build_wheel_raises_if_rdkit_has_no_wheel_for_the_running_python() -> None:
    with (
        patch("build_wheel.sys.version_info", SimpleNamespace(major=3, minor=12)),
        patch("build_wheel.get_cargo_version", return_value=Version("0.2.0")),
        patch("build_wheel.get_supported_pythons", return_value=[Version("3.10")]),
        pytest.raises(ValueError, match="no PyPI wheel for the running Python"),
    ):
        build_wheel.build_wheel("2026.3.2")


def test_build_wheel_builds_and_repairs_on_linux(tmp_path: Path) -> None:
    worktree_dir = tmp_path / "worktree"
    worktree_dir.mkdir()

    def fake_check_call(cmd: list[object], **unused_kwargs: object) -> None:
        out_dir = Path(cmd[cmd.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "smiles_fp-0.2.0-cp312-linux.whl").write_bytes(b"data")

    with (
        patch("build_wheel.sys.version_info", SimpleNamespace(major=3, minor=12)),
        patch("build_wheel.sys.platform", "linux"),
        patch("build_wheel.get_cargo_version", return_value=Version("0.2.0")),
        patch("build_wheel.get_supported_pythons", return_value=[Version("3.12")]),
        patch("build_wheel.sync_worktree", return_value=worktree_dir),
        patch("build_wheel.patch_pyproject") as mock_patch,
        patch("build_wheel.subprocess.check_call", side_effect=fake_check_call),
        patch("build_wheel.repair_linux_wheel") as mock_repair,
    ):
        build_wheel.build_wheel("2026.3.2")

    mock_patch.assert_called_once()
    mock_repair.assert_called_once()


def test_build_wheel_skips_repair_off_linux(tmp_path: Path) -> None:
    worktree_dir = tmp_path / "worktree"
    worktree_dir.mkdir()

    with (
        patch("build_wheel.sys.version_info", SimpleNamespace(major=3, minor=12)),
        patch("build_wheel.sys.platform", "darwin"),
        patch("build_wheel.get_cargo_version", return_value=Version("0.2.0")),
        patch("build_wheel.get_supported_pythons", return_value=[Version("3.12")]),
        patch("build_wheel.sync_worktree", return_value=worktree_dir),
        patch("build_wheel.patch_pyproject"),
        patch("build_wheel.subprocess.check_call") as mock_call,
        patch("build_wheel.repair_linux_wheel") as mock_repair,
        patch.object(build_wheel, "WHEEL_DIR", tmp_path / "wheels"),
    ):
        build_wheel.build_wheel("2026.3.2")

    mock_call.assert_called_once()
    mock_repair.assert_not_called()


def test_build_wheel_propagates_maturin_build_failures(tmp_path: Path) -> None:
    worktree_dir = tmp_path / "worktree"
    worktree_dir.mkdir()

    with (
        patch("build_wheel.sys.version_info", SimpleNamespace(major=3, minor=12)),
        patch("build_wheel.sys.platform", "linux"),
        patch("build_wheel.get_cargo_version", return_value=Version("0.2.0")),
        patch("build_wheel.get_supported_pythons", return_value=[Version("3.12")]),
        patch("build_wheel.sync_worktree", return_value=worktree_dir),
        patch("build_wheel.patch_pyproject"),
        patch(
            "build_wheel.subprocess.check_call",
            side_effect=subprocess.CalledProcessError(1, ["maturin"]),
        ),
        patch("build_wheel.repair_linux_wheel") as mock_repair,
        pytest.raises(subprocess.CalledProcessError),
    ):
        build_wheel.build_wheel("2026.3.2")

    mock_repair.assert_not_called()


def test_build_wheel_propagates_repair_failures(tmp_path: Path) -> None:
    worktree_dir = tmp_path / "worktree"
    worktree_dir.mkdir()

    def fake_check_call(cmd: list[object], **unused_kwargs: object) -> None:
        out_dir = Path(cmd[cmd.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "smiles_fp-0.2.0-cp312-linux.whl").write_bytes(b"data")

    with (
        patch("build_wheel.sys.version_info", SimpleNamespace(major=3, minor=12)),
        patch("build_wheel.sys.platform", "linux"),
        patch("build_wheel.get_cargo_version", return_value=Version("0.2.0")),
        patch("build_wheel.get_supported_pythons", return_value=[Version("3.12")]),
        patch("build_wheel.sync_worktree", return_value=worktree_dir),
        patch("build_wheel.patch_pyproject"),
        patch("build_wheel.subprocess.check_call", side_effect=fake_check_call),
        patch(
            "build_wheel.repair_linux_wheel",
            side_effect=subprocess.CalledProcessError(1, ["auditwheel"]),
        ),
        pytest.raises(subprocess.CalledProcessError),
    ):
        build_wheel.build_wheel("2026.3.2")
