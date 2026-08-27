"""Tests for build_wheels.py."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import build_wheels
import pytest
import requests
from packaging.version import Version


def test_get_cargo_version_raises_if_cargo_toml_is_missing(tmp_path: Path) -> None:
    with (
        patch.object(build_wheels, "CARGO_TOML", tmp_path / "missing.toml"),
        pytest.raises(FileNotFoundError, match="Missing"),
    ):
        build_wheels.get_cargo_version()


def test_get_cargo_version_raises_if_version_key_is_missing(tmp_path: Path) -> None:
    cargo_toml = tmp_path / "Cargo.toml"
    cargo_toml.write_text('[package]\nname = "smiles_fp"\n')
    with (
        patch.object(build_wheels, "CARGO_TOML", cargo_toml),
        pytest.raises(ValueError, match=r"\[package\] -> 'version'"),
    ):
        build_wheels.get_cargo_version()


def test_get_cargo_version_reads_the_package_version(tmp_path: Path) -> None:
    cargo_toml = tmp_path / "Cargo.toml"
    cargo_toml.write_text('[package]\nname = "smiles_fp"\nversion = "0.2.0"\n')
    with patch.object(build_wheels, "CARGO_TOML", cargo_toml):
        assert build_wheels.get_cargo_version() == Version("0.2.0")


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
    with patch("build_wheels.requests.get", return_value=mock_response({"urls": urls})):
        assert build_wheels.get_supported_pythons(Version("2026.3.2")) == [
            Version("3.10"),
            Version("3.12"),
        ]


def test_get_supported_pythons_raises_on_request_failure() -> None:
    with (
        patch("build_wheels.requests.get", side_effect=requests.RequestException("boom")),
        pytest.raises(RuntimeError, match="Failed to fetch RDKit"),
    ):
        build_wheels.get_supported_pythons(Version("2026.3.2"))


def test_get_python_exe_installs_and_locates_the_interpreter() -> None:
    with (
        patch("build_wheels.subprocess.check_call") as mock_call,
        patch(
            "build_wheels.subprocess.check_output", return_value="/opt/py/bin/python3.12\n"
        ) as mock_output,
    ):
        assert build_wheels.get_python_exe(Version("3.12")) == "/opt/py/bin/python3.12"

    mock_call.assert_called_once_with([build_wheels.UV_EXE, "python", "install", "3.12"])
    mock_output.assert_called_once_with([build_wheels.UV_EXE, "python", "find", "3.12"], text=True)


def test_sync_worktree_reuses_a_dirty_snapshot_and_resets_an_existing_worktree(
    tmp_path: Path,
) -> None:
    worktree_root = tmp_path / "worktrees"
    worktree_dir = worktree_root / "rdkit-2026.3.2"
    worktree_dir.mkdir(parents=True)

    with (
        patch.object(build_wheels, "WORKTREE_ROOT", worktree_root),
        patch("build_wheels.subprocess.check_output", return_value="deadbeef\n") as mock_output,
        patch("build_wheels.subprocess.check_call") as mock_call,
    ):
        result = build_wheels.sync_worktree(Version("2026.3.2"))

    assert result == worktree_dir
    mock_output.assert_called_once_with(
        ["git", "stash", "create"], cwd=build_wheels.REPO_ROOT, text=True
    )
    mock_call.assert_called_once_with(
        ["git", "-C", str(worktree_dir), "reset", "--hard", "deadbeef"]
    )


def test_sync_worktree_falls_back_to_head_and_creates_a_new_worktree(tmp_path: Path) -> None:
    worktree_root = tmp_path / "worktrees"

    def fake_check_output(cmd: list[str], **unused_kwargs: object) -> str:
        return "" if cmd == ["git", "stash", "create"] else "cafebabe\n"

    with (
        patch.object(build_wheels, "WORKTREE_ROOT", worktree_root),
        patch("build_wheels.subprocess.check_output", side_effect=fake_check_output),
        patch("build_wheels.subprocess.check_call") as mock_call,
    ):
        result = build_wheels.sync_worktree(Version("2026.3.2"))

    assert result == worktree_root / "rdkit-2026.3.2"
    assert worktree_root.is_dir()
    mock_call.assert_called_once_with(
        ["git", "worktree", "add", "--detach", str(result), "cafebabe"],
        cwd=build_wheels.REPO_ROOT,
    )


def test_patch_pyproject_pins_version_python_range_and_pinned_rdkit(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        '[project]\nversion = "0.0.0"\nrequires-python = ">=3.10"\n'
        'dependencies = ["joblib", "rdkit==2026.3.2", "scipy"]\n'
    )

    build_wheels.patch_pyproject(path, "0.2.0.2026.3.5", ">=3.10,<=3.13", Version("2026.3.5"))

    content = path.read_text()
    assert 'version = "0.2.0.2026.3.5"' in content
    assert 'requires-python = ">=3.10,<=3.13"' in content
    assert '"rdkit==2026.3.5"' in content


def test_patch_pyproject_pins_an_unpinned_rdkit_dependency(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text('[project]\nversion = "0.0.0"\ndependencies = ["rdkit"]\n')

    build_wheels.patch_pyproject(path, "0.2.0.2026.3.5", ">=3.10,<=3.13", Version("2026.3.5"))

    assert '"rdkit==2026.3.5"' in path.read_text()


def test_repair_linux_wheel_invokes_auditwheel_with_excludes(tmp_path: Path) -> None:
    wheel_path = tmp_path / "smiles_fp-0.2.0-cp312-linux.whl"
    with (
        patch.object(build_wheels, "WHEEL_DIR", tmp_path / "wheels"),
        patch("build_wheels.subprocess.check_call") as mock_call,
    ):
        build_wheels.repair_linux_wheel(wheel_path)

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


def test_build_wheels_raises_if_rdkit_has_no_supported_pythons() -> None:
    with (
        patch("build_wheels.get_cargo_version", return_value=Version("0.2.0")),
        patch("build_wheels.get_supported_pythons", return_value=[Version("3.8")]),
        pytest.raises(ValueError, match="No Python wheels found"),
    ):
        build_wheels.build_wheels(Version("2026.3.2"))


def test_build_wheels_builds_and_repairs_on_linux(tmp_path: Path) -> None:
    worktree_dir = tmp_path / "worktree"
    worktree_dir.mkdir()
    wheel_dir = tmp_path / "wheels"

    def fake_check_call(
        cmd: list[object], unused_cwd: object = None, **unused_kwargs: object
    ) -> None:
        out_dir = Path(cmd[cmd.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "smiles_fp-0.2.0-cp312-linux.whl").write_bytes(b"data")

    with (
        patch("build_wheels.get_cargo_version", return_value=Version("0.2.0")),
        patch("build_wheels.get_supported_pythons", return_value=[Version("3.12")]),
        patch("build_wheels.sync_worktree", return_value=worktree_dir),
        patch("build_wheels.patch_pyproject") as mock_patch,
        patch("build_wheels.get_python_exe", return_value="/opt/py/bin/python3.12"),
        patch("build_wheels.subprocess.check_call", side_effect=fake_check_call),
        patch("build_wheels.repair_linux_wheel") as mock_repair,
        patch.object(build_wheels, "WHEEL_DIR", wheel_dir),
        patch("build_wheels.sys.platform", "linux"),
    ):
        build_wheels.build_wheels(Version("2026.3.2"))

    mock_patch.assert_called_once()
    mock_repair.assert_called_once()


def test_build_wheels_skips_repair_off_linux(tmp_path: Path) -> None:
    worktree_dir = tmp_path / "worktree"
    worktree_dir.mkdir()
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()

    with (
        patch("build_wheels.get_cargo_version", return_value=Version("0.2.0")),
        patch("build_wheels.get_supported_pythons", return_value=[Version("3.12")]),
        patch("build_wheels.sync_worktree", return_value=worktree_dir),
        patch("build_wheels.patch_pyproject"),
        patch("build_wheels.get_python_exe", return_value="/opt/py/bin/python3.12"),
        patch("build_wheels.subprocess.check_call") as mock_call,
        patch("build_wheels.repair_linux_wheel") as mock_repair,
        patch.object(build_wheels, "WHEEL_DIR", wheel_dir),
        patch("build_wheels.sys.platform", "darwin"),
    ):
        build_wheels.build_wheels(Version("2026.3.2"))

    mock_call.assert_called_once()
    mock_repair.assert_not_called()


def test_build_wheels_logs_and_continues_when_the_maturin_build_fails(tmp_path: Path) -> None:
    worktree_dir = tmp_path / "worktree"
    worktree_dir.mkdir()

    with (
        patch("build_wheels.get_cargo_version", return_value=Version("0.2.0")),
        patch("build_wheels.get_supported_pythons", return_value=[Version("3.12")]),
        patch("build_wheels.sync_worktree", return_value=worktree_dir),
        patch("build_wheels.patch_pyproject"),
        patch("build_wheels.get_python_exe", return_value="/opt/py/bin/python3.12"),
        patch(
            "build_wheels.subprocess.check_call",
            side_effect=subprocess.CalledProcessError(1, ["maturin"]),
        ),
        patch("build_wheels.repair_linux_wheel") as mock_repair,
        patch("build_wheels.sys.platform", "linux"),
    ):
        build_wheels.build_wheels(Version("2026.3.2"))

    mock_repair.assert_not_called()


def test_build_wheels_logs_and_continues_when_the_repair_fails(tmp_path: Path) -> None:
    worktree_dir = tmp_path / "worktree"
    worktree_dir.mkdir()
    wheel_dir = tmp_path / "wheels"

    def fake_check_call(
        cmd: list[object], unused_cwd: object = None, **unused_kwargs: object
    ) -> None:
        out_dir = Path(cmd[cmd.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "smiles_fp-0.2.0-cp312-linux.whl").write_bytes(b"data")

    with (
        patch("build_wheels.get_cargo_version", return_value=Version("0.2.0")),
        patch("build_wheels.get_supported_pythons", return_value=[Version("3.12")]),
        patch("build_wheels.sync_worktree", return_value=worktree_dir),
        patch("build_wheels.patch_pyproject"),
        patch("build_wheels.get_python_exe", return_value="/opt/py/bin/python3.12"),
        patch("build_wheels.subprocess.check_call", side_effect=fake_check_call),
        patch(
            "build_wheels.repair_linux_wheel",
            side_effect=subprocess.CalledProcessError(1, ["auditwheel"]),
        ),
        patch.object(build_wheels, "WHEEL_DIR", wheel_dir),
        patch("build_wheels.sys.platform", "linux"),
    ):
        build_wheels.build_wheels(Version("2026.3.2"))


def test_cli_builds_every_requested_rdkit_version() -> None:
    with patch("build_wheels.build_wheels") as mock_build:
        build_wheels.cli(["2026.3.2", "2025.9.6"])

    assert mock_build.call_args_list == [
        ((Version("2026.3.2"),),),
        ((Version("2025.9.6"),),),
    ]


def test_main_registers_the_command_and_runs_the_app() -> None:
    mock_app = MagicMock()
    with patch("build_wheels.doctyper.DocTyper", return_value=mock_app):
        build_wheels.main()

    mock_app.command.return_value.assert_called_once_with(build_wheels.cli)
    mock_app.assert_called_once_with()
