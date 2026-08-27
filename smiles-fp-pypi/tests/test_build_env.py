"""Tests for build_env.py."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from unittest.mock import patch

import build_env
import pytest
from packaging.version import Version


def test_find_required_exe() -> None:
    with patch("build_env.shutil.which", return_value="/usr/bin/uv"):
        assert build_env.find_required_exe("uv") == Path("/usr/bin/uv")

    with (
        patch("build_env.shutil.which", return_value=None),
        pytest.raises(FileNotFoundError, match="No uv binary found"),
    ):
        build_env.find_required_exe("uv")


def test_find_install_name_tool() -> None:
    with patch("build_env.shutil.which", return_value="/usr/bin/install_name_tool"):
        assert build_env.find_install_name_tool("darwin") == Path("/usr/bin/install_name_tool")

    assert build_env.find_install_name_tool("linux") == Path()


@pytest.mark.parametrize(
    ("platform", "machine", "expected"),
    [
        ("win32", "x86_64", "x86_64-pc-windows-msvc"),
        ("darwin", "arm64", "aarch64-apple-darwin"),
        ("darwin", "x86_64", "x86_64-apple-darwin"),
        ("linux", "aarch64", "aarch64-manylinux_2_28"),
        ("linux", "x86_64", "x86_64-manylinux_2_28"),
    ],
)
def test_get_uv_platform(
    platform: Literal["darwin", "linux", "win32"],
    machine: Literal["arm64", "aarch64", "x86_64"],
    expected: str,
) -> None:
    assert build_env.get_uv_platform(platform, machine) == expected


def write_rdkit_headers_metadata(cache_dir: Path, rdkit_ver: str, boost_req: str) -> None:
    dist_info = cache_dir / "headers" / rdkit_ver / "rdkit_headers-1.0.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        f"Version: {rdkit_ver}\nRequires-Dist: {boost_req}\nRequires-Dist: rdkit=={rdkit_ver}\n"
    )


def test_fetch_headers(tmp_path: Path) -> None:
    header_dir = tmp_path / "headers" / "2026.3.2"
    (header_dir / "rdkit_headers" / "include" / "rdkit").mkdir(parents=True)
    (header_dir / "boost_headers" / "include").mkdir(parents=True)

    with patch("build_env.subprocess.check_call") as mock_call:
        boost_include, rdkit_include = build_env.fetch_headers(Version("2026.3.2"), tmp_path)

    mock_call.assert_not_called()
    assert boost_include == header_dir / "boost_headers" / "include"
    assert rdkit_include == header_dir / "rdkit_headers" / "include" / "rdkit"


def test_fetch_headers_installs_boost_headers_matching_rdkit_headers_metadata(
    tmp_path: Path,
) -> None:
    def fake_check_call(cmd: list[object], **unused_kwargs: object) -> None:
        if "rdkit-headers==2026.3.2.*" in cmd:
            write_rdkit_headers_metadata(tmp_path, "2026.3.2", "boost-headers~=1.85.0")

    with patch("build_env.subprocess.check_call", side_effect=fake_check_call) as mock_call:
        boost_include, rdkit_include = build_env.fetch_headers(Version("2026.3.2"), tmp_path)

    assert mock_call.call_count == 2
    boost_cmd = mock_call.call_args_list[1].args[0]
    assert "boost-headers==1.85.0.*" in boost_cmd
    assert (
        rdkit_include == tmp_path / "headers" / "2026.3.2" / "rdkit_headers" / "include" / "rdkit"
    )
    assert boost_include == tmp_path / "headers" / "2026.3.2" / "boost_headers" / "include"


def test_fetch_headers_raises_if_metadata_is_missing(tmp_path: Path) -> None:
    with (
        patch("build_env.subprocess.check_call"),
        pytest.raises(FileNotFoundError, match="dist-info metadata"),
    ):
        build_env.fetch_headers(Version("2026.3.2"), tmp_path)


def test_fetch_headers_raises_if_boost_requirement_is_missing(tmp_path: Path) -> None:
    dist_info = tmp_path / "headers" / "2026.3.2" / "rdkit_headers-1.0.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text("Version: 2026.3.2\nRequires-Dist: rdkit==2026.3.2\n")

    with (
        patch("build_env.subprocess.check_call"),
        pytest.raises(ValueError, match="pinned boost-headers version"),
    ):
        build_env.fetch_headers(Version("2026.3.2"), tmp_path)


def test_get_lib_cache(tmp_path: Path) -> None:
    cache = build_env.get_lib_cache(
        tmp_path, Version("3.12"), Version("2026.3.2"), "darwin", "arm64"
    )
    assert cache == tmp_path / "3.12_2026.3.2_darwin_arm64"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("libfoo-abc.dylib.1", ".dylib"),
        ("libfoo-abc.dll", ".dll"),
        ("libfoo-abc.so.1.2", ".so"),
    ],
)
def test_lib_ext(filename: str, expected: str) -> None:
    assert build_env.lib_ext(Path(filename)) == expected


def test_stage_lib(tmp_path: Path) -> None:
    windows_dll = "python312.dll"
    linux_so = "libRDKitDataStructs-abcdef.so.1"
    linux_name = "RDKitDataStructs"
    semver_linux_so = "libboost_python312-abcdef.so.1.85.0"
    semver_linux_name = "boost_python312"
    darwin_so = "libboost_python312.so.1.85.0"
    for name in (windows_dll, linux_so, semver_linux_so, darwin_so):
        (tmp_path / name).write_bytes(b"data")

    lib_cache = tmp_path / "cache"
    lib_cache.mkdir()

    assert build_env.stage_lib(tmp_path / ".hidden.so", lib_cache, "linux") is None
    assert build_env.stage_lib(tmp_path / "notlib-123.so", lib_cache, "linux") is None
    assert build_env.stage_lib(tmp_path / windows_dll, lib_cache, "win32") is None
    assert build_env.stage_lib(tmp_path / linux_so, lib_cache, "linux") is None
    assert build_env.stage_lib(tmp_path / semver_linux_so, lib_cache, "linux") == semver_linux_name
    assert (lib_cache / windows_dll).exists()
    assert (lib_cache / linux_so).exists()
    assert (lib_cache / f"lib{linux_name}.so").exists()
    assert (lib_cache / semver_linux_so).exists()
    assert (lib_cache / f"lib{semver_linux_name}.so").exists()

    # install_name_tool is called
    with patch("build_env.subprocess.run") as mock_run:
        build_env.stage_lib(tmp_path / darwin_so, lib_cache, "darwin")

    assert mock_run.call_count == 2


def test_find_cached_boost_link_name(tmp_path: Path) -> None:
    (tmp_path / "libRDKitDataStructs.so").write_bytes(b"data")
    assert build_env.find_cached_boost_link_name(tmp_path) is None

    (tmp_path / "libboost_python312.so").write_bytes(b"data")
    assert build_env.find_cached_boost_link_name(tmp_path) == "boost_python312"


def test_fetch_libs_reuses_an_existing_cache(tmp_path: Path) -> None:
    lib_cache = tmp_path / "3.12_2026.3.2_linux_x86_64"
    lib_cache.mkdir()
    (lib_cache / "libboost_python312.so").write_bytes(b"data")

    with patch("build_env.subprocess.check_call") as mock_call:
        result = build_env.fetch_libs(
            tmp_path, Version("3.12"), Version("2026.3.2"), "linux", "x86_64"
        )

    mock_call.assert_not_called()
    assert result == (lib_cache, "boost_python312")


def test_fetch_libs_refetches_when_the_cache_dir_exists_but_is_incomplete(tmp_path: Path) -> None:
    lib_cache = tmp_path / "3.12_2026.3.2_linux_x86_64"
    lib_cache.mkdir()  # exists, but no boost_python lib staged yet

    def fake_check_call(unused_cmd: list[object], **unused_kwargs: object) -> None:
        wheel_dir = lib_cache / "_wheel"
        wheel_dir.mkdir(parents=True)
        (wheel_dir / "libboost_python312-abc.so.1.85.0").write_bytes(b"data")

    with patch("build_env.subprocess.check_call", side_effect=fake_check_call):
        result = build_env.fetch_libs(
            tmp_path, Version("3.12"), Version("2026.3.2"), "linux", "x86_64"
        )

    assert result == (lib_cache, "boost_python312")


def test_fetch_libs_extracts_the_rdkit_wheel_when_uncached(tmp_path: Path) -> None:
    lib_cache = tmp_path / "3.12_2026.3.2_linux_x86_64"

    def fake_check_call(unused_cmd: list[object], **unused_kwargs: object) -> None:
        wheel_dir = lib_cache / "_wheel" / "rdkit.libs"
        wheel_dir.mkdir(parents=True)
        (wheel_dir / "libboost_python312-abc.so.1.85.0").write_bytes(b"data")
        (wheel_dir / "libRDKitDataStructs-abc.so.1").write_bytes(b"data")
        (wheel_dir / "not_a_library.txt").write_bytes(b"data")

    with patch("build_env.subprocess.check_call", side_effect=fake_check_call):
        result = build_env.fetch_libs(
            tmp_path, Version("3.12"), Version("2026.3.2"), "linux", "x86_64"
        )

    assert result == (lib_cache, "boost_python312")
    assert (lib_cache / "libboost_python312.so").exists()
    assert (lib_cache / "libRDKitDataStructs.so").exists()
    assert not (lib_cache / "_wheel").exists()


def test_fetch_libs_raises_if_boost_python_is_never_found(tmp_path: Path) -> None:
    lib_cache = tmp_path / "3.12_2026.3.2_linux_x86_64"

    def fake_check_call(unused_cmd: list[object], **unused_kwargs: object) -> None:
        wheel_dir = lib_cache / "_wheel"
        wheel_dir.mkdir(parents=True)
        (wheel_dir / "libRDKitDataStructs-abc.so.1").write_bytes(b"data")

    with (
        patch("build_env.subprocess.check_call", side_effect=fake_check_call),
        pytest.raises(ValueError, match="boost_python library name"),
    ):
        build_env.fetch_libs(tmp_path, Version("3.12"), Version("2026.3.2"), "linux", "x86_64")


def test_build_env_wires_headers_and_libs_into_the_expected_env_vars(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with (
        patch.object(build_env, "CACHE_DIR", tmp_path),
        patch(
            "build_env.fetch_headers",
            return_value=(tmp_path / "boost" / "include", tmp_path / "rdkit" / "include"),
        ) as mock_headers,
        patch(
            "build_env.fetch_libs", return_value=(tmp_path / "libs", "boost_python312")
        ) as mock_libs,
    ):
        build_env.build_env(Version("2026.3.2"), Version("3.12"))

    mock_headers.assert_called_once_with(Version("2026.3.2"), tmp_path)
    mock_libs.assert_called_once()
    assert (tmp_path / "pip_libs").is_dir()

    out = capsys.readouterr().out
    assert f"BOOST_INCLUDE_DIR={tmp_path / 'boost' / 'include'}" in out
    assert f"RDKIT_INCLUDE_DIR={tmp_path / 'rdkit' / 'include'}" in out
    assert f"PIP_LIB_DIR={tmp_path / 'libs'}" in out
    assert "BOOST_LINK_NAME=boost_python312" in out


def test_main_derives_the_python_version_from_the_running_interpreter() -> None:
    version_info = build_env.sys.version_info
    expected_py_ver = Version(f"{version_info.major}.{version_info.minor}")
    with patch("build_env.build_env") as mock_build_env:
        build_env.main("2026.3.2")

    mock_build_env.assert_called_once_with(Version("2026.3.2"), expected_py_ver)
