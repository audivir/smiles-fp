"""Tests for build_env.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import build_env
import pytest
from packaging.version import Version


def test_find_required_exe_returns_the_resolved_path() -> None:
    with patch("build_env.shutil.which", return_value="/usr/bin/uv"):
        assert build_env.find_required_exe("uv") == Path("/usr/bin/uv")


def test_find_required_exe_raises_if_missing() -> None:
    with (
        patch("build_env.shutil.which", return_value=None),
        pytest.raises(FileNotFoundError, match="No uv binary found"),
    ):
        build_env.find_required_exe("uv")


def test_find_install_name_tool_on_darwin() -> None:
    with patch("build_env.shutil.which", return_value="/usr/bin/install_name_tool"):
        assert build_env.find_install_name_tool("darwin") == Path("/usr/bin/install_name_tool")


def test_find_install_name_tool_off_darwin_is_not_needed() -> None:
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
def test_get_uv_platform_maps_host_to_uv_tag(platform: str, machine: str, expected: str) -> None:
    assert build_env.get_uv_platform(platform, machine) == expected  # type: ignore[arg-type]


def test_fetch_headers_reuses_an_existing_cache(tmp_path: Path) -> None:
    header_dir = tmp_path / "headers" / "2026.3.2"
    (header_dir / "rdkit_headers" / "include" / "rdkit").mkdir(parents=True)
    (header_dir / "boost_headers" / "include").mkdir(parents=True)

    with patch("build_env.subprocess.check_call") as mock_call:
        boost_include, rdkit_include = build_env.fetch_headers(Version("2026.3.2"), tmp_path)

    mock_call.assert_not_called()
    assert boost_include == header_dir / "boost_headers" / "include"
    assert rdkit_include == header_dir / "rdkit_headers" / "include" / "rdkit"


def write_rdkit_headers_metadata(cache_dir: Path, rdkit_ver: str, boost_req: str) -> None:
    dist_info = cache_dir / "headers" / rdkit_ver / "rdkit_headers-1.0.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        f"Version: {rdkit_ver}\nRequires-Dist: {boost_req}\nRequires-Dist: rdkit=={rdkit_ver}\n"
    )


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


def test_get_lib_cache_defaults_to_the_current_host(tmp_path: Path) -> None:
    with (
        patch("build_env.sys.platform", "linux"),
        patch("build_env.platform_mod.machine", return_value="x86_64"),
    ):
        cache = build_env.get_lib_cache(tmp_path, Version("3.12"), Version("2026.3.2"))
    assert cache == tmp_path / "3.12_2026.3.2_linux_x86_64"


def test_get_lib_cache_uses_explicit_platform_and_machine(tmp_path: Path) -> None:
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
def test_lib_ext_maps_versioned_filenames_to_a_canonical_extension(
    filename: str, expected: str
) -> None:
    assert build_env.lib_ext(Path(filename)) == expected


def test_stage_lib_skips_files_with_no_alnum_core_name(tmp_path: Path) -> None:
    assert build_env.stage_lib(Path(".hidden.so"), tmp_path, "linux") is None


def test_stage_lib_skips_non_lib_files_off_windows(tmp_path: Path) -> None:
    assert build_env.stage_lib(Path("notlib-123.so"), tmp_path, "linux") is None


def test_stage_lib_accepts_non_lib_files_on_windows(tmp_path: Path) -> None:
    src = tmp_path / "python312.dll"
    src.write_bytes(b"data")
    lib_cache = tmp_path / "cache"
    lib_cache.mkdir()

    result = build_env.stage_lib(src, lib_cache, "win32")

    assert result is None
    assert (lib_cache / "python312.dll").exists()


def test_stage_lib_copies_boost_python_and_returns_its_link_name(tmp_path: Path) -> None:
    src = tmp_path / "libboost_python312-abcdef.so.1.85.0"
    src.write_bytes(b"data")
    lib_cache = tmp_path / "cache"
    lib_cache.mkdir()

    result = build_env.stage_lib(src, lib_cache, "linux")

    assert result == "boost_python312"
    assert (lib_cache / src.name).exists()
    assert (lib_cache / "libboost_python312.so").exists()


def test_stage_lib_copies_non_boost_lib_without_a_link_name(tmp_path: Path) -> None:
    src = tmp_path / "libRDKitDataStructs-abcdef.so.1"
    src.write_bytes(b"data")
    lib_cache = tmp_path / "cache"
    lib_cache.mkdir()

    result = build_env.stage_lib(src, lib_cache, "linux")

    assert result is None
    assert (lib_cache / "libRDKitDataStructs.so").exists()


def test_stage_lib_rewrites_the_install_name_on_darwin(tmp_path: Path) -> None:
    src = tmp_path / "libboost_python312.so.1.85.0"
    src.write_bytes(b"data")
    lib_cache = tmp_path / "cache"
    lib_cache.mkdir()

    with patch("build_env.subprocess.run") as mock_run:
        build_env.stage_lib(src, lib_cache, "darwin")

    assert mock_run.call_count == 2


def test_find_cached_boost_link_name_returns_none_when_absent(tmp_path: Path) -> None:
    (tmp_path / "libRDKitDataStructs.so").write_bytes(b"data")
    assert build_env.find_cached_boost_link_name(tmp_path) is None


def test_find_cached_boost_link_name_finds_the_staged_library(tmp_path: Path) -> None:
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


def test_cli_converts_versions_and_delegates() -> None:
    with patch("build_env.build_env") as mock_build_env:
        build_env.cli("2026.3.2", "3.12")

    mock_build_env.assert_called_once_with(Version("2026.3.2"), Version("3.12"))


def test_main_registers_the_command_and_runs_the_app() -> None:
    mock_app = MagicMock()
    with patch("build_env.doctyper.DocTyper", return_value=mock_app):
        build_env.main()

    mock_app.command.return_value.assert_called_once_with(build_env.cli)
    mock_app.assert_called_once_with()
