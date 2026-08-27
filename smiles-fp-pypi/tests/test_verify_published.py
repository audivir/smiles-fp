"""Tests for verify_published.py."""

from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING
from unittest.mock import patch

import build_env
import verify_published
from packaging.version import Version

if TYPE_CHECKING:
    import pytest


def test_run_pytest_builds_the_expected_uv_command() -> None:
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")
    with patch("verify_published.subprocess.run", return_value=completed) as mock_run:
        result = verify_published.run_pytest("0.2.1", Version("2026.3.5"), Version("3.12"))

    assert result is completed
    mock_run.assert_called_once_with(
        [
            str(build_env.UV_EXE),
            "run",
            "--no-project",
            "--python",
            "3.12",
            "--with",
            "smiles-fp==0.2.1.2026.3.5",
            "--with",
            "rdkit==2026.3.5",
            "--with",
            "pytest",
            "--with",
            "pytest-benchmark",
            "python",
            "-m",
            "pytest",
            str(verify_published.TESTS_DIR),
            "--benchmark-skip",
            "-q",
        ],
        cwd=verify_published.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_main_warns_and_returns_1_when_nothing_is_published(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    with (
        patch("verify_published.get_cargo_version", return_value="0.2.1"),
        patch("verify_published.get_published_wheel_pairs", return_value=set()),
    ):
        assert verify_published.main() == 1

    assert "No published wheels found for smiles-fp==0.2.1.*" in caplog.text


def test_main_returns_0_when_every_pair_passes(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    pairs = {(Version("2026.3.5"), Version("3.12"))}
    passing = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    with (
        patch("verify_published.get_cargo_version", return_value="0.2.1"),
        patch("verify_published.get_published_wheel_pairs", return_value=pairs),
        patch("verify_published.tqdm", side_effect=lambda it, **unused_kwargs: it),
        patch("verify_published.run_pytest", return_value=passing) as mock_run_pytest,
    ):
        assert verify_published.main() == 0

    mock_run_pytest.assert_called_once_with("0.2.1", Version("2026.3.5"), Version("3.12"))
    assert "1/1 (RDKit, Python) pairs passed." in caplog.text


def test_main_logs_failures_and_returns_1_when_a_pair_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    pairs = {(Version("2024.3.6"), Version("3.10")), (Version("2026.3.5"), Version("3.12"))}
    passing = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    failing = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="assertion failed\n", stderr=""
    )

    def fake_run_pytest(
        unused_cargo_version: str, rdkit_ver: Version, unused_py_ver: Version
    ) -> subprocess.CompletedProcess[str]:
        return failing if rdkit_ver == Version("2026.3.5") else passing

    with (
        patch("verify_published.get_cargo_version", return_value="0.2.1"),
        patch("verify_published.get_published_wheel_pairs", return_value=pairs),
        patch("verify_published.tqdm", side_effect=lambda it, **unused_kwargs: it),
        patch("verify_published.run_pytest", side_effect=fake_run_pytest),
    ):
        assert verify_published.main() == 1

    assert "rdkit==2026.3.5 python==3.12" in caplog.text
    assert "assertion failed" in caplog.text
    assert "1/2 (RDKit, Python) pairs passed." in caplog.text
