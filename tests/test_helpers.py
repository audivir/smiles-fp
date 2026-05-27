"""Test the helpers."""

from __future__ import annotations

import conftest
import numpy as np
import pytest
import smiles_fp_rs


def test_to_matrix() -> None:
    np.testing.assert_array_equal(smiles_fp_rs.to_matrix(np.array([]), 0), conftest.zero_array(2))
    np.testing.assert_array_equal(
        smiles_fp_rs.to_matrix(np.array([]), 1), conftest.zero_array(2, 1)
    )
    np.testing.assert_array_equal(
        smiles_fp_rs.to_matrix(np.array([1.0], dtype=np.float64), 2),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        smiles_fp_rs.to_matrix(np.array([1.0], dtype=np.float64), 2, diagonal=1.0),
        np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="must be 1D"):
        (smiles_fp_rs.to_matrix(np.array([[1.0]], dtype=np.float64), 1),)

    with pytest.raises(ValueError, match="unexpected size"):
        (smiles_fp_rs.to_matrix(np.array([1.0], dtype=np.float64), 1),)
