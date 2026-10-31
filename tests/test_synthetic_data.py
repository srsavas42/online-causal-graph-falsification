import numpy as np
import pandas as pd
import pytest

from ccrulsif.synthetic_data import synthetic_data


def test_generation_reproducible():
    matrix = np.array([
        [0, 0],
        [1, 0],
    ], dtype=float)

    first = synthetic_data(matrix, N=30, seed=42)
    second = synthetic_data(matrix, N=30, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert first.shape == (30, 2)
    assert list(first.columns) == ["Var0", "Var1"]


def test_structural_change_occurs_at_correct_time():
    base = np.zeros((2, 2))
    changed = np.array([
        [0, 0],
        [2, 0],
    ], dtype=float)

    change_point = 7
    baseline = synthetic_data(base, N=15, seed=42)
    shifted = synthetic_data(
        base,
        N=15,
        change_points=[(change_point, changed)],
        seed=42,
    )

    # Same noise and structural equations before the change.
    np.testing.assert_allclose(
        shifted.iloc[:change_point],
        baseline.iloc[:change_point],
    )

    # Var0 is unaffected by the change.
    np.testing.assert_allclose(
        shifted["Var0"],
        baseline["Var0"],
    )

    # After the change: Var1 = 2 * Var0 + noise.
    # The unchanged simulation has Var1 = noise.
    np.testing.assert_allclose(
        shifted["Var1"].to_numpy()[change_point:]
        - baseline["Var1"].to_numpy()[change_point:],
        2 * baseline["Var0"].to_numpy()[change_point:],
    )


def test_singular_system_rejected():
    matrix = np.eye(2)

    with pytest.raises(ValueError, match="singular"):
        synthetic_data(matrix, N=5, seed=42)
