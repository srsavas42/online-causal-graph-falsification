import networkx as nx
import numpy as np
import pandas as pd
import pytest

from ccrulsif.ccrulsif import (
    calibrate_delta,
    fit_rulsif,
    get_child_parent_sets,
)


def test_rulsif_returns_finite_outputs():
    rng = np.random.default_rng(10)
    ref = rng.normal(size=(40, 2))
    new = rng.normal(loc=1.0, size=(40, 2))

    ratio, divergence = fit_rulsif(
        ref, new, n_centers=12,
        rng=np.random.default_rng(20),
    )

    estimates = ratio(new[:5])

    assert estimates.shape == (5,)
    assert np.isfinite(estimates).all()
    assert np.isfinite(divergence)
    assert divergence >= 0


def test_rulsif_reproducible():
    rng = np.random.default_rng(10)
    ref = rng.normal(size=(40, 2))
    new = rng.normal(size=(40, 2))

    ratio1, div1 = fit_rulsif(
        ref, new, n_centers=12,
        rng=np.random.default_rng(42),
    )
    ratio2, div2 = fit_rulsif(
        ref, new, n_centers=12,
        rng=np.random.default_rng(42),
    )

    assert div1 == pytest.approx(div2)
    np.testing.assert_allclose(
        ratio1(new[:5]), ratio2(new[:5])
    )


def test_parent_set_extraction():
    graph = nx.DiGraph([
        ("A", "C"),
        ("B", "C"),
        ("C", "D"),
    ])

    assert get_child_parent_sets(graph) == {
        "C": ["A", "B"],
        "D": ["C"],
    }

    with_roots = get_child_parent_sets(
        graph, include_roots=True
    )
    assert with_roots["A"] == []
    assert with_roots["B"] == []


def test_bootstrap_calibration_reproducible():
    rng = np.random.default_rng(10)
    ref = pd.DataFrame({
        "A": rng.normal(size=40),
        "B": rng.normal(size=40),
    })
    parent_sets = {"B": ["A"]}

    kwargs = dict(
        n_bootstrap=8,
        n_centers=8,
        random_state=42,
    )

    thresholds1 = calibrate_delta(ref, parent_sets, **kwargs)
    thresholds2 = calibrate_delta(ref, parent_sets, **kwargs)

    assert thresholds1 == pytest.approx(thresholds2)
    assert set(thresholds1) == {"B"}
    assert np.isfinite(thresholds1["B"])
    assert thresholds1["B"] >= 0
