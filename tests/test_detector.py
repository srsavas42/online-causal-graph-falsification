import networkx as nx
import numpy as np
import pandas as pd
import pytest

from ccrulsif.detector import CCRuLSIFDetector


def make_detector():
    """Create a small deterministic streaming test case."""
    rng = np.random.default_rng(13)
    a = rng.normal(size=40)
    ref = pd.DataFrame({
        "A": a,
        "B": 1.5 * a + rng.normal(size=40),
    })

    graph = nx.DiGraph([("A", "B")])
    detector = CCRuLSIFDetector(
        graph=graph,
        reference_data=ref,
        window_size=5,
        n_centers=8,
        n_bootstrap=8,
        random_state=42,
    )
    return detector, ref


def test_waits_for_full_window():
    detector, ref = make_detector()
    records = ref.to_dict("records")

    for observation in records[:4]:
        assert detector.update(observation) is None

    result = detector.update(records[4])

    assert result is not None
    assert result.time_index == 4
    assert set(result.divergences) == {"B"}
    assert set(result.exceedances) == {"B"}
    assert np.isfinite(result.divergences["B"])
    assert result.exceedances["B"] == (
        result.divergences["B"] > detector.thresholds["B"]
    )


def test_thresholds_remain_fixed():
    detector, ref = make_detector()
    original = detector.thresholds

    for observation in ref.to_dict("records")[:10]:
        detector.update(observation)

    assert detector.thresholds == original

    # The public property should not expose mutable internal state.
    thresholds = detector.thresholds
    thresholds["B"] = -999.0
    assert detector.thresholds == original


def test_stream_reproducible():
    first, ref = make_detector()
    second, _ = make_detector()

    for observation in ref.to_dict("records")[:10]:
        result1 = first.update(observation)
        result2 = second.update(observation)

        if result1 is None:
            assert result2 is None
        else:
            assert result2 is not None
            assert result1.time_index == result2.time_index
            assert result1.divergences == pytest.approx(
                result2.divergences
            )
            assert result1.exceedances == result2.exceedances


def test_reset_clears_window():
    detector, ref = make_detector()
    records = ref.to_dict("records")

    for observation in records[:5]:
        detector.update(observation)

    detector.reset()

    for observation in records[:4]:
        assert detector.update(observation) is None

    result = detector.update(records[4])
    assert result is not None
    assert result.time_index == 4


def test_rejects_cyclic_graph():
    graph = nx.DiGraph([
        ("A", "B"),
        ("B", "A"),
    ])
    ref = pd.DataFrame({
        "A": [1.0, 2.0, 3.0],
        "B": [2.0, 3.0, 4.0],
    })

    with pytest.raises(ValueError, match="acyclic"):
        CCRuLSIFDetector(
            graph, ref, window_size=2
        )


def test_rejects_nonfinite_observation():
    detector, _ = make_detector()

    with pytest.raises(ValueError, match="finite"):
        detector.update({
            "A": np.nan,
            "B": 1.0,
        })
