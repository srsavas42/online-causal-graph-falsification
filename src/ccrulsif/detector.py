
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from .cc_rulsif import (
    calibrate_delta,
    fit_rulsif,
    get_child_parent_sets,
)


@dataclass(frozen=True)
class DetectionResult:
    time_index: int
    divergences: dict[str, float]
    exceedances: dict[str, bool]


class CCRuLSIFDetector:
    """Monitor a supplied DAG using online RuLSIF comparisons."""

    def __init__(
        self,
        graph: nx.DiGraph,
        reference_data: pd.DataFrame,
        window_size: int,
        *,
        alpha: float = 0.1,
        n_centers: int = 80,
        n_bootstrap: int = 250,
        percentile: float = 95.0,
        random_state: int | None = None,
    ):
        if not isinstance(graph, nx.DiGraph):
            raise TypeError("graph must be a networkx.DiGraph")

        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("graph must be acyclic")

        if window_size < 2:
            raise ValueError("window_size must be at least 2")

        self._columns = list(graph.nodes)

        missing = set(self._columns) - set(reference_data.columns)
        if missing:
            raise ValueError(f"Missing reference columns: {missing}")

        self._reference = (
            reference_data.loc[:, self._columns]
            .astype(float)
            .copy()
        )

        if not np.isfinite(self._reference.to_numpy()).all():
            raise ValueError("Reference data must be finite")

        self._parent_sets = get_child_parent_sets(graph)

        if not self._parent_sets:
            raise ValueError("Graph has no parent-child groups")

        self._alpha = alpha
        self._n_centers = n_centers
        self._window_size = window_size

        # Baseline threhsold calibration
        self._thresholds = calibrate_delta(
            self._reference,
            self._parent_sets,
            alpha=alpha,
            n_bootstrap=n_bootstrap,
            percentile=percentile,
            n_centers=n_centers,
            random_state=random_state,
        )

        self.reset()

    def update(
        self,
        observation: Mapping[str, float],
    ) -> DetectionResult | None:
        """Consume one observation and evaluate a full window."""

        values = np.asarray(
            [observation[col] for col in self._columns],
            dtype=float,
        )

        if not np.isfinite(values).all():
            raise ValueError("Observation must be finite")

        self._time_index += 1
        self._window.append(values)

        if len(self._window) < self._window_size:
            return None

        recent_data = pd.DataFrame(
            list(self._window),
            columns=self._columns,
        )

        divergences = {}
        exceedances = {}

        for child, parents in self._parent_sets.items():
            columns = parents + [child]

            X_ref = self._reference[columns].to_numpy()
            X_new = recent_data[columns].to_numpy()

            _, divergence = fit_rulsif(
                X_ref,
                X_new,
                alpha=self._alpha,
                n_centers=self._n_centers,
            )

            score = float(divergence)

            divergences[child] = score
            exceedances[child] = (
                score > self._thresholds[child]
            )

        return DetectionResult(
            time_index=self._time_index,
            divergences=divergences,
            exceedances=exceedances,
        )

    def reset(self) -> None:
        """Clear streaming observations; retain calibration."""
        self._window = deque(maxlen=self._window_size)
        self._time_index = -1

    @property
    def thresholds(self) -> dict[str, float]:
        return dict(self._thresholds)
