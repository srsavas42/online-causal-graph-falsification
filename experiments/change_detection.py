import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ccrulsif.detector import CCRuLSIFDetector
from ccrulsif.synthetic_data import synthetic_data


def main():
    seed = 42
    n_samples = 500
    reference_size = 150
    change_point = 250
    window_size = 50

    # B[child, parent] is the coefficient on parent -> child.
    base_matrix = np.array([
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
    ], dtype=float)

    # Original thesis scenario: Var4 switches parent from Var3 to Var1.
    changed_matrix = base_matrix.copy()
    changed_matrix[4, 3] = 0.0
    changed_matrix[4, 1] = 1.0

    graph = nx.DiGraph([
        ("Var0", "Var1"),
        ("Var0", "Var2"),
        ("Var1", "Var3"),
        ("Var2", "Var3"),
        ("Var3", "Var4"),
    ])

    data = synthetic_data(
        base_matrix,
        N=n_samples,
        change_points=[(change_point, changed_matrix)],
        seed=seed,
    )

    detector = CCRuLSIFDetector(
        graph=graph,
        reference_data=data.iloc[:reference_size],
        window_size=window_size,
        alpha=0.1,
        n_centers=80,
        n_bootstrap=100,
        percentile=95,
        random_state=seed,
    )

    times = []
    divergences = {node: [] for node in detector.thresholds}
    alarms = {node: [] for node in detector.thresholds}

    for t, observation in data.iloc[reference_size:].iterrows():
        result = detector.update(observation.to_dict())
        if result is None:
            continue  

        times.append(t)  
        for node, score in result.divergences.items():
            divergences[node].append(score)
            if result.exceedances[node]:
                alarms[node].append(t)

    print(f"True change point: t={change_point}")
    for node, alarm_times in alarms.items():
        pre = next((t for t in alarm_times if t < change_point), None)
        post = next((t for t in alarm_times if t >= change_point), None)
        print(f"{node}: first pre-change alarm={pre}; first post-change alarm={post}")

    fig, ax = plt.subplots(figsize=(11, 6))
    for node, scores in divergences.items():
        (line,) = ax.plot(times, scores, label=node, linewidth=1.5)
        ax.axhline(
            detector.thresholds[node],
            color=line.get_color(),
            linestyle="--",
            alpha=0.65,
        )

    ax.axvline(change_point, color="black", linestyle=":", label="True change point")
    ax.plot([], [], color="gray", linestyle="--", label="Calibrated thresholds")
    ax.set(
        xlabel="Observation index",
        ylabel="Estimated divergence",
        title="Online CC-RuLSIF: change in the parent of Var4",
    )
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
