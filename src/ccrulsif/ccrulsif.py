import numpy as np
import pandas as pd
import math
import networkx as nx

def fit_rulsif(X_ref, X_new, alpha=0.1, n_centers=100, sigma=None, lambda_reg=1e-6):
    """
    Fit an alpha-relative density ratio model g(x) ~ p_new(x)/[alpha*p_new(x)+(1-alpha)*p_ref(x)]
    using Gaussian kernel basis. Returns the learned function handle and the estimated divergence.
    X_ref: reference samples (N_ref x d array)
    X_new: new samples (N_new x d array)
    """
    X_ref = np.asarray(X_ref)
    X_new = np.asarray(X_new)
    n_ref, d = X_ref.shape
    n_new, d = X_new.shape
    
    # Choose kernel centers (random subset of new samples)
    if n_new < n_centers:
        centers = X_new.copy()
    else:
        centers_idx = np.random.choice(n_new, size=n_centers, replace=False)
        centers = X_new[centers_idx]
    m = centers.shape[0]
    
    # Determine kernel bandwidth sigma if not provided
    if sigma is None:
        # heuristic: median distance between centers
        if m > 300:  # sample for efficiency if many centers
            subset = centers[np.random.choice(m, size=300, replace=False)]
        else:
            subset = centers
        # compute pairwise distances
        dists = []
        for i in range(len(subset)):
            for j in range(i+1, len(subset)):
                dists.append(np.linalg.norm(subset[i] - subset[j]))
        sigma = np.median(dists) / math.sqrt(2) if dists else 1.0
        if sigma <= 1e-8:
            sigma = 1.0
    # Gaussian kernel feature maps
    def rbf_feat(X, centers, sigma):
        # compute RBF kernel matrix between X and centers
        X = X[:, None, :]   # shape (N, 1, d)
        C = centers[None, :, :]  # shape (1, m, d)
        dist_sq = np.sum((X - C)**2, axis=2)  # shape (N, m)
        return np.exp(- dist_sq / (2 * sigma**2))
    Phi_ref = rbf_feat(X_ref, centers, sigma)   # (N_ref x m) features for reference data
    Phi_new = rbf_feat(X_new, centers, sigma)   # (N_new x m) features for new data
    # Solve for theta: (alpha * Phi_new^T Phi_new / n_new + (1-alpha)*Phi_ref^T Phi_ref / n_ref + λI) θ 
    #               = Phi_new^T (1/n_new * 1)  (the right side is E_new[φ])
    A = alpha * (Phi_new.T @ Phi_new) / n_new  + (1-alpha) * (Phi_ref.T @ Phi_ref) / n_ref
    b = Phi_new.mean(axis=0)  # (1 x m) mean of features under new
    # Regularize and solve
    A += lambda_reg * np.eye(m)
    theta = np.linalg.solve(A, b)
    # Define the learned ratio function
    def ratio_func(X):
        X = np.atleast_2d(X)
        Phi = rbf_feat(X, centers, sigma)  # (N_test x m)
        return Phi @ theta
    # Estimate alpha-relative Pearson divergence
    w_new = Phi_new @ theta  # g(u,x) on new samples
    w_ref = Phi_ref @ theta  # g(u,x) on reference samples
    PE_est = 0.5 * (alpha * np.mean((w_new - 1)**2) + (1-alpha) * np.mean((w_ref - 1)**2))
    return ratio_func, PE_est

def get_child_parent_sets(graph, include_roots=False):
    """
    Decompose a directed graph into child-parent sets.
    
    Parameters:
    - graph: networkx.DiGraph
        A directed acyclic graph (DAG).
    - include_roots: bool (default False)
        Whether to include root nodes (nodes with no parents) in the output.
        
    Returns:
    - dict: Mapping from each child node to a list of its parent nodes.
    """
    if not isinstance(graph, nx.DiGraph):
        raise ValueError("Graph must be a networkx.DiGraph.")
    
    parent_sets = {node: list(graph.predecessors(node)) for node in graph.nodes()}
    
    if include_roots:
        return parent_sets
    else:
        # Only return nodes that have one or more parents
        return {node: parents for node, parents in parent_sets.items() if parents}

def calibrate_delta(
    ref_df,
    child_parent_dict,
    *,
    alpha        = 0.1,     # same α you use in fit_rulsif
    n_bootstrap  = 250,      # # resamples
    frac_sample  = 0.5,      # fraction of ref data per split
    percentile   = 95,       # percentile → δ
    n_centers    = 80,       # passed through to fit_rulsif
    random_state = None
):
    """
    Return a dict  {(child, tuple(parents)): delta}.

    Each delta_X is the percentile-th quantile of the bootstrap
    distribution of  D_PE,α  when *both* samples come from the
    (unchanged) reference distribution.
    """
    rng = np.random.default_rng(random_state)
    deltas = {}

    for child, parents in child_parent_dict.items():
        cols   = parents + [child]
        ref_xy = ref_df[cols].values
        n_ref  = len(ref_xy)
        m      = int(np.ceil(frac_sample * n_ref))

        boot_div = []
        for _ in range(n_bootstrap):
            # two independent resamples *with* replacement
            idx_q = rng.choice(n_ref, m, replace=True)
            idx_p = rng.choice(n_ref, m, replace=True)
            X_q   = ref_xy[idx_q]   # acts as "reference"
            X_p   = ref_xy[idx_p]   # acts as "new"
            # estimate divergence under null = 0 drift
            _, div = fit_rulsif(X_q, X_p,
                                alpha=alpha,
                                n_centers=n_centers)
            boot_div.append(div)

        delta = np.percentile(boot_div, percentile)
        deltas[child] = float(delta)

    return deltas


if __name__ == "__main__":
    # Reference data generation (baseline)
    for N_ref in [50, 100, 250, 500]:
        print(f"{N_ref} reference points:\n")
        A_ref = np.random.normal(0, 1, size=N_ref)
        B_ref = np.random.normal(0, 1, size=N_ref)    # B | A baseline
        C_ref = A_ref + B_ref + np.random.normal(0, 1, size=N_ref)      # C | B baseline
        D_ref = C_ref + np.random.normal(0, 1, size=N_ref)
        E_ref = B_ref + np.random.normal(0, 1, size=N_ref)
        F_ref = A_ref + np.random.normal(0, 1, size=N_ref)

        ref_df = pd.DataFrame([A_ref, B_ref, C_ref, D_ref, E_ref, F_ref], ['A', 'B', 'C', 'D', 'E', 'F']).T

        # np.array(A_ref, B_ref, C_ref, D_ref, E_ref)

        # New data generation (after shift)
        N_new = 500
        confounder = np.random.normal(0, 1, size=N_new) 
        A_new = np.random.normal(0, 1, size=N_new)              # assume A's distribution unchanged
        B_new = np.random.normal(0, 1, size=N_new)              # B | A shifted (offset +1)
        C_new = 2*A_new + B_new + np.random.normal(0, 1, size=N_new)        # C | B (same formula as baseline)
        D_new = C_new + np.random.normal(0, 1, size=N_new)
        E_new = B_new + np.random.normal(0, 1, size=N_new)
        F_new = confounder + A_new + np.random.normal(0, 1, size=N_new)

        new_df = pd.DataFrame([A_new, B_new, C_new, D_new, E_new, F_new], ['A', 'B', 'C', 'D', 'E', 'F']).T
        
        # Create a sample DAG
        G = nx.DiGraph()
        G.add_edges_from([
            ("A", "C"),
            ("A", "F"),
            ("B", "C"),
            ("C", "D"),
            ("B", "E"),
        ])

        # Get all child-parent sets, excluding roots
        child_parent = get_child_parent_sets(G)
        print("Child -> Parents (excluding roots):")
        for child, parents in child_parent.items():
            cols = np.concatenate([[child], parents])
            
            X_ref = ref_df[cols]
            
            for batch_size in [10, 25, 50, 100, 200, 500]:
                X_new = new_df[cols].iloc[:batch_size]
                ratio, divergence = fit_rulsif(X_ref, X_new, alpha=0.1)
                print(f"{batch_size} samples \t {child} | {parents} divergence estimate: {divergence:.3f}")
                
            print('------------')
        
        print("\n")