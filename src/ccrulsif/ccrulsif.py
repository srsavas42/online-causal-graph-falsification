import numpy as np
import pandas as pd
import math
import networkx as nx

def fit_rulsif(X_ref, X_new, alpha=0.1, n_centers=100, sigma=None, lambda_reg=1e-6, *, rng = np.random.Generator | None = None):
    """
    Fit an alpha-relative density ratio model g(x) ~ p_new(x)/[alpha*p_new(x)+(1-alpha)*p_ref(x)]
    using Gaussian kernel basis. Returns the learned function handle and the estimated divergence.
    X_ref: reference samples (N_ref x d array)
    X_new: new samples (N_new x d array)
    """
    
    rng = np.random.default_rng() if rng is None else rng
    
    X_ref = np.asarray(X_ref)
    X_new = np.asarray(X_new)
    n_ref, d = X_ref.shape
    n_new, d = X_new.shape
    
    # Choose kernel centers (random subset of new samples)
    if n_new < n_centers:
        centers = X_new.copy()
    else:
        centers_idx = rng.choice(n_new, size=n_centers, replace=False)
        centers = X_new[centers_idx]
    m = centers.shape[0]
    
    # Determine kernel bandwidth sigma if not provided
    if sigma is None:
        # heuristic: median distance between centers
        if m > 300:  # sample for efficiency if many centers
            subset = centers[rng.choice(m, size=300, replace=False)]
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
        X = X[:, None, :]   
        C = centers[None, :, :]  
        dist_sq = np.sum((X - C)**2, axis=2)  
        return np.exp(- dist_sq / (2 * sigma**2))
    
    Phi_ref = rbf_feat(X_ref, centers, sigma)
    Phi_new = rbf_feat(X_new, centers, sigma)
    
    # Solve for theta: (alpha * Phi_new^T Phi_new / n_new + (1-alpha)*Phi_ref^T Phi_ref / n_ref + Lambda * I) Theta 
    #               = Phi_new^T (1/n_new * 1)  (the right side is E_new[Phi])
    A = alpha * (Phi_new.T @ Phi_new) / n_new  + (1-alpha) * (Phi_ref.T @ Phi_ref) / n_ref
    b = Phi_new.mean(axis=0)  
    
    A += lambda_reg * np.eye(m)
    theta = np.linalg.solve(A, b)
    
    def ratio_func(X):
        X = np.atleast_2d(X)
        Phi = rbf_feat(X, centers, sigma)  
        return Phi @ theta
    
    # Estimate alpha-relative Pearson divergence
    w_new = Phi_new @ theta  
    w_ref = Phi_ref @ theta  
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
    alpha        = 0.1,      # same alpha from fit_rulsif
    n_bootstrap  = 250,      # # resamples
    frac_sample  = 0.5,      # fraction of ref data per split
    percentile   = 95,       # percentile -> delta
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
            # two independent resamples with replacement
            idx_q = rng.choice(n_ref, m, replace=True)
            idx_p = rng.choice(n_ref, m, replace=True)
            X_q   = ref_xy[idx_q]   # acts as reference
            X_p   = ref_xy[idx_p]   # acts as new
            
            # estimate divergence under null = 0 drift
            _, div = fit_rulsif(X_q, X_p,
                                alpha=alpha,
                                n_centers=n_centers,
                                rng=rng)
            boot_div.append(div)

        delta = np.percentile(boot_div, percentile)
        deltas[child] = float(delta)

    return deltas
