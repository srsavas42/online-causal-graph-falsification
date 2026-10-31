import numpy as np
import pandas as pd


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