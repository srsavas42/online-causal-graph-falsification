import numpy as np
import pandas as pd

def synthetic_data(
    base_matrix,
    N=1000,
    noise=1.0,
    baseline=0.0,
    
    # Change point:
    change_points=None,         # list of tuples: (time_index, new_matrix)
    transition_type='abrupt',   # 'abrupt' or 'gradual'
    transition_duration=0,      # how long to transition between graphs if gradual
    
    # Flexible generation parameters:
    nonlinear=False,            # if True, use a nonlinear relationship between variables
    nonlinear_func=None,        # nonlinear function to use
    include_lags=False,         # if True, create lagged causal relationships (currently 1, future use lag_order)
    # lag_order=1,              # NOT IMPLEMENTED
    
    # Noise and distribution shifts:
    heteroscedastic=False,      # if True, use a time-varying noise scale
    noise_function=None,        # a function of time (t) that adds noise
    distribution_shift=False,   # if True, shift the mean of the noise over time
    distribution_shift_function=None,  # function of time (t) to shift the mean using a particular function
    
    # Latent confounders:
    n_latent=0,                 # number of latent confounders to include
    latent_strength=0.5,        # effect of latent confounders
    
    # Reproducibility:
    seed=None
):
    """
    Generate synthetic (causal) time series data with options for change points,
    nonlinear effects, lag dependence, heteroscedastic noise, latent confounding,
    and interventions.
    
    Parameters:
        base_matrix: np.ndarray, shape (num_vars, num_vars)
            The base “cause” matrix. Must be square.
        N: int
            Number of time steps (samples) to generate.
        noise: float
            Base noise standard deviation.
        baseline: float or np.array
            Constant offset added at each time step.
        change_points: list of tuples, optional
            Each tuple (t, new_matrix) indicates that at time t the causal
            matrix will change to new_matrix (abruptly or gradually).
        transition_type: str, 'abrupt' or 'gradual'
            How to move from one causal regime to the next.
        transition_duration: int
            If using a gradual transition, the number of steps over which to interpolate.
        nonlinear: bool
            Whether to apply a nonlinear transformation (after linear computation).
        nonlinear_func: callable, optional
            A function to apply to the computed vector (e.g., np.tanh). If None and nonlinear
            is True, np.tanh is used.
        include_lags: bool
            If True, simulate a time-series with lag dependence
            Otherwise, each time point is an independent draw from the equilibrium matrix.
        lag_order: int
            (NOT IMPLEMENTED)
        heteroscedastic: bool
            If True, the noise scale is allowed to change over time using noise_function.
        noise_function: callable, optional
            A function f(t) that returns a multiplier for the noise at time t.
        distribution_shift: bool
            If True, the mean of the noise shifts over time.
        distribution_shift_function: callable, optional
            A function f(t) that returns the noise mean at time t.
        n_latent: int
            Number of latent (unobserved) variables to include.
        latent_strength: float
            Multiplier for the latent confounder effect.
        seed: int, optional
            Random seed for reproducibility.
          
    Returns:
        df: pandas.DataFrame
            A DataFrame with columns named Var0, Var1, …, Var(num_vars-1).
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    num_vars = base_matrix.shape[0]
    
    if n_latent > 0:
        # generate random loadings for each observed variable.
        latent_loadings = np.random.uniform(low=0.5, high=1.5, size=(num_vars, n_latent))
    else:
        latent_loadings = None
    
    X = np.zeros((N, num_vars))
    
    # For a lagged (recursive) simulation, need an initial condition
    if include_lags:
        # Set the initial observation
        X[0, :] = np.random.normal(loc=0.0, scale=1.0, size=num_vars)
        start_index = 1
    else:
        start_index = 0
    
    # Organize change points
    if change_points is None:
        change_points = []
    else:
        change_points = sorted(change_points, key=lambda cp: cp[0])
    
    # Set the current causal matrix
    current_matrix = base_matrix.copy()
    next_cp_index = 0
    
    # For gradual transitions:
    in_transition = False
    transition_start = None
    transition_end = None
    prev_matrix = None
    target_matrix = None
    
    # Loop over time steps
    for t in range(start_index, N):
        # Check if it is time to start a change point
        if next_cp_index < len(change_points) and t == change_points[next_cp_index][0]:
            if transition_type == 'gradual' and transition_duration > 0:
                in_transition = True
                transition_start = t
                transition_end = t + transition_duration
                prev_matrix = current_matrix.copy()
                target_matrix = change_points[next_cp_index][1]
            else:
                current_matrix = change_points[next_cp_index][1]
            next_cp_index += 1
        
        # If in a gradual transition, interpolate between the matrices
        if in_transition:
            if t < transition_end:
                alpha = (t - transition_start) / transition_duration
                current_matrix = (1 - alpha) * prev_matrix + alpha * target_matrix
            else:
                in_transition = False
                current_matrix = target_matrix
        
        # Determine the noise
        if heteroscedastic and noise_function is not None:
            noise_scale = noise * noise_function(t)
        else:
            noise_scale = noise
        
        # Determine a possible mean shift
        if distribution_shift and distribution_shift_function is not None:
            error_mean = distribution_shift_function(t)
        else:
            error_mean = 0.0
        
        # Generate the error vector
        error_vec = np.random.normal(loc=error_mean, scale=noise_scale, size=num_vars)
        
        # Generate latent effects (if any)
        if n_latent > 0:
            latent_factors = np.random.normal(loc=0.0, scale=1.0, size=n_latent)
            latent_effect = latent_strength * (latent_loadings @ latent_factors)
        else:
            latent_effect = np.zeros(num_vars)
        
        # Compute X[t] depending on lags
        if include_lags:
            # Simulate lag of 1
            # FUTURE: Use lag order parameter
            X[t, :] = (baseline if np.isscalar(baseline)
                       else np.array(baseline)) + current_matrix @ X[t-1, :] + error_vec + latent_effect
        else:
            I = np.eye(num_vars)
            I_minus_B = I - current_matrix
            if np.linalg.det(I_minus_B) == 0:
                raise ValueError(f"(I - B) is singular at time {t}!")
            inv_matrix = np.linalg.inv(I_minus_B)
            X[t, :] = inv_matrix @ ((baseline if np.isscalar(baseline)
                                      else np.array(baseline)) + error_vec + latent_effect)
        
        # Apply nonlinearity
        if nonlinear:
            if nonlinear_func is not None:
                X[t, :] = nonlinear_func(X[t, :])
            else:
                X[t, :] = np.tanh(X[t, :])

    col_names = [f"Var{i}" for i in range(num_vars)]
    X_cpu = np.asnumpy(X)   
    
    df = pd.DataFrame(X_cpu, columns=col_names)
    return df


def generate_datasets(test_configs, base_matrix):
    """Parameterized function to easily generate new time-series datasets with various changes 
    by using easy configurations.

    Args:
        test_configs Dictionary: 
            A dictionary containing a set of parameters to use in generating a new dataset.
        base_matrix np.ndarray, shape (num_vars, num_vars): 
            The starting causal relationship between all variables.

    Returns:
        df: pandas.DataFrame
            A DataFrame with columns named Var0, Var1, …, Var(num_vars-1).
    """
    datasets = []

    for i, config in enumerate(test_configs):
        df = synthetic_data(
            base_matrix,
            N=config['size'] if 'size' in config.keys() else 1000,
            noise=config['noise'] if 'noise' in config.keys() else 1,
            baseline=0.0,
            change_points=config["change_points"] if 'change_points' in config.keys() else None,
            transition_type=config["abrupt"] if 'transition_type' in config.keys() else 'abrupt',
            transition_duration=0,
            nonlinear=config['nonlinear'] if 'nonlinear' in config.keys() else False,
            include_lags=config['lags'] if 'lags' in config.keys() else False,
            heteroscedastic=config["heteroscedastic"] if 'heteroscedastic' in config.keys() else False,
            noise_function=lambda t: 1 + 0.001*t if 'heteroscedastic' in config.keys() else None,
            n_latent=config["n_latent"] if 'n_latent' in config.keys() else 0,
            latent_strength=0.5  if 'n_latent' in config.keys() else 0,
            interventions=config["interventions"] if 'interventions' in config.keys() else None,
            seed=None
        )
        datasets.append((df, config))

    return datasets