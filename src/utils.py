import numpy as np


def normalize(states_flattened):
    """
    Normalize states to have mean 0 and std 1.
    But I don't think this was ever used.

    Parameters
    ----------
    states_flattened : np.ndarray
        States to be normalized, shape (T*B, H*W*C)

    Returns
    -------
    normalized : np.ndarray
        Normalized states, shape (T*B, H*W*C)
    """
    mean = states_flattened.mean(axis=0)
    std = states_flattened.std(axis=0)

    # Avoid division by zero
    std_safe = np.where(std < 1e-8, 1.0, std)

    normalized = (states_flattened - mean) / std_safe

    return normalized
