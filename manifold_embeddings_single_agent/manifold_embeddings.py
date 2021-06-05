"""
Custom reimplementation of paper:
Manifold Embeddings for Model-Based RL under Partial Observability
Bush et al. 2009
"""

import numpy as np
from scipy.signal import find_peaks

def modeling_phase():
    pass

def learning_phase():
    pass

def spectral_parameter_selection(obs, embed_dim):
    """
    obs: Tensor
    embed_dim: int
    """
    K = min(embed_dim, obs.shape[1])
    singular_values = []
    obs_len = obs.shape[1] # \widetilde{S}
    for t_min in range(embed_dim, obs_len):
        S_E = np.zeros(obs_len-t_min, embed_dim)
        tau = t_min / (embed_dim - 1)
        for t in range(t_min, obs_len):
            S_E_t = np.array([obs[t - i * tau] for i in range(embed_dim)])
            S_E[t] = S_E_t

        u, s, vh = np.linalg.svd(S_E)
        
        singular_values.append(s)
 
    singular_values = np.array(singular_values)
    # determining embedding parameters
    approx_t_min = None # T_min value of first local maxima of second singular values of T_min
    approx_embed_dim = None # number of nontrivial singular values

    second_singular = singular_values[:, 1] # check correct dim
    local_maxima, _ = find_peaks(second_singular)
    approx_t_min = embed_dim + local_maxima[0]

    long_term_trend = np.mean(seq[-embed_dim:])
    approx_embed_dim = np.ones_like(second_singular[second_singular<long_term_trend]).sum()
    
    return approx_t_min, approx_embed_dim

def run():
    pass
