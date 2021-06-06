"""
Custom reimplementation of paper:
Manifold Embeddings for Model-Based RL under Partial Observability
Bush et al. 2009
"""

import numpy as np
import gym
from scipy.signal import find_peaks
from env import DiscretizeEnv
from agents import RandomAgent

def random_walk():
    """
    Paper example uses MountainCar, so we'll use that here
    """
    STATE_DIM = 400
    ACTION_DIM = 3
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env = DiscretizeEnv(env_example,num_bins=20,show_info=True)
    agent = RandomAgent()
    max_timesteps = 10000
    states, actions = [], []
    
    state = env_example.reset()
    t = 0
    while episode_timesteps < max_timesteps: 
        t += 1
        action = agent.select_action(state)
        next_state,reward,done  = env.step(action)
        next_action = agent.select_action(next_state)
        done_bool = float(done) if t < env._max_episode_steps else -1
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    return states, actions, rewards

def modeling_phase(env, init_steps):
    # Record partially observable system under random control
    obs, actions, rewards = random_walk()
    # Spectral embedding
    embed_init = env.state_space * 2 + 1
    approx_t, approx_e = spectral_parameter_selection(obs, embed_init)
    # construct embedding vectors
    embeddings = gen_embedding(obs, approx_t, approx_e)

    # TODO: Define local model?
    
    return embeddings

def learning_phase():
    pass

def gen_embedding(obs, t_min, embed_dim):
    obs_len = obs.shape[1] # \widetilde{S}
    S_E = np.zeros(obs_len-t_min, embed_dim)
    tau = t_min / (embed_dim - 1)
    for t in range(t_min, obs_len):
        S_E_t = np.array([obs[t - i * tau] for i in range(embed_dim)])
        S_E[t] = S_E_t

    return S_E

def spectral_parameter_selection(obs, embed_dim):
    """
    obs: Tensor
    embed_dim: int
    """
    K = min(embed_dim, obs.shape[1])
    singular_values = []
    for t_min in range(obs_len, embed_dim):
        S_E = gen_embedding(obs, t_min, embed_dim)
        u, s, vh = np.linalg.svd(S_E)        
        singular_values.append(s)
 
    singular_values = np.array(singular_values)
    # determining embedding parameters
    approx_t_min = None # T_min value of first local maxima of second singular values of T_min
    approx_embed_dim = None # number of nontrivial singular values

    second_singular = singular_values[:, 1] # check correct dim
    local_maxima, _ = find_peaks(second_singular)
    approx_t_min = embed_dim + local_maxima[0]

    long_term_trend = np.mean(singular_values[approx_t_min, -embed_dim:])
    approx_embed_dim = np.ones_like(second_singular[second_singular<long_term_trend]).sum()
    
    return approx_t_min, approx_embed_dim

def local_model(obs, actions, rewards, t_min, embed_dim):
    S_E = gen_embedding(obs, t_min, embed_dim)
    # TODO
    pass
