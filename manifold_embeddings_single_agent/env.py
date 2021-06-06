import numpy as np
import matplotlib.pyplot as plt
import random
import gym
import copy
from gym import Env

class DiscretizeEnv(Env):
    def __init__(self,env,num_bins,show_info=False):
      self.env = env
      self.num_bins = num_bins
      self.action_space = self.env.action_space
      self.observation_space = self.env.observation_space
      self.low = self.env.observation_space.low
      self.high = self.env.observation_space.high
      self._max_episode_steps = env._max_episode_steps 
      self.discrete_states = [
            np.linspace(self.low[0], self.high[0], num=(num_bins + 1))[1:-1],
            np.linspace(self.low[1], self.high[1], num=(num_bins + 1))[1:-1],
        ]
      self.action_dim = 3
      self.state_dim = self.num_bins ** len(self.discrete_states)
      if show_info:
          print(f'State space: Position Range:({self.low[0]:.2f},{self.high[0]:.2f}) Velocity Range:({self.low[1]:.2f},{self.high[1]:.2f})')
          print(f'State space shape: {env.observation_space.shape}')
          print(f'Action space shape: {env.action_space}')
          print(f'Dimension of state after discreting: {self.state_dim}')
    
    def get_index(self,state):
      state_ind = sum(np.digitize(feature, self.discrete_states[i]) * (self.num_bins ** i)
                    for i, feature in enumerate(state))
      return state_ind

    def reset(self):
      state = self.env.reset()
      return self.get_index(state)

    def render(self):
      self.env.render()

    def step(self, action):
      next_state,reward,done,_ = self.env.step(action)
      if next_state[0] >= 0.5:
          reward = 0 
      next_state_ind = self.get_index(next_state)
      return next_state_ind,reward,done


def eval_policy(agent,bins,eval_episodes=10):
    env_name = 'MountainCar-v0'
    eval_env = gym.make(env_name)
    eval_env = DiscretizeEnv(eval_env,bins)
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = agent.select_action(state)
            state, reward, done = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

def return_episode(agent,bins):
    env_name = 'MountainCar-v0'
    gen_env = gym.make(env_name)
    #gen_env._max_episode_steps = 2000
    gen_env = DiscretizeEnv(gen_env,bins)
    state_list = []
    action_list = []
    reward_list = []
    done_list = []
    
    state, done = gen_env.reset(), False
    state_list.append(state)
    episode_timesteps = 0
    while not done:
        episode_timesteps += 1
        action = agent.select_action(state)
        state, reward, done = gen_env.step(action)
        done_bool = float(done) if episode_timesteps < gen_env._max_episode_steps else 0
        action_list.append(action)
        state_list.append(state)
        reward_list.append(reward)
        if done:
            if not done_bool:
                done_list.append(1)
            else:
                done_list.append(-1)
        else:
            done_list.append(0)
    return state_list,action_list,reward_list,done_list
