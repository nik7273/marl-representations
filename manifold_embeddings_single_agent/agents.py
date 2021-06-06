import numpy as np
from env import DiscretizeEnv

class Random_Agent(object):
    def __init__(self):
        print("random agent")
    def select_action(self,state):
        return int(np.random.choice(3, 1))

env_name = 'MountainCar-v0'
env_example = gym.make(env_name)
env_example = DiscretizeEnv(env_example,num_bins=20,show_info=True)
agent = Random_Agent()

for ii in range(3):
    state = env_example.reset()
    ep_reward = 0
    episode_timesteps = 0
    while True: 
        episode_timesteps += 1
        action = agent.select_action(state)
        next_state,reward,done  = env_example.step(action)
        next_action = agent.select_action(next_state)     
        done_bool = float(done) if episode_timesteps < env_example._max_episode_steps else 0
        state = next_state
        ep_reward += reward
        if done:
            print(f'episode:{ii},reward:{ep_reward}')
            break

class Q_learning(object):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_table = np.zeros((state_dim, action_dim))
        self.discount_factor = 0.98
        self.learning_rate = 0.4
        self.learning_rate_decay = 0.999
        self.epsilon = 0.3
        self.epsilon_decay = 0.95

    def train(self,state_list,action_list,reward_list,done_list):
        '''
        done_list[i](ith element): -1 terminal(reach the goal); 1: termianl(max length) 0: otherwise
        '''
        curr_state = state_list[0]
        for i, next_state in enumerate(state_list[1:]):
            action = action_list[i]
            td = self.discount_factor*np.max(self.q_table[next_state]) - self.q_table[curr_state, action]
            self.q_table[curr_state, action] = self.q_table[curr_state, action] + self.learning_rate * (reward_list[i] + td)
            curr_state = next_state
        if done_list[-1] == -1:
          self.learning_rate *= self.learning_rate_decay
          self.epsilon *= self.epsilon_decay

    def select_action(self,state):
        return np.argmax(self.q_table[state]) if random.uniform(0,1) >= self.epsilon else np.random.randint(0, self.action_dim)
