import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import supersuit as ss
from tqdm import tqdm
from pettingzoo.butterfly import pistonball_v4
from pettingzoo.butterfly import cooperative_pong_v2
from stable_baselines3.ppo import CnnPolicy
from ppo import PPO
from dqn import DQN
from cpc import CPC
from cpc import mesh_inputs
from agent import Agent, ObsBuffer
from memory import ReplayMemory
from utils import get_args, init_env, init_models, prefill_buffer, set_mode
from test_model import test_model
from optimizer import ScheduledOptim

# TODO: Need to call mesh_inputs function somewhere here
def train(args, env, models, memory, val_memory, central_rep, central_rep_optimizer):
    # Training
    set_mode(models, mode="train")
    t, converged = 0, False
    progress_bar = tqdm(total=args.max_horizon)
    obs_buffer = ObsBuffer(args.num_timesteps)

    priority_weight_increase = (1 - args.priority_weight) / (
        args.max_horizon - args.learn_start
    )
    while not converged:
        env.reset()
        i = 0
        for agent in env.agent_iter():
            i += 1
            if i % len(env.agents):
                progress_bar.update(1)
                t += 1
                if t > args.max_horizon:
                    converged = True
                    break
            if t % args.replay_frequency == 0:
                models[agent].reset_noise()  # Draw a new set of noisy weights

            observation, reward, done, info = env.last()

            action = models[agent].act(torch.tensor(observation))
            if t < args.learn_start:
                action = np.random.randint(0,env.action_spaces[agent].n)
            if not done:
                env.step(action)
            else:
                env.step(None)
            if args.reward_clip > 0:
                reward = max(
                    min(reward, args.reward_clip), -args.reward_clip
                )  # Clip rewards

            memory[agent].append(torch.tensor(observation), action, reward, done)

            if t >= args.learn_start:
                memory[agent].priority_weight = min(memory[agent].priority_weight + priority_weight_increase, 1)

                if i % args.update_period == 0:
                    obs_buffer.append(observation) # here is where mesh_inputs() would be called
                    # Train individual agent
                    models[agent].update_params(memory[agent], central_rep.get_loss())
                    # Train central representation
                    central_rep.update_params(obs_buffer, central_rep_optimizer)
                # Update target network
                if t % args.target_update == 0:
                    models[agent].update_target_net()

def evaluate(args, env, models, val_memory, central_rep):
    # init metrics map
    metrics = {'steps': [], 'rewards': {}, 'Qs': {}, 'best_avg_reward': {}}
    for agent in env.agents:
        metrics['rewards'][agent] = []
        metrics['Qs'][agent] = []
        metrics['best_avg_reward'][agent] = -float("inf")

    # evaluate
    set_mode(models, mode="eval")  # Set DQN (online network) to evaluation mode
    avg_reward, avg_Q = test_model(
        args, 0, models, val_memory, metrics, args.results_dir, evaluate=True
    )  # Test
    print("Avg. reward: " + str(avg_reward) + " | Avg. Q: " + str(avg_Q))

def main():
    args = get_args()

    env = init_env(args)

    models, memory, val_memory, central_rep, central_rep_optimizer = init_models(args, env)
    
    prefill_buffer(args, env, val_memory)

    if args.mode == "train":
        train(args, env, models, memory, val_memory, central_rep, central_rep_optimizer)
    elif args.mode == "eval":
        evaluate(args, env, models, val_memory, central_rep)
    else:
        raise Exception("Bad argument: --mode")
                    
if __name__ == '__main__':
    main()
