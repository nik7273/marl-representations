import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import supersuit as ss
from pettingzoo.butterfly import pistonball_v4
from pettingzoo.butterfly import cooperative_pong_v2
from dqn import DQN
from cpc import CPC
from agent import Agent, ObsBuffer
from memory import ReplayMemory
from tqdm import tqdm

def get_args():
    seed = torch.randint(0, 10000, ())
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",
                        type=str,
                        default=None,
                        help="PettingZoo environment to use. Choose between `pistonball` and `pong`.")
    parser.add_argument("--mode",
                        type=str,
                        default="train",
                        help="`train` or `eval` mode")
    parser.add_argument("--num-timesteps",
                        type=int,
                        default=12,
                        help="Number of timesteps to consider for central model")
    parser.add_argument("--batch-size",
                        type=int,
                        default=8,
                        help="Batch size for central model training")
    parser.add_argument("--adam-eps",
                        type=float,
                        default=1.5e-4,
                        help="Adam epsilon") 
    parser.add_argument("--seq-len",
                        type=str,
                        default=20480,
                        help="Total sequence length for central model")
    parser.add_argument("--buffer-size",
                        type=int,
                        default=int(1e5),
                        help="Experience replay memory capacity")
    parser.add_argument("--max-horizon",
                        type=int,
                        default=int(2e5),
                        help="Number of training steps (4x number of frames)")
    parser.add_argument("--learn-start",
                        type=int,
                        default=int(1000),
                        help="Number of steps before starting training")
    parser.add_argument("--priority-weight",
                        type=float,
                        default=0.4,
                        help="Initial prioritised experience replay importance sampling weight")
    parser.add_argument("--eval-buffer-size",
                        type=int,
                        default=500,
                        help="number of evaluations for validation of update parameters")
    parser.add_argument("--update-period",
                        type=int,
                        default=1,
                        help="Frequency with which to update parameters")
    parser.add_argument("--multi-step",
                        type=int,
                        default=20,
                        help="Number of steps for multi-step return")
    parser.add_argument("--V-min",
                        type=float,
                        default=-10,
                        help="Minimum of value distribution support")
    parser.add_argument("--V-max",
                        type=float,
                        default=10,
                        help="Maximum of value distribution support")
    parser.add_argument("--atoms",
                        type=int,
                        default=51,
                        help="Discretised size of value distribution")
    parser.add_argument("--architecture",
                        type=str,
                        default="data-efficient",
                        choices=["canonical", "data-efficient"],
                        help="Network architecture")
    parser.add_argument("--history-length",
                        type=int,
                        default=4,
                        help="Number of consecutive states processed")
    parser.add_argument("--hidden-size",
                        type=int,
                        default=256,
                        help="Network hidden size")
    parser.add_argument("--noisy-std",
                        type=float,
                        default=0.1,                        
                        help="Initial standard deviation of noisy linear layers")
    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.0001,                        
                        help="Learning")
    parser.add_argument("--enable-cudnn",
                        action="store_true",
                        help="Enable cuDNN (faster but nondeterministic)")
    parser.add_argument("--discount",
                        type=float,
                        default=0.99,
                        help="Discount factor")
    parser.add_argument("--priority-exponent",
                        type=float,
                        default=0.5,
                        help="Prioritised experience replay exponent (originally denoted α)")
    parser.add_argument("--seed",
                        type=int,
                        default=seed,
                        help="Random seed")
    parser.add_argument("--reward-clip",
                        type=int,
                        default=0,
                        help="Reward clipping (0 to disable)")
    parser.add_argument("--replay-frequency",
                        type=int,
                        default=1,
                        help="Frequency of sampling from memory")
    parser.add_argument('--norm-clip',
                        type=float,
                        default=10,                        
                        help='Max L2 norm for gradient clipping')
    parser.add_argument("--target-update",
                        type=int,
                        default=int(2e3),
                        help="Number of steps after which to update target network")
    parser.add_argument("--results-dir",
                        type=str,
                        default="results",
                        help="directory where results should be placed")
    args = parser.parse_args()

    # device
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(1, 10000))
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        torch.cuda.manual_seed(np.random.randint(1, 10000))
        torch.backends.cudnn.enabled = args.enable_cudnn
    else:
        args.device = torch.device("cpu")

    return args

def change_reward_fn(reward):
    if reward == 200:
        return -1
    return reward/100

def init_env(args):    
    env = None
    if args.env == 'pistonball':
        env = pistonball_v4.env(n_pistons=20,
                                local_ratio=0,
                                time_penalty=-0.1,
                                continuous=False,
                                random_drop=True,
                                random_rotate=True,
                                ball_mass=0.75,
                                ball_friction=0.3,
                                ball_elasticity=1.5,
                                max_cycles=125)
    elif args.env == "pong":
        env = cooperative_pong_v2.env(ball_speed=9,
                                      left_paddle_speed=12,
                                      right_paddle_speed=12,
                                      cake_paddle=True,
                                      max_cycles=900,
                                      bounce_randomness=False)
    else:
        raise Exception("Bad argument: --env")

    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    env = ss.sticky_actions_v0(env, repeat_action_probability=0.25)
    env = ss.color_reduction_v0(env, mode="full")
    env = ss.resize_v0(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.frame_skip_v0(env, 4)
    env = ss.dtype_v0(env,np.float32)
    env = ss.normalize_obs_v0(env)
    env = ss.clip_reward_v0(env, -1, 1)
    env = ss.reward_lambda_v0(env, change_reward_fn)
    # env = ss.pettingzoo_env_to_vec_env_v0(env)
    # env = ss.concat_vec_envs_v0(env, 8, num_cpus=4, base_class='stable_baselines3')
    env.reset()

    return env

def init_models(args, env):
    models = {} # agent models
    memory = {} # execution memory
    val_memory = {} # validation memory

    central_rep = CPC(args.num_timesteps, args.batch_size, args.seq_len)

    for agent in env.agents:
        model = DQN(args, env.action_spaces[agent].n)
        models[agent] = Agent(args, env.action_spaces[agent].n, model, central_rep)
        
        # Construct working memory
        memory[agent] = ReplayMemory(args, args.buffer_size)

        # Construct validation memory
        val_memory[agent] = ReplayMemory(args, args.eval_buffer_size)

    return models, memory, val_memory, central_rep

def prefill_buffer(args, env, val_memory):
    # prefill buffer before training
    t, converged = 0, False
    while not converged:
        i = 0
        env.reset()
        for agent in env.agent_iter(args.eval_buffer_size):
            if i % len(env.agents):
                t += 1
                if t > args.eval_buffer_size:
                    converged=True
                    break
            observation, reward, done, info = env.last()
            action = np.random.randint(0,env.action_spaces[agent].n) if not done else None
            env.step(action)
            val_memory[agent].append(torch.tensor(observation), None, None, done)
            i += 1


def set_mode(models, mode="train"):
    if mode == "train":
        for agent in models.keys():
            models[agent].train()
    elif mode == "eval":
        for agent in models.keys():
            models[agent].eval()
    else:
        raise Exception("mode does not exist")
