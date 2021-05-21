import os
import argparse
import torch
import torch.nn as nn
import supersuit as ss
from pettingzoo.butterfly import pistonball_v4
from pettingzoo.butterfly import cooperative_pong_v2
from ppo import PPO
from dqn import DQN
from cpc import CPC
from stable_baselines3.ppo import CnnPolicy
from agent import Agent, ObsBuffer, set_mode
from memory import ReplayMemory

def main():
    seed = torch.randint(0, 10000, ())
    parser = argparse.ArgumentParser()
    # parser.add_argument("--num_agents", type=int, default=None, help="Number of agents in simulation")
    parser.add_argument("--env",
                        type=str,
                        default=None,
                        help="PettingZoo environment to use. Choose between `pistonball` and `pong`.")
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
    args = parser.parse_args()
    
    env = None
    if args.env == 'pistonball':
        env = pistonball_v4.parallel_env(n_pistons=20,
                                         local_ratio=0,
                                         time_penalty=-0.1,
                                         continuous=True,
                                         random_drop=True,
                                         random_rotate=True,
                                         ball_mass=0.75,
                                         ball_friction=0.3,
                                         ball_elasticity=1.5,
                                         max_cycles=125)
    elif args.env == "pong":
        env = cooperative_pong_v2.parallel_env(ball_speed=9,
                                               left_paddle_speed=12,
                                               right_paddle_speed=12,
                                               cake_paddle=True,
                                               max_cycles=900,
                                               bounce_randomness=False)
    else:
        raise Exception("bad env argument")
    
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.resize_v0(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)

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

    models = {} # agent models
    memory = {} # execution memory
    val_memory = {} # validation memory

    central_rep = CPC(args.num_timesteps, args.batch_size, args.seq_len)

    for agent in env.agents:
        # model = PPO(CnnPolicy,
        #             env,
        #             verbose=3,
        #             gamma=0.99,
        #             n_steps=125,
        #             ent_coef=0.01,
        #             learning_rate=0.00025,
        #             vf_coef=0.5,
        #             max_grad_norm=0.5,
        #             gae_lambda=0.95,
        #             n_epochs=4,
        #             clip_range=0.2,
        #             clip_range_vf=1) # TODO: check and fix
        model = DQN(args, env.action_spaces[agent].n)
        models[agent] = Agent(args, env.action_spaces[agent].n, model, central_rep)
        
        memory[agent] = ReplayMemory(args, args.buffer_size)

        priority_weight_increase = (1 - args.priority_weight) / (
            args.max_horizon - args.learn_start
        )

        # Construct validation memory
        val_memory[agent] = ReplayMemory(args, args.eval_buffer_size)


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

    # Training
    set_mode(models)
    t, converged = 0, False
    progress_bar = tqdm(total=args.max_horizon)
    obs_buffer = ObsBuffer(args.num_timesteps)
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

            observation, reward, done, info = env.last()

            action = models[agent].act(torch.tensor(observation))
            if not done:
                env.step(action)
            else:
                env.step(None)

            memory[agent].append(torch.tensor(observation), action, reward, done)

            if t >= args.learn_start:
                memory[agent].priority_weight = min(memory[agent].priority_weight + priority_weight_increase, 1)

                if i % args.update_period:
                    obs_buffer.append(observation)
                    # Train individual agent
                    models[agent].update_params(memory[agent], central_rep.get_loss())
                    # Train central representation
                    central_rep.update_params(observation_buffer)


if __name__ == '__main__':
    main()
