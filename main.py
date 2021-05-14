import os
import argparse
import torch
import torch.nn as nn
import supersuit as ss
from pettingzoo.butterfly import pistonball_v4
from pettingzoo.butterfly import cooperative_pong_v2
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy

def main():
    seed = torch.randint(10000)
    parser = argparse.ArgumentParser()
    # parser.add_argument("--num_agents", type=int, default=None, help="Number of agents in simulation")
    parser.add_argument("--env",
                        type=str,
                        default=None,
                        help="PettingZoo environment to use. Choose between `pistonball` and `pong`.")
    # TODO: Add all the running arguments

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
    env = ss.pettingzoo_env_to_vec_env_v0(env)
    env = ss.concat_vec_envs_v0(env, 8, num_cpus=4, base_class='stable_baselines3')
    env.reset()

    # Check necessity of all of this
    metrics = {'steps': [], 'rewards': {}, 'q_values': {}, 'best_avg_reward': {}}
    for agent in env.agents:
        metrics['rewards'][agent] = []
        metrics['q_values'][agent] = []
        metrics['best_avg_reward'][agent] = -float("inf")

    models = {} # agent models
    memory = {} # execution memory
    val_memory = {} # validation memory

    central_rep = CPC(args.num_timesteps, args.batch_size, args.seq_len) # TODO: add to args
    
    for agent in env.agents:
        model = PPO(CnnPolicy,
                    env,
                    verbose=3,
                    gamma=0.99,
                    n_steps=125,
                    ent_coef=0.01,
                    learning_rate=0.00025,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    gae_lambda=0.95,
                    n_epochs=4,
                    clip_range=0.2,
                    clip_range_vf=1) # TODO: check and fix
        models[agent] = Agent(args, env.action_spaces[agent].n, model, central_rep)
        memory[agent] = ReplayMemory() # figure this out, it's probably just a typical replaybuffer
        val_memory[agent] = ReplayMemory()
            
    T, converged = 0, False
    while not converged:
        i = 0
        env.reset()
        for agent in env.agent_iter():
            if i % len(env.agents):
                T += 1
                if T > args.evaluation_size:
                    converged=True
                    break
            observation, reward, done, info = env.last()
            action = models[agent].action(observation) if not done else None
            env.step(action)
            val_memory[agent].append(torch.tensor(observation), None, None, done)
            i += 1
    

    
if __name__ == '__main__':
    main()
