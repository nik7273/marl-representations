import os
import plotly
import torch
import numpy as np
import supersuit as ss

from collections import defaultdict
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
from array2gif import write_gif
from utils import init_env, change_reward_fn, set_mode

def test_model(args, t, models, val_memory, metrics, results_dir, evaluate=False):
    env = init_env(args)
    
    metrics["steps"].append(t)
    T_rewards, T_Qs = {agent: [] for agent in env.agents}, {agent: [] for agent in env.agents}

    obs_list = {agent: [] for agent in env.agents}
    best_total_reward = -float("inf")

    for _ in range(args.eval_episodes):
        env.reset()
        reward_sum = defaultdict(lambda: 0)
        curr_obs_list = {agent: [] for agent in env.agents}
        for agent in env.agent_iter():
            observation, reward, done, _ = env.last()
            action = models[agent].act_e_greedy(torch.tensor(observation)) if not done else None
            env.step(action)
            reward_sum[agent] += reward

            # add frames to list for gameplay gif generation    
            temp_obs = observation * 255
            for x in np.transpose(temp_obs,(2,0,1)):
                curr_obs_list[agent].append(np.stack((x,)*3,axis=0))

        for agent, agent_reward in reward_sum.items():
            T_rewards[agent].append(agent_reward)

        env.reset()

        if max(reward_sum.values()) > best_total_reward:
            best_total_reward =  max(reward_sum.values())
            for agent in env.agents:
                obs_list[agent] = curr_obs_list[agent]            
    env.reset()


    for agent in env.agents:
        for obs in val_memory[agent]:
            T_Qs[agent].append(models[agent].evaluate_q(torch.tensor(obs)))

    avg_reward = {}
    avg_Q = {}
    for agent in env.agents:
        avg_reward[agent] = sum(T_rewards[agent]) / len(T_rewards[agent])
        avg_Q[agent] = sum(T_Qs[agent]) / len(T_Qs[agent])

    if not evaluate:
        for agent in env.agents:
            # save model params if improved
            if avg_reward[agent] > metrics["best_avg_reward"][agent]:
                metrics["best_avg_reward"][agent] = avg_reward[agent]
                models[agent].save(results_dir)

            # Append to results and save metrics
            metrics["rewards"][agent].append(T_rewards[agent])
            metrics["Qs"][agent].append(T_Qs[agent])
            torch.save(metrics, os.path.join(results_dir, "metrics.pth"))

        # Plot
        for agent in env.agents:
            _plot_line(
                metrics["steps"],
                metrics["rewards"][agent],
                f"{agent} Reward",
                path=results_dir,
            )
            _plot_line(
                metrics["steps"], metrics["Qs"][agent], f"{agent} Q", path=results_dir
            )

            # Save best run as gif
            write_gif(obs_list[agent], f"{results_dir}/{agent}-{T}.gif", fps=15)

    return avg_reward, avg_Q


# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=""):
    max_colour, mean_colour, std_colour, transparent = (
        "rgb(0, 132, 180)",
        "rgb(0, 172, 237)",
        "rgba(29, 202, 255, 0.2)",
        "rgba(0, 0, 0, 0)",
    )

    ys = torch.tensor(ys_population, dtype=torch.float32)
    ys_min, ys_max, ys_mean, ys_std = (
        ys.min(1)[0].squeeze(),
        ys.max(1)[0].squeeze(),
        ys.mean(1).squeeze(),
        ys.std(1).squeeze(),
    )
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    trace_max = Scatter(
        x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash="dash"), name="Max"
    )
    trace_upper = Scatter(
        x=xs,
        y=ys_upper.numpy(),
        line=Line(color=transparent),
        name="+1 Std. Dev.",
        showlegend=False,
    )
    trace_mean = Scatter(
        x=xs,
        y=ys_mean.numpy(),
        fill="tonexty",
        fillcolor=std_colour,
        line=Line(color=mean_colour),
        name="Mean",
    )
    trace_lower = Scatter(
        x=xs,
        y=ys_lower.numpy(),
        fill="tonexty",
        fillcolor=std_colour,
        line=Line(color=transparent),
        name="-1 Std. Dev.",
        showlegend=False,
    )
    trace_min = Scatter(
        x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash="dash"), name="Min"
    )

    plotly.offline.plot(
        {
            "data": [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
            "layout": dict(
                title=title, xaxis={"title": "Step"}, yaxis={"title": title}
            ),
        },
        filename=os.path.join(path, title + ".html"),
        auto_open=False,
    )
