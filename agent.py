"""
Single agent that interacts with central learner and learns its own parameters.
Informed by https://github.com/Kaixhin/Rainbow/blob/master/agent.py
"""
import os
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from collections import deque 
from dqn import DQN
from torch.nn.utils import clip_grad_norm_

class Agent:
    def __init__(self, args, action_space, model, central_rep):
        self.args = args
        self.model = model # ex. PPO, but starting simple with DQN
        self.online_net = self.model.to(device=args.device)
        self.action_space = action_space
        self.central_rep = central_rep
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, eps=args.adam_eps)
        self.batch_size = args.batch_size # is there a different batch size between them?
        self.Vmin = args.V_min
        self.Vmax = args.V_max
        self.atoms = args.atoms
        self.n = args.multi_step
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.discount = args.discount
        self.norm_clip = args.norm_clip
        self.device = args.device

        self.online_net.train()

        self.target_net = DQN(args, self.action_space).to(device=args.device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

    # DQN parameter update
    def update_params(self, memory, central_rep_loss):
        idxs, states, actions, returns, next_states, nonterminals, weights = memory.sample(self.batch_size)
        log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
        log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

        with torch.no_grad():
            # Calculate nth next state probabilities
            pns = self.online_net(next_states) # Probabilities p(s_t+n, ·; θonline)
            dns = self.support.expand_as(pns) * pns # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            argmax_indices_ns = dns.sum(2).argmax(1) # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
            pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            # Compute Tz (Bellman operator T applied to z)
            Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(self.batch_size, self.atoms)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)


        loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        loss += central_rep_loss # nce_loss, TODO: modify weighting later
        self.online_net.zero_grad()
        (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
        self.optimizer.step()

        memory.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions
        

    def reset_noise(self):
        self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state):
        state=state.permute(2,0,1).to(device=self.device)
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
        return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Save model parameters on current device (don't move model between devices)
    def save(self, path, name='model.pth'):
      torch.save(self.online_net.state_dict(), os.path.join(path, name))

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
      with torch.no_grad():
        return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

    def train(self):
      self.online_net.train()

    def eval(self):
      self.online_net.eval()

class ObsBuffer:
    def __init__(self, max_len):
        self.buf = torch.tensor([])
        self.max_len = max_len

    def append(self, new_obs):
        if self.buf.size()[0] >= self.max_len:
            self.buf = torch.cat((self.buf[1:], torch.tensor([new_obs])))
        else:
            self.buf = torch.cat((self.buf, torch.tensor([new_obs])))
    
    def drain(self):
        self.buf = torch.tensor([])

# class ObsBuffer:
#     def __init__(self, max_len):
#         self.buf = deque([])
#         self.max_len = max_len

#     def append(self, new_obs):
#         # if len(self.buf) >= self.max_len:
#         if self.buf.size()[0] >= self.max_len:
#             self.buf.popleft()
#         self.buf.append(new_obs)

#     def drain(self):
#         while self.buf:
#             self.buf.popleft()
