"""
Single agent that interacts with central learner and learns its own parameters.
"""
import os
import torch
import torch.nn as nn
from collections import deque 

class Agent:
    def __init__(self, args, action_space, model, central_rep):
        self.args = args
        self.model = model
        self.central_rep = central_rep
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, eps=args.adam_eps)

    def action(self, state):
        return torch.randint(len(self.action_space), size=(1,)).long()

    def update_params(self):
        # learn for a specific agent is called during trainingp
        # we want to use the CPC loss function as part of training for this model
        # but the CPC is centralized; so we somehow need to share parameter updates in this central model???
        nce_loss = self.central_rep.get_loss()
        loss = nce_loss + log_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
def set_mode(models, mode="train"):
    if mode == "train":
        for agent in models.keys():
            models[agent].train()
    elif mode == "eval":
        for agent in models.keys():
            models[agent].eval()
    else:
        raise Exception("mode does not exist")

class ObsBuffer:
    def __init__(self, max_len):
        self.buf = deque([])
        self.max_len = max_len

    def append(self, new_obs):
        if len(self.buf) >= self.max_len:
            self.buf.popleft()
        self.buf.append(new_obs)

    def drain(self):
        while self.buf:
            self.buf.popleft()
    
