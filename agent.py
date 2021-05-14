"""
Single agent that interacts with central learner and learns its own parameters.
"""
import os
import torch
import torch.nn as nn

class Agent:
    def __init__(self, args, action_space, model, central_rep):
        self.args = args
        self.model = model
        self.central_rep = central_rep

    def action(self, state):
        return torch.randint(len(self.action_space), size=(1,)).long()

    def learn(self):
        # learn for a specific agent is called during training
        # we want to use the CPC loss function as part of training for this model
        # but the CPC is centralized; so we somehow need to share parameter updates in this central model???
        nce_loss = self.central_rep()
        loss = nce_loss + log_loss
        loss.backward()
        
def set_mode(models, mode="train"):
    if mode == "train":
        for agent in models.keys():
            models[agent].train()
    elif mode == "eval":
        for agent in models.keys():
            models[agent].eval()
    else:
        raise Exception("mode does not exist")
