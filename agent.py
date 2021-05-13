"""
Single agent that interacts with central learner and learns its own parameters.
"""
import os
import torch
import torch.nn as nn

class Agent:
    def __init__(self, args, action_space, model):
        self.args = args
        self.model = model

    def action(self, state):
        pass

    def learn(self, memory):
        pass
