# -*- coding: utf-8 -*-
from collections import deque
import numpy as np


class ReplayMemory:
    def __init__(self, max_size=100):
        self.memory = deque(maxlen=max_size)

    def push(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.memory)),
                               size=batch_size,
                               replace=False)
        return [self.memory[ii] for ii in idx]
