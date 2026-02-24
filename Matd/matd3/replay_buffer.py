import numpy as np


class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.storage = []
        self.max_size = max_size

    def add(self, data):
        if len(self.storage) >= self.max_size:
            self.storage.pop(0)
        self.storage.append(data)

    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.storage), size=batch_size)
        return [self.storage[i] for i in indices]