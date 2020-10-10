import numpy as np


class PSTestSeed:
    def __init__(self, size):
        self.size = size

        np.random.seed(13)
        self.seeds = np.random.randint(0, 1000000, size)
        self.pointer = 0

    def get_seed(self):
        seed = self.seeds[self.pointer]
        self.pointer = (self.pointer + 1) % self.size
        return seed
