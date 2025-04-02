import numpy as np
from disney import DisneyBSDF

class AliasTable(object):
    def __init__(self, ws: np.ndarray):
        self.ws = np.array(ws)
        self.num_samples = ws.shape[0]
        self.alias = np.zeros(self.num_samples, dtype=np.int32)
        self.prob = np.zeros(self.num_samples, dtype=np.float32)
        self._build_alias_table()

    def _build_alias_table(self):
        scaled_probs = self.ws * (self.num_samples / np.sum(self.ws))
        small = np.where(scaled_probs < 1.0)[0]
        large = np.where(scaled_probs > 1.0)[0]
        small_probs = scaled_probs[small]
        large_probs = scaled_probs[large]
        while small_probs.size > 0 and large_probs.size > 0:
            s = small_probs[-1]
            l = large_probs[-1]
            self.prob[s] = small_probs[-1]
            self.alias[s] = l
            small_probs = small_probs[:-1]
            large_probs = large_probs[:-1]
            large_probs[-1] -= (1.0 - s)
            if large_probs[-1] < 1.0:
                small_probs = np.append(small_probs, large_probs[-1])
                large_probs = large_probs[:-1]
        for i in small_probs:
            self.prob[i] = 1.0
        for i in large_probs:
            self.prob[i] = 1.0

    def sample(self, n: int) -> np.ndarray:
        indices = np.random.randint(0, self.num_samples, size=n)
        samples = np.where(np.random.rand(n) < self.prob[indices], indices, self.alias[indices])
        return samples

class Surfel(object):
    def __init__(self, normal: np.ndarray, bsdf: DisneyBSDF):
        self.normal = normal
        self.bsdf = bsdf
