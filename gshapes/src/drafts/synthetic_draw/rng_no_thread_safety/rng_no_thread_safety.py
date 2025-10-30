import random
import os
import time


class RNG:
    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed or (os.getpid() ^ int(time.time())))

    def __repr__(self) -> str:
        return f"<RNG id={id(self.rng)} seed={getattr(self.rng, 'seed', 'unknown')}>"
        
    def seed(self, seed: int | None = None) -> None:
        self._rng.seed(seed or (os.getpid() ^ int(time.time())))

    def randint(self, *a, **kw): return self._rng.randint(*a, **kw)
    def uniform(self, *a, **kw): return self._rng.uniform(*a, **kw)
    def choice(self, *a, **kw):  return self._rng.choice(*a, **kw)
    def normal(self, mu, sigma): return self._rng.normalvariate(mu, sigma)
