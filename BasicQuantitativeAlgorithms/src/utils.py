from dataclasses import dataclass
from contextlib import contextmanager
import time
import numpy as np

@dataclass
class RNG:
    seed: int | None = None
    def post_init(self):
        self.rng = np.random.default_rng(self.seed)
    def normal(self, *a, **k): return self.rng.normal(*a, **k)
    def gamma(self, *a, **k): return self.rng.gamma(*a, **k)

    @contextmanager
    def timer(name: str):
        t0 = time.time()
        yield
        t1 = time.time()
        print(f"[{name}] {t1 - t0:.3f}s")