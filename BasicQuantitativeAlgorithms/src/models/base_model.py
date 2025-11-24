from abc import ABC, abstractmethod
import numpy as np

class PathModel(ABC):
    @abstractmethod
    def simulate(self, T: float, N: int, M: int, random_state=None) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError