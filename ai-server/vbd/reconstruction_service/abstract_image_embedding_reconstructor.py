from abc import ABC, abstractmethod
import numpy as np

class AbstractImageReconstructor(ABC):
    @abstractmethod
    def reconstruct(self, embeddings: np.ndarray, number_of_steps: int = 100) -> np.ndarray:
        pass