from abc import ABC, abstractmethod

import numpy as np


class EmbeddingCalculator(ABC):
    @abstractmethod
    def calculate_embedding(self, image_path: str) -> np.ndarray:
        pass
