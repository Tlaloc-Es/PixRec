from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class ImageToText(ABC):
    @abstractmethod
    def describe(self, image: Union[str, np.array]) -> str:
        pass
