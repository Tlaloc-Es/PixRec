from abc import ABC, abstractmethod
from typing import Any, List


class ImageSegmentation(ABC):
    @abstractmethod
    def segment(self, image_path: str) -> List[Any]:
        pass
