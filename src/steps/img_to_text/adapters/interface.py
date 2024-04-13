from abc import ABC, abstractmethod


class ImageToText(ABC):
    @abstractmethod
    def describe(self, image_path: str) -> str:
        pass
