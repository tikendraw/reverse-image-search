from abc import ABC, abstractmethod
from pathlib import Path


class BaseVectorDB(ABC):

    @abstractmethod
    def get_similar_images(self, x: list[float], k: int = 5) -> list[float]:
        pass

    @abstractmethod
    def update_images(self, images: list) -> None:
        pass
    
