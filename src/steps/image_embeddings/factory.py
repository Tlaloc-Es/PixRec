from src.steps.image_embeddings.adapters.interface import EmbeddingCalculator
from src.steps.image_embeddings.adapters.resnet50 import Resnet50EmbeddingCalculator


def EmbeddingsGeneratorFactory(embeddings_generator: str) -> EmbeddingCalculator:
    return {"Resnet50": Resnet50EmbeddingCalculator}[embeddings_generator]
