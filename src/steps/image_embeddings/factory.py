from src.steps.image_embeddings.adapters import Resnet50EmbeddingCalculator
from src.steps.image_embeddings.adapters.interface import EmbeddingCalculator


def EmbeddingsGeneratorFactory(embeddings_generator: str) -> EmbeddingCalculator:
    return {"Resnet50": Resnet50EmbeddingCalculator}[embeddings_generator]
