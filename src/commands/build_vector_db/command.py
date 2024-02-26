import click

from src.settings import load_config
from src.steps.image_embeddings.factory import EmbeddingsGeneratorFactory


@click.command()
@click.option("-c", "--config_path", required=True)
def build_vector_db_command(config_path):
    raw_config = load_config(config_path)
    pipeline_config = raw_config["pipeline"]
    steps = pipeline_config["steps"]
    imageEmbeddingsGenerator = EmbeddingsGeneratorFactory(
        steps["image-embeddings"]["selected"]
    )(**steps["image-embeddings"].get("vars", {}))

    print(imageEmbeddingsGenerator)
