import os
from glob import iglob

import click

from src.services.vectordb.annoy_vector_db import AnnoyVectorDB
from src.settings import load_config
from src.steps.image_embeddings.factory import EmbeddingsGeneratorFactory


@click.command()
@click.option("-c", "--config_path", required=True)
def build_vector_db_command(config_path):
    raw_config = load_config(config_path)
    pipeline_config = raw_config["pipeline"]
    steps_config = pipeline_config["steps"]
    input_config = pipeline_config["input"]
    vectordb_config = pipeline_config["annoy-config"]

    imageEmbeddingsGenerator = EmbeddingsGeneratorFactory(
        steps_config["image-embeddings"]["selected"]
    )(**steps_config["image-embeddings"].get("vars", {}))

    AnnoyVectorDB().build(
        vectordb_config["f"],
        vectordb_config["metric"],
        vectordb_config["n_trees"],
        [
            imageEmbeddingsGenerator.calculate_embedding(path)
            for path in iglob(
                os.path.join(input_config["path"], f"**{input_config['format']}")
            )
        ],
    )
