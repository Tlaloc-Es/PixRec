import click

from src.commands import (
    build_vector_db_command,
    img_to_text_command,
    pipeline_command,
    segment_command,
)
from src.logger import setup_logger


@click.group()
def cli() -> None:
    pass


cli.add_command(build_vector_db_command, name="build-vector-db")
cli.add_command(segment_command, name="segment")
cli.add_command(img_to_text_command, name="img-to-text")
cli.add_command(pipeline_command, name="pipeline")

if __name__ == "__main__":
    setup_logger()
    cli()
