import click

from src.commands import build_vector_db_command
from src.logger import setup_logger


@click.group()
def cli() -> None:
    pass


cli.add_command(build_vector_db_command, name="build-vector-db")

if __name__ == "__main__":
    setup_logger()
    cli()
