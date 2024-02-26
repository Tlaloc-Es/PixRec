"""logger.py: This file is to handler all things about logging in the app."""

import logging


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,  # Set the minimum level to log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s [%(levelname)s]: %(message)s",  # Define the log message format
        handlers=[
            logging.StreamHandler(),  # Output logs to the console
        ],
    )
