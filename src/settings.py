import os
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

OPENAI_ENGINE = os.getenv("OPENAI_ENGINE")

DATABASE_URL = os.getenv("DATABASE_URL")


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        config = yaml.safe_load(f.read())
    return config
