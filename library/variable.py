from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from pathlib import Path


class Variable(BaseSettings):
    MODEL_PATH: str
    LOG_PATH: str

    model_config = ConfigDict(
        env_file=str(Path(__file__).resolve().parents[1] / ".env")
    )


variables = Variable()
