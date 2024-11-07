from dataclasses import dataclass

from src.config import (
    Navigator,
    MultilineWrapper,
    ModelConfig
)


@dataclass
class Config:
    """
    This class is a shortcut for importing configuration for the application.
    Use-case: convenient import in notebooks.
    """
    navigator: Navigator = Navigator()
    model_config: ModelConfig = ModelConfig()
    text_wrapper: MultilineWrapper = MultilineWrapper()
