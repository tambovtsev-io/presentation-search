import os
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.language_models.base import BaseLanguageModel
from src.testing_utils.echo_llm import EchoLLM

from dotenv import load_dotenv
load_dotenv()

class ModelConfig:
    """
    Configuration class for loading different language models.
    Provides methods to load various model providers.
    """
    def load_vsegpt(
        self,
        model: str = "vis-openai/gpt-4o",
        temperature: float = 0.0
    ) -> BaseLanguageModel:
        """Load VSEGPT OpenAI-compatible model.

        Args:
            model: Model identifier from vsegpt.ru/Docs/Models
            temperature: Sampling temperature (0.0 = deterministic)

        Returns:
            Configured language model instance
        """
        api_base = os.environ["VSEGPT_API_BASE"]
        api_key = os.environ["VSEGPT_API_KEY"]

        return ChatOpenAI(
            base_url=api_base,
            model=model,
            api_key=api_key,
            temperature=temperature
        )

    def load_echo_llm(self) -> EchoLLM:
        return EchoLLM()
