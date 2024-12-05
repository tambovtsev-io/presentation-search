import os
from typing import Optional

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai.embeddings import OpenAIEmbeddings

from src.testing_utils.echo_llm import EchoLLM

load_dotenv()


class ModelConfig:
    """
    Configuration class for loading different language models.
    Provides methods to load various model providers.
    """

    def load_vsegpt(
        self, model: str = "vis-openai/gpt-4o-mini", temperature: float = 0.2
    ) -> ChatOpenAI:
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
            base_url=api_base, model=model, api_key=api_key, temperature=temperature
        )

    def load_openai(
        self, model: str = "gpt-4o-mini", temperature: float = 0.2
    ) -> ChatOpenAI:
        """Load OpenAI model.

        Args:
            model: Model identifier from vsegpt.ru/Docs/Models
            temperature: Sampling temperature (0.2 = deterministic)

        Returns:
            Configured language model instance
        """
        api_key = os.environ["OPENAI_API_KEY"]

        return ChatOpenAI(model=model, api_key=api_key, temperature=temperature)

    def load_echo_llm(self) -> EchoLLM:
        return EchoLLM()


class EmbeddingConfig:
    """
    Configuration class for loading different language models.
    Provides methods to load various model providers.
    """

    def load_openai(self, model: str = "text-embedding-3-small") -> Embeddings:
        api_key = os.environ["OPENAI_API_KEY"]
        return OpenAIEmbeddings(model=model, api_key=api_key)

    def load_vsegpt(self, model: str = "text-embedding-3-small") -> Embeddings:
        api_base = os.environ["VSEGPT_API_BASE"]
        api_key = os.environ["VSEGPT_API_KEY"]

        return OpenAIEmbeddings(model=model, api_key=api_key, base_url=api_base)
