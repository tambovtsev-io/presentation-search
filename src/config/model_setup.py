import logging
import os
from enum import Enum
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from src.testing_utils.echo_llm import EchoLLM

load_dotenv()

logger = logging.getLogger(__name__)


class Provider(str, Enum):
    VSEGPT = "vsegpt"
    OPENAI = "openai"


class ModelConfig:
    """
    Configuration class for loading different language models.
    Provides methods to load various model providers.
    """

    def load_vsegpt(
        self,
        model: str = "vis-openai/gpt-4o-mini",
        temperature: float = 0.2,
        *args,
        **kwargs,
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
            base_url=api_base,
            model=model,
            api_key=api_key,
            temperature=temperature,
            *args,
            **kwargs,
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

    def get_llm(
        self,
        provider: Provider,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
    ) -> Any:
        """Get LLM based on type and name

        Args:
            model_type: Type of model to use (vsegpt or openai)
            model_name: Optional model name (e.g. "gpt-4-vision-preview")

        Returns:
            Configured LLM instance
        """

        if provider == Provider.VSEGPT:
            model_name = model_name or "openai/gpt-4o-mini"
            logger.info(f"Using VSEGPT model: {model_name}")
            return self.load_vsegpt(model=model_name, temperature=temperature)

        elif provider == Provider.OPENAI:
            model_name = model_name or "gpt-4o-mini"
            logger.info(f"Using OpenAI model: {model_name}")
            return self.load_openai(model=model_name, temperature=temperature)
        else:
            raise ValueError(f"Unknown model type: {model_name}")


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

    def get_embeddings(
        self,
        provider: Provider,
        model_name: str = "text-embedding-3-small",
        temperature: float = 0.2,
    ) -> Any:
        """Get Embeddings based on type and name

        Args:
            model_type: Type of model to use (vsegpt or openai)
            model_name: Optional model name (e.g. "gpt-4-vision-preview")

        Returns:
            Configured LLM instance
        """

        if provider == Provider.VSEGPT:
            logger.info(f"Using VSEGPT model: {model_name}")
            return self.load_vsegpt(model=model_name)

        elif provider == Provider.OPENAI:
            logger.info(f"Using OpenAI model: {model_name}")
            return self.load_openai(model=model_name)
        else:
            raise ValueError(f"Unknown model type: {model_name}")
