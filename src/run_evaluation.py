import asyncio
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import fire
import pandas as pd
from langchain_openai import ChatOpenAI

from src.config import Config, Provider, load_spreadsheet
from src.config.logging import setup_logging
from src.eval.eval_mlflow import (
    BaseMetric,
    EvaluationConfig,
    PageMatch,
    PresentationMatch,
    RAGEvaluator,
)
from src.rag import (
    ChromaSlideStore,
    HyperbolicScorer,
    MinScorer,
    PresentationRetriever,
    ScorerTypes,
)
from src.rag.storage import LLMPresentationRetriever

logger = logging.getLogger(__name__)


class RetrieverType(str, Enum):
    """Available retriever types"""

    BASIC = "basic"  # Basic vector retriever
    LLM = "llm"  # LLM-enhanced retriever


def get_retriever(
    storage: ChromaSlideStore,
    retriever_type: RetrieverType,
    llm: Optional[ChatOpenAI] = None,
) -> Union[PresentationRetriever, LLMPresentationRetriever]:
    """Get appropriate retriever based on type"""
    if retriever_type == RetrieverType.LLM:
        if llm is None:
            raise ValueError("LLM required for LLM-enhanced retriever")
        return LLMPresentationRetriever(storage=storage, llm=llm)
    return PresentationRetriever(storage=storage)


class EvaluationCLI:
    """CLI for RAG evaluation pipeline"""

    def __init__(self):
        """Initialize CLI with logging setup"""
        setup_logging(logger, Path("logs"))
        self.config = Config()

    def mlflow(
        self,
        retriever: str = "basic",
        provider: str = "vsegpt",
        model_name: Optional[str] = None,
        collection: str = "pres1",
        experiment: str = "PresRetrieve_eval",
        n_questions: int = -1,
        max_concurrent: int = 5,
        rate_limit_timeout: float = -1,
        temperature: float = 0.2,
        spread_id: Optional[str] = None,
        sheet_id: Optional[str] = None,
    ) -> None:
        """Run MLflow-based evaluation pipeline

        Args:
            retriever: Retriever type ('basic' or 'llm')
            provider: Model provider ('vsegpt' or 'openai')
            model_name: Optional specific model name
            collection: ChromaDB collection name
            experiment: MLflow experiment name
            n_questions: Number of questions to evaluate (-1 for all)
            max_concurrent: Maximum concurrent operations
            temperature: Model temperature
            sheet_id: Optional spreadsheet ID to override env variable
        """
        try:
            retriever_type = RetrieverType(retriever.lower())
            provider = Provider(provider.lower())
        except ValueError as e:
            logger.error(f"Invalid parameter: {str(e)}")
            return

        try:
            # Initialize LLM if needed for retriever
            ## TODO Separate llms for eval and inference
            llm = self.config.model_config.get_llm(provider, model_name, temperature)
            embeddings = self.config.embedding_config.get_embeddings(provider)
            storage = ChromaSlideStore(
                collection_name=collection, embedding_model=embeddings
            )

            logger.info(f"Initialized storage collection: {collection}")

            # Setup evaluation config
            db_path = self.config.navigator.eval_runs / "mlruns.db"
            artifacts_path = self.config.navigator.eval_artifacts

            eval_config = EvaluationConfig(
                experiment_name=experiment,
                metrics=[
                    "presentationmatch",
                    "pagematch",
                    "presentationfound",
                    "pagefound",
                    "presentationcount",
                    "llmrelevance",
                ],
                scorers=[MinScorer(), HyperbolicScorer()],
                retriever=get_retriever(storage, retriever_type, llm),
            )

            evaluator = RAGEvaluator(
                config=eval_config, llm=llm, max_concurrent=max_concurrent
            )

            # Load and process questions
            spreadsheet_id = spread_id or os.getenv("BENCHMARK_SPREADSHEET_ID")
            if spreadsheet_id is None:
                raise ValueError("No spreadsheet ID provided")

            questions_df = evaluator.load_questions_from_sheet(
                spreadsheet_id, gid=sheet_id
            )
            logger.info(f"Loaded {len(questions_df)} questions")

            if n_questions > 0:
                questions_df = questions_df.sample(n_questions).reset_index()
                logger.info(f"Selected {len(questions_df)} random questions")

            evaluator.run_evaluation(questions_df)
            logger.info("MLflow evaluation completed successfully")

        except KeyboardInterrupt:
            logger.warning("Evaluation interrupted by user")
        except Exception as e:
            logger.error("MLflow evaluation failed", exc_info=True)
            raise

    def langsmith(
        self,
        retriever: str = "basic",
        provider: str = "vsegpt",
        model_name: Optional[str] = None,
        collection: str = "pres1",
        dataset: str = "RAG_test",
        experiment_prefix: Optional[str] = None,
        n_questions: int = -1,
        max_concurrent: int = 2,
        temperature: float = 0.2,
    ) -> None:
        """Run LangSmith-based evaluation pipeline

        Args:
            retriever: Retriever type ('basic' or 'llm')
            provider: Model provider ('vsegpt' or 'openai')
            model_name: Optional specific model name
            collection: ChromaDB collection name
            dataset: LangSmith dataset name
            experiment_prefix: Optional prefix for experiment names
            n_questions: Number of questions to evaluate (-1 for all)
            max_concurrent: Maximum concurrent operations
            temperature: Model temperature
        """
        try:
            retriever_type = RetrieverType(retriever.lower())
            provider = Provider(provider.lower())
        except ValueError as e:
            logger.error(f"Invalid parameter: {str(e)}")
            return

        try:
            # Initialize components
            llm = (
                self.config.model_config.get_llm(provider, model_name, temperature)
                if retriever_type == RetrieverType.LLM
                else None
            )
            embeddings = self.config.embedding_config.get_embeddings(provider)
            storage = ChromaSlideStore(
                collection_name=collection, embedding_model=embeddings
            )

            logger.info(f"Initialized storage collection: {collection}")

            # Configure evaluation
            retriever = get_retriever(storage, retriever_type, llm)
            scorers = [MinScorer(), HyperbolicScorer()]

            langsmith_config = LangSmithConfig(
                dataset_name=dataset,
                experiment_prefix=experiment_prefix,
                max_concurrency=max_concurrent,
            )

            evaluator = LangSmithEvaluator(
                config=langsmith_config,
                retriever=retriever,
                scorers=scorers,
                llm=llm,
            )

            # Load and process questions
            sheet_id = os.get_env("BENCHMARK_SPREADSHEET_ID")
            questions_df = evaluator.load_questions_from_sheet(sheet_id)
            logger.info(f"Loaded {len(questions_df)} questions")

            if n_questions > 0:
                questions_df = questions_df.sample(n_questions).reset_index()
                logger.info(f"Selected {len(questions_df)} random questions")

            evaluator.run_evaluation(questions_df)
            logger.info("LangSmith evaluation completed successfully")

        except KeyboardInterrupt:
            logger.warning("Evaluation interrupted by user")
        except Exception as e:
            logger.error("LangSmith evaluation failed", exc_info=True)
            raise


def main():
    """Entry point for Fire CLI"""
    fire.Fire(EvaluationCLI)


if __name__ == "__main__":
    main()
