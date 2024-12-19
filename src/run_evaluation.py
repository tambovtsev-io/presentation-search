import asyncio
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import fire
import pandas as pd
from langchain_openai import ChatOpenAI

from src.config import Config, Provider, load_spreadsheet
from src.config.logging import setup_logging
from src.eval.eval_mlflow import (
    BaseMetric,
    MlflowConfig,
    PageMatch,
    PresentationMatch,
    RAGEvaluatorMlflow,
)
from src.eval.evaluate import LangsmithConfig, RAGEvaluatorLangsmith
from src.rag import ChromaSlideStore, PresentationRetriever
from src.rag.preprocess import RegexQueryPreprocessor
from src.rag.score import (
    BaseScorer,
    ExponentialScorer,
    HyperbolicScorer,
    ScorerFactory,
    ScorerPresets,
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


@dataclass
class EvalComponents:
    """Container for evaluation components"""

    llm: ChatOpenAI
    storage: ChromaSlideStore
    retriever: Union[PresentationRetriever, LLMPresentationRetriever]
    scorer_instances: List[BaseScorer]


class EvaluationCLI:
    """CLI for RAG evaluation pipeline"""

    def __init__(self):
        """Initialize CLI with logging setup"""
        setup_logging(logger, Path("logs"))
        self.config = Config()

    def _get_scorers(self, scorers: List[str]) -> List[BaseScorer]:
        """Get scorer instances from specifications

        Args:
            scorers: List of scorer specifications. Each item can be:
                - Preset name: "default", "weighted", "all"
                - Scorer spec: "min", "hyperbolic_k2.0_p3.0", etc

        Returns:
            List of configured scorer instances
        """
        scorer_specs = []

        # Process each specification
        for spec in scorers:
            if hasattr(ScorerPresets, spec.upper()):
                scorer_specs.extend(getattr(ScorerPresets, spec.upper()))
            else:
                scorer_specs.append(spec)

        # Create scorer instances
        scorer_instances = ScorerFactory.parse_scorer_specs(scorer_specs)

        if not scorer_instances:
            logger.warning("No valid scorers specified, using default")
            scorer_instances = [ScorerFactory.create_default()]
        else:
            logger.info(f"Using scorers: {[s.id for s in scorer_instances]}")

        return scorer_instances

    def _initialize_components(
        self,
        retriever: str,
        provider: str,
        model_name: Optional[str],
        collection: str,
        scorers: List[str],
        preprocessing: Optional[str] = None,
        temperature: float = 0.2,
    ) -> EvalComponents:
        """Initialize common evaluation components

        Args:
            retriever: Retriever type ('basic' or 'llm')
            provider: Model provider ('vsegpt' or 'openai')
            model_name: Optional specific model name
            collection: ChromaDB collection name
            scorers: List of scorer specifications
            temperature: Model temperature

        Returns:
            Configured evaluation components

        Raises:
            ValueError: If invalid retriever type or provider specified
        """
        try:
            retriever_type = RetrieverType(retriever.lower())
            provider = Provider(provider.lower())
        except ValueError as e:
            logger.error(f"Invalid parameter: {str(e)}")
            raise

        # Initialize components
        llm = self.config.model_config.get_llm(provider, model_name, temperature)
        embeddings = self.config.embedding_config.get_embeddings(provider)
        query_preprocessor = {"regex": RegexQueryPreprocessor()}.get(preprocessing) if preprocessing else None

        storage = ChromaSlideStore(
            collection_name=collection, embedding_model=embeddings, query_preprocessor=query_preprocessor
        )

        logger.info(f"Initialized storage collection: {collection}")

        # Get scorer instances
        scorer_instances = self._get_scorers(scorers)

        # Configure retriever
        retriever_instance = get_retriever(storage, retriever_type, llm)

        return EvalComponents(
            llm=llm,
            storage=storage,
            retriever=retriever_instance,
            scorer_instances=scorer_instances,
        )

    def mlflow(
        self,
        retriever: str = "basic",
        n_query_results: int = 50,
        n_contexts: int = -1,
        n_pages: int = -1,
        preprocessing: str = "regex",
        provider: str = "vsegpt",
        model_name: Optional[str] = None,
        collection: str = "pres1",
        experiment: str = "PresRetrieve_eval",
        scorers: List[str] = ["default"],
        metrics: List[str] = ["basic"],
        n_judge_contexts: int = 8,
        n_questions: int = -1,
        max_concurrent: int = 8,
        rate_limit_timeout: float = -1,
        temperature: float = 0.2,
        spread_id: Optional[str] = None,
        sheet_id: Optional[str] = None,
        write_to_google: bool = False,
    ) -> None:
        """Run evaluation pipeline with MLflow tracking.

        Key Arguments:
            scorers: List of scorer specifications for ranking results
                Options:
                    - Presets: 'default', 'all', 'weightedall', 'hyperbolic', 'exponential', 'step', 'linear'
                    - Individual: 'min', 'hyperbolic_k2.0_p3.0'
                Default: ['default']

            metrics: List of evaluation metrics to use
                Options:
                    - Presets: 'basic', 'llm', 'all'
                    - Individual: 'presentationmatch', 'presentationfound', 'pagematch',
                                'pagefound', 'presentationcount', 'llmrelevance'
                Default: ['basic']

            n_query_results: Number of results to fetch from vector store (default: 50)
            n_contexts: Number of contexts per presentation, -1 for unlimited (default: -1)
            n_pages: Number of pages per presentation, -1 for unlimited (default: -1)
            n_judge_contexts: Number of contexts for LLM evaluation (default: 8)
            preprocessing: Query preprocessing type ('regex' or None) (default: 'regex')
            rate_limit_timeout: Delay between API calls in seconds, -1 to disable (default: -1)

        Examples:
            # Basic evaluation with default settings
            python -m src.run_evaluation mlflow

            # Custom scoring and metrics
            python -m src.run_evaluation mlflow \
                --scorers=[min,hyperbolic_k2.0_p3.0] \
                --metrics=[basic,llmrelevance] \
                --n_query_results=100
        """
        try:
            # Initialize components
            components = self._initialize_components(
                retriever=retriever,
                provider=provider,
                model_name=model_name,
                collection=collection,
                scorers=scorers,
                preprocessing=preprocessing,
                temperature=temperature,
            )

            # Set attributes
            components.retriever.n_query_results = n_query_results
            components.retriever.n_contexts = n_contexts
            components.retriever.n_pages = n_pages

            # Setup evaluation config
            db_path = self.config.navigator.eval_runs / "mlruns.db"
            artifacts_path = self.config.navigator.eval_artifacts

            eval_config = MlflowConfig(
                experiment_name=experiment,
                metrics=metrics,
                scorers=components.scorer_instances,
                retriever=components.retriever,
                metric_args=dict(
                    rate_limit_timeout=(
                        rate_limit_timeout or 1.05
                        if provider == Provider.VSEGPT
                        else -1.0
                    )
                ),
                n_judge_contexts=n_judge_contexts,
                write_to_google=write_to_google,
            )

            evaluator = RAGEvaluatorMlflow(
                config=eval_config,
                llm=components.llm,
                max_concurrent=max_concurrent,
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
        scorers: List[str] = ["default"],
        n_questions: int = -1,
        max_concurrent: int = 5,
        temperature: float = 0.2,
    ) -> None:
        """Run LangSmith-based evaluation pipeline"""
        try:
            # Initialize components
            components = self._initialize_components(
                retriever=retriever,
                provider=provider,
                model_name=model_name,
                collection=collection,
                scorers=scorers,
                temperature=temperature,
            )

            # Configure evaluation
            langsmith_config = LangsmithConfig(
                dataset_name=dataset,
                experiment_prefix=experiment_prefix,
                retriever=components.retriever,
                scorers=components.scorer_instances,
                max_concurrency=max_concurrent,
            )

            evaluator = RAGEvaluatorLangsmith(
                config=langsmith_config,
                llm=components.llm,
            )

            # Load and process questions
            sheet_id = os.getenv("BENCHMARK_SPREADSHEET_ID")
            questions_df = evaluator.load_questions_from_sheet(sheet_id)
            logger.info(f"Loaded {len(questions_df)} questions")

            if n_questions > 0:
                questions_df = questions_df.sample(n_questions).reset_index()
                logger.info(f"Selected {len(questions_df)} random questions")

            evaluator.run_evaluation()
            logger.info("LangSmith evaluation completed successfully")

        except Exception as e:
            logger.error("LangSmith evaluation failed", exc_info=True)
            raise


def main():
    """Entry point for Fire CLI"""
    fire.Fire(EvaluationCLI)


if __name__ == "__main__":
    main()


"""
EXAMPLES



# Basic MLflow evaluation with default settings
python -m src.run_evaluation mlflow

# MLflow with specific scorer combinations
python -m src.run_evaluation mlflow \
    --scorers=[min,hyperbolic_k2.0_p3.0]

# MLflow with preset scorer configurations
python -m src.run_evaluation mlflow \
    --scorers=[default,weighted]

# MLflow with LLM-enhanced retrieval
python -m src.run_evaluation mlflow \
    --retriever=llm \
    --scorers=[exponential_a0.7_w1.7_s2.8] \
    --provider=openai \
    --model-name=gpt-4 \
    --temperature=0.1

# MLflow with limited questions and custom experiment name
python -m src.run_evaluation mlflow \
    --n-questions=20 \
    --experiment=custom_experiment \
    --max-concurrent=3

# MLflow with specific spreadsheet
python -m src.run_evaluation mlflow \
    --spread-id=your_spreadsheet_id \
    --sheet-id=your_sheet_id

# My extended command
poetry run python -m src.run_evaluation mlflow \
          --retriever="basic" \
          --provider="vsegpt" \
          --scorers=["min", "exponential"] \
          --metrics=[basic] \
          --max_concurrent=5 \
          --model_name="openai/gpt-4o-mini" \
          --collection="pres_45" \
          --experiment="PresRetrieve_45" \
          --n_questions=3 \
          --temperature=0.2 \
          --sheet_id="1636334554" \
          --write_to_google=true


# Basic LangSmith evaluation
python -m src.run_evaluation langsmith

# LangSmith with custom configuration
python -m src.run_evaluation langsmith \
    --retriever=llm \
    --scorers=[default,exponential_a0.7_w1.7_s2.8] \
    --dataset=custom_dataset \
    --experiment-prefix=test_run \
    --n-questions=10

# LangSmith with VSE-GPT provider
python -m src.run_evaluation langsmith \
    --provider=vsegpt \
    --model-name=custom_model \
    --max-concurrent=2
"""
