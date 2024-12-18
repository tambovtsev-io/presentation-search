import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from json import load
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Protocol, Union

import mlflow
import mlflow.config
import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from src.config import Config, load_spreadsheet
from src.config.logging import setup_logging
from src.config.spreadsheets import GoogleSpreadsheetManager
from src.rag import (
    ChromaSlideStore,
    HyperbolicScorer,
    MinScorer,
    PresentationRetriever,
    ScorerTypes,
)
from src.rag.storage import LLMPresentationRetriever

logger = logging.getLogger(__name__)


class MetricResult(BaseModel):
    """Container for metric calculation results"""

    name: str
    score: float
    explanation: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseMetric(BaseModel):
    """Base class for evaluation metrics"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def name(self) -> str:
        """Get metric name"""
        return self.__class__.__name__.lower()

    @abstractmethod
    async def acalculate(self, run_output: Dict, ground_truth: Dict) -> MetricResult:
        """Calculate metric value asynchronously"""
        pass

    def calculate(self, run_output: Dict, ground_truth: Dict) -> MetricResult:
        """Synchronous wrapper for calculate"""
        return asyncio.run(self.acalculate(run_output, ground_truth))


class PresentationMatch(BaseMetric):
    """Check if top-1 retrieved presentation matches ground truth"""

    async def acalculate(self, run_output: Dict, ground_truth: Dict) -> MetricResult:
        best_pres_info = run_output["contexts"][0]
        best_pres_name = best_pres_info["pres_name"]
        score = float(best_pres_name == ground_truth["pres_name"])
        return MetricResult(
            name=self.name,
            score=score,
            explanation=f"Retrieved: {best_pres_name}, Expected: {ground_truth['pres_name']}",
        )


class PresentationFound(BaseMetric):
    """Check if ground truth presentation is in top-k"""

    async def acalculate(self, run_output: Dict, ground_truth: Dict) -> MetricResult:
        found_pres_names = [c["pres_name"] for c in run_output["contexts"]]
        score = float(ground_truth["pres_name"] in found_pres_names)
        return MetricResult(
            name=self.name,
            score=score,
            explanation=f"Found in positions: {[i for i, p in enumerate(found_pres_names) if p == ground_truth['pres_name']]}",
        )


class PageMatch(BaseMetric):
    """Check if best page matches ground truth"""

    async def acalculate(self, run_output: Dict, ground_truth: Dict) -> MetricResult:
        score = 0.0
        explanation = ""
        for pres_info in run_output["contexts"]:
            best_page_found = pres_info["pages"][0]
            if pres_info["pres_name"] == ground_truth["pres_name"]:
                reference_pages = ground_truth["pages"]
                if not reference_pages:
                    score = 1.0
                    explanation = "No specific page required"
                elif best_page_found in reference_pages:
                    score = 1.0
                    explanation = f"Found correct page {best_page_found}"
                else:
                    explanation = f"Page mismatch: found {best_page_found}, expected {reference_pages}"

        return MetricResult(name=self.name, score=score, explanation=explanation)


class PageFound(BaseMetric):
    """Check if any of ground truth pages are found in retrieved results

    The page is considered found if it appears in ANY position in the correct presentation.
    This is less strict than PageMatch which checks best matching page.
    """

    async def acalculate(self, run_output: Dict, ground_truth: Dict) -> MetricResult:
        """Calculate metric value"""
        score = 0.0
        explanation = ""

        # Get all pages from each presentation's results
        for pres_info in run_output["contexts"]:
            # Only check pages from the correct presentation
            if pres_info["pres_name"] == ground_truth["pres_name"]:
                found_pages = pres_info["pages"]
                reference_pages = ground_truth["pages"]

                # Handle case when no specific page required
                if not reference_pages:
                    score = 1.0
                    explanation = "No specific page required"
                    break

                # Check if any reference page is found
                matching_pages = set(found_pages) & set(reference_pages)
                if matching_pages:
                    score = 1.0
                    explanation = f"Found pages {matching_pages} in positions {[found_pages.index(p)+1 for p in matching_pages]}"
                    break
                else:
                    explanation = f"No matching pages found. Retrieved: {found_pages}, Expected: {reference_pages}"

        return MetricResult(name=self.name, score=score, explanation=explanation)


class PresentationCount(BaseMetric):
    """Count number of retrieved presentations"""

    async def acalculate(self, run_output: Dict, ground_truth: Dict) -> MetricResult:
        """Count presentations in retrieved results"""
        n_pres = len(run_output["contexts"])
        return MetricResult(
            name=self.name,
            score=float(n_pres),
            explanation=f"Retrieved {n_pres} presentations",
        )


class LLMRelevance(BaseMetric):
    """LLM-based relevance scoring"""

    class RelevanceOutput(BaseModel):
        explanation: str = Field(
            description="Detailed explanation of why the content is/isn't relevant and how it relates to the query"
        )
        relevance_score: int = Field(description="Relevance score from 0-10")

        model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: ChatOpenAI = Field(description="LLM for relevance scoring")
    n_contexts: int = Field(default=-1, description="Number of contexts to evaluate")
    rate_limit_timeout: float = Field(
        default=-1.0, description="Rate limit timeout in seconds"
    )

    def model_post_init(self, __context: Any):
        prompt_template = PromptTemplate.from_template(
            """\
You are an expert relevance assessor for a presentation retrieval system. Your task is to evaluate whether the retrieved slide descriptions contain relevant information that answers the user's query. Analyze all provided slide descriptions as a collective unit and provide a detailed explanation along with a relevance score.

Each slide description contains these equally weighted sections:
- Text Content: The actual text present on the slide
- Visual Content: Description of images, charts, or other visual elements
- Topic Overview: Main themes and subjects covered
- Insights and Conclusions: Key takeaways and conclusions
- Layout and Composition: Structural organization of the slide

Scoring Guidelines:
- 9-10: Perfect match - Content directly and comprehensively answers the query (e.g., query asks about sales trends, and slides show exact sales data and analysis)
- 7-8: Strong relevance - Content clearly relates to the query but may miss minor details (e.g., query asks about complete workflow, slides show most but not all steps)
- 5-6: Moderate relevance - Content addresses the query partially or indirectly (e.g., query asks about specific feature, slides discuss it briefly among other topics)
- 3-4: Weak relevance - Content touches the topic but doesn't provide substantial answer (e.g., query asks about implementation details, slides only mention the concept)
- 1-2: Minimal relevance - Only slight connection to the query (e.g., query asks about specific metric, slides only mention related general category)
- 0: No relevance - Content has no connection to the query

Evaluation Rules:
1. Award points if ANY section (text, visual, etc.) contains relevant information
2. In your explanation, cite specific sections and content that justify your score
3. Treat all sections equally - a match in visual content is as valuable as a match in text content
4. Consider all slides collectively - relevant information might be spread across multiple slides
5. Partial matches are valuable if they provide any useful information related to the query

# Slide Descriptions
{context_str}

--- END OF SLIDE DESCRIPTIONS ---

Question: {query_str}

Output formatting:
{format_instructions}
"""
        )

        self._parser = JsonOutputParser(pydantic_object=self.RelevanceOutput)
        self._chain = prompt_template | self.llm.with_structured_output(
            self.RelevanceOutput
        )

    async def acalculate(self, run_output: Dict, ground_truth: Dict) -> MetricResult:
        """Evaluate relevance of retrieved content"""
        if self.rate_limit_timeout > 0:
            time.sleep(1.05)  # Rate limiting
        question = ground_truth["question"]
        pres = run_output["contexts"][0]

        contexts_used = (
            pres["contexts"]
            if self.n_contexts <= 0
            else pres["contexts"][: self.n_contexts]
        )
        pres_context = "\n\n---\n\n".join(contexts_used)

        llm_out = await self._chain.ainvoke(
            dict(
                query_str=question,
                context_str=pres_context,
                format_instructions=self._parser.get_format_instructions(),
            )
        )
        llm_out_dict = llm_out.model_dump()
        return MetricResult(
            name=self.name,
            score=float(llm_out_dict["relevance_score"]),
            explanation=llm_out_dict["explanation"],
        )


class MetricsRegistry:
    """Factory for creating metric instances"""

    _metrics = {
        "presentationmatch": PresentationMatch,
        "presentationfound": PresentationFound,
        "pagematch": PageMatch,
        "pagefound": PageFound,
        "llmrelevance": LLMRelevance,
        "presentationcount": PresentationCount,
    }

    @classmethod
    def create(cls, metric_name: str, **kwargs) -> BaseMetric:
        """Create metric instance by name"""
        # __import__('pdb').set_trace()
        metric_cls = cls._metrics.get(metric_name.lower())
        if metric_cls is None:
            raise ValueError(f"Unknown metric: {metric_name}")
        return metric_cls(**kwargs)


class MlflowConfig(BaseModel):
    """Configuration for RAG evaluation"""

    experiment_name: str = "RAG_test"
    tracking_uri: str = f"sqlite:////{Config().navigator.eval_runs / 'mlruns.db'}"
    artifacts_uri: str = f"file:////{Config().navigator.eval_artifacts}"

    scorers: List[ScorerTypes]
    retriever: Union[PresentationRetriever, LLMPresentationRetriever]
    metrics: List[str] = ["presentationmatch", "pagematch"]
    n_judge_contexts: int = 10

    write_to_google: bool = False

    metric_args: Dict[str, Any] = {}
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_retriever_with_scorer(self, scorer: ScorerTypes) -> PresentationRetriever:
        self.retriever.set_scorer(scorer)
        return self.retriever


class RAGEvaluatorMlflow:
    """MLFlow-based evaluator for RAG pipeline"""

    def __init__(
        self,
        config: MlflowConfig,
        llm: Optional[ChatOpenAI] = None,
        max_concurrent: int = 5,
    ):
        load_dotenv()

        # Setup logging
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Setup Evaluation
        self.config = config
        self.llm = llm or Config().model_config.load_vsegpt(model="openai/gpt-4o-mini")
        self._max_concurrent = max_concurrent

        # Setup GoogleSheets
        eval_spreadsheet_id = os.getenv("EVAL_SPREADSHEET_ID")
        if eval_spreadsheet_id is not None:
            self.gsheets = GoogleSpreadsheetManager(eval_spreadsheet_id)
        else:
            raise FileNotFoundError("no eval_spreadsheet_id in .env")

        # Setup MLFlow
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.config.enable_async_logging(True)
        self._logger.info(
            f"MLflow tracking URI: {config.tracking_uri}, artifacts: {config.artifacts_uri}"
        )

        # Initialize metrics
        self.metrics: List[BaseMetric] = []
        for metric_name in config.metrics:
            kwargs = {}
            if "llm" in metric_name and llm:
                kwargs = dict(llm=self.llm, n_contexts=config.n_judge_contexts)
            self.metrics.append(MetricsRegistry.create(metric_name, **kwargs))
            self._logger.info(f"Initialized metric: {metric_name}")

    @staticmethod
    def load_questions_from_sheet(*args, **kwargs) -> pd.DataFrame:
        """Load evaluation questions from spreadsheet"""
        df = load_spreadsheet(*args, **kwargs)
        df.fillna(dict(page=""), inplace=True)
        return df

    async def evaluate_single(
        self, output: Dict[str, Any], question: str, ground_truth: Dict
    ) -> Dict[str, MetricResult]:
        """Evaluate single search result against ground truth.

        Args:
            output: Dictionary with retrieval results including:
                - contexts: List of presentation results with metadata
            question: Original search query
            ground_truth: Dictionary with:
                - pres_name: Expected presentation name
                - pages: List of expected page numbers
                - question: Original question

        Returns:
            Dictionary mapping metric names to MetricResult objects
        """
        # Log evaluation start
        self._logger.info(f"Evaluating question: {question}")

        results = {}

        # Calculate each metric
        for metric in self.metrics:
            try:
                result = await metric.acalculate(output, ground_truth)
                results[metric.name] = result

                # Log metric result
                log_msg = f"Metric {metric.name}: {result.score}"
                if result.explanation:
                    log_msg += f" ({result.explanation})"
                self._logger.info(log_msg)

            except Exception as e:
                self._logger.error(
                    f"Failed to calculate metric {metric.name}: {str(e)}"
                )
                # Create failure result
                results[metric.name] = MetricResult(
                    name=metric.name,
                    score=0.0,
                    explanation=f"Calculation failed: {str(e)}",
                )

        return results

    async def process_question(
        self,
        retriever: LLMPresentationRetriever,
        row: pd.Series,
        metric_values: Dict[str, List[float]],
        results_log: List[Dict],
        question_idx: int,
        total_questions: int,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Process single question with semaphore-controlled concurrency"""
        async with semaphore:
            self._logger.info(
                f"Processing question {question_idx+1}/{total_questions}: "
                f"{row['question'][:50]}..."
            )

            ground_truth = {
                "question": row["question"],
                "pres_name": row["pres_name"],
                "pages": [int(x) if x else -1 for x in row["page"].split(",")],
            }

            try:
                # Retrieve asynchronously
                output = await retriever.aretrieve(query=row["question"])

                # Evaluate results
                results = await self.evaluate_single(
                    output=output,
                    question=row["question"],
                    ground_truth=ground_truth,
                )

                # Update aggregated results
                result_row = {
                    "question": row["question"],
                    "expected_presentation": row["pres_name"],
                    "expected_pages": row["page"],
                    "retrieved_presentations": [
                        p["pres_name"] for p in output["contexts"]
                    ],
                    "retrieved_pages": [
                        ",".join(map(str, p["pages"])) for p in output["contexts"]
                    ],
                }

                for metric_name, metric_result in results.items():
                    result_row[f"metric_{metric_name}_score"] = metric_result.score
                    if metric_result.explanation:
                        result_row[f"metric_{metric_name}_explanation"] = (
                            metric_result.explanation
                        )
                    metric_values[metric_name].append(metric_result.score)

                results_log.append(result_row)

            except Exception as e:
                self._logger.error(
                    f"Failed to process question {question_idx+1}: {str(e)}"
                )

    async def process_questions_batch(
        self,
        retriever: LLMPresentationRetriever,
        questions_df: pd.DataFrame,
        metric_values: Dict[str, List[float]],
        results_log: List[Dict],
    ) -> None:
        """Process questions with controlled concurrency"""
        # Create semaphore within the async context
        semaphore = asyncio.Semaphore(self._max_concurrent)

        tasks = [
            self.process_question(
                retriever=retriever,
                row=row,
                metric_values=metric_values,
                results_log=results_log,
                question_idx=idx,
                total_questions=len(questions_df),
                semaphore=semaphore,
            )
            for idx, (_, row) in enumerate(questions_df.iterrows())
        ]

        for completed in tqdm(
            asyncio.as_completed(tasks),
            desc=f"Processing questions (max {self._max_concurrent} concurrent)",
            total=len(tasks),
        ):
            await completed

    def run_evaluation(self, questions_df: pd.DataFrame) -> None:
        """Run evaluation with async LLM queries and controlled concurrency"""
        timestamp = datetime.now().replace(microsecond=0).isoformat()
        self._logger.info(f"Starting evaluation with {len(questions_df)} questions")

        # MLflow setup
        experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
        if experiment is not None:
            experiment_id = experiment.experiment_id
            self._logger.info(
                f"Using existing experiment: {self.config.experiment_name}"
            )
        else:
            experiment_id = mlflow.create_experiment(
                self.config.experiment_name,
                artifact_location=self.config.artifacts_uri,
            )
            self._logger.info(f"Created new experiment: {self.config.experiment_name}")

        mlflow.set_experiment(experiment_id=experiment_id)

        for scorer in self.config.scorers:
            self._logger.info(f"Evaluating with scorer: {scorer.id}")
            with mlflow.start_run(run_name=f"scorer_{scorer.id}"):
                mlflow.log_params(scorer.model_dump())
                self._logger.debug(f"Logged scorer parameters: {scorer.model_dump()}")

                # Initialize retriever
                retriever = self.config.get_retriever_with_scorer(scorer)

                # Initialize aggregation containers
                results_log = []
                metric_values = {m.name: [] for m in self.metrics}

                # Process questions with async handling
                asyncio.run(
                    self.process_questions_batch(
                        retriever, questions_df, metric_values, results_log
                    )
                )

                # Process results
                results_df = pd.DataFrame(results_log)
                results_df["experiment_id"] = experiment_id
                results_df["scorer"] = scorer.id
                results_df["retriever"] = retriever.id
                results_df["timestamp"] = timestamp

                # Save results
                with NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                    results_df.to_csv(f.name, index=False)
                    fpath = str("detailed_results")
                    mlflow.log_artifact(f.name, fpath)
                    self._logger.info(f"Saved detailed results to {fpath}")

                # Write to google sheets if enabled
                if self.config.write_to_google:
                    self.write_to_google_sheet(results_df)

                # Log metrics
                for name, values in metric_values.items():
                    if values:
                        mean_value = sum(values) / len(values)
                        mlflow.log_metric(f"mean_{name}", mean_value)
                        self._logger.info(f"Mean {name}: {mean_value:.3f}")


def main():
    from dotenv import load_dotenv

    # Load environment
    load_dotenv()
    logger.info("Starting RAG evaluation pipeline")

    # Mlflow setup logging
    setup_logging(logger)
    mlflow.langchain.autolog()

    # Setup components
    project_config = Config()
    llm = project_config.model_config.load_vsegpt(model="openai/gpt-4o-mini")
    embeddings = project_config.embedding_config.load_vsegpt()
    logger.info("Initialized LLM and embeddings models")

    storage = ChromaSlideStore(collection_name="pres1", embedding_model=embeddings)
    logger.info("Initialized ChromaDB storage")

    db_path = project_config.navigator.eval_runs / "mlruns.db"
    artifacts_path = project_config.navigator.eval_artifacts
    eval_config = MlflowConfig(
        experiment_name="PresRetrieve_speed_eval",
        metrics=["presentationmatch", "pagematch"],
        scorers=[MinScorer(), HyperbolicScorer()],
        tracking_uri=f"sqlite:////{db_path}",
        artifacts_uri=f"file:////{artifacts_path}",
    )
    logger.info("Created evaluation config")

    evaluator = RAGEvaluatorMlflow(storage=storage, config=eval_config, llm=llm)
    logger.info("Initialized evaluator")

    # Load questions
    sheet_id = os.environ["BENCHMARK_SPREADSHEET_ID"]
    questions_df = evaluator.load_questions_from_sheet(sheet_id)
    logger.info(f"Loaded {len(questions_df)} questions from spreadsheet")

    questions_df = questions_df.sample(5).reset_index()
    logger.info(f"Selected {len(questions_df)} random questions for evaluation")
    evaluator.run_evaluation(questions_df)

    # Run evaluation
    try:
        evaluator.run_evaluation(questions_df)
        logger.info("Evaluation completed successfully")
    except Exception as e:
        logger.exception("Evaluation failed")
        raise


if __name__ == "__main__":
    main()
