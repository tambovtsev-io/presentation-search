import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Protocol, Union

import mlflow
import mlflow.config
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from src.config import Config, load_spreadsheet
from src.config.logging import setup_logging
from src.rag import (
    ChromaSlideStore,
    HyperbolicScorer,
    MinScorer,
    PresentationRetriever,
    ScorerTypes,
)

logger = logging.getLogger(__name__)


class MetricResult(BaseModel):
    """Container for metric calculation results"""

    name: str
    score: float
    explanation: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseMetric(ABC):
    """Base class for evaluation metrics"""

    @property
    def name(self) -> str:
        """Get metric name"""
        return self.__class__.__name__.lower()

    @abstractmethod
    def calculate(self, run_output: Dict, ground_truth: Dict) -> MetricResult:
        """Calculate metric value"""
        pass


class PresentationMatch(BaseMetric):
    """Check if top-1 retrieved presentation matches ground truth"""

    def calculate(self, run_output: Dict, ground_truth: Dict) -> MetricResult:
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

    def calculate(self, run_output: Dict, ground_truth: Dict) -> MetricResult:
        found_pres_names = [c["pres_name"] for c in run_output["contexts"]]
        score = float(ground_truth["pres_name"] in found_pres_names)
        return MetricResult(
            name=self.name,
            score=score,
            explanation=f"Found in positions: {[i for i, p in enumerate(found_pres_names) if p == ground_truth['pres_name']]}",
        )


class PageMatch(BaseMetric):
    """Check if best page matches ground truth"""

    def calculate(self, run_output: Dict, ground_truth: Dict) -> MetricResult:
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


class LLMRelevance(BaseMetric):
    """LLM-based relevance scoring"""

    class RelevanceOutput(BaseModel):
        explanation: str = Field(description="Explanation for the relevance score")
        relevance_score: int = Field(description="Relevance score (0 or 1)")

        model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, llm: ChatOpenAI, n_contexts: int = -1):
        self.n_contexts = n_contexts
        prompt_template = PromptTemplate.from_template(
            """\
You will act as an expert relevance assessor for a presentation retrieval system. Your task is to evaluate whether the retrieved slide descriptions contain relevant information for the user's query. Consider both textual content and references to visual elements (images, charts, graphs) as equally valid sources of information.

Evaluation Rules:
- Assign score 1 if the descriptions contain ANY relevant information that helps answer the query
- Assign score 0 only if the descriptions are completely unrelated or provide no useful information
- Treat references to visual elements (e.g., "graph shows increasing trend" or "image depicts workflow") as valid information
- Consider partial matches as relevant (score 1) as long as they provide some value in answering the query

For each evaluation, you will receive:
1. The user's query
2. Retrieved slide descriptions

# Query
{query}

--- END OF QUERY ---

# Slide Descriptions
{context}

--- END OF SLIDE DESCRIPTIONS ---

Format output as JSON:

```json
{{
  "explanation": string, # Clear justification explaining why the content is relevant or irrelevant
  "relevance_score": int  # 1 if any relevant information is found, 0 if completely irrelevant
}}
```
"""
        )

        self.chain = (
            prompt_template
            | llm
            | StrOutputParser()
            | JsonOutputParser(pydantic_object=self.RelevanceOutput)
        )

    def calculate(self, run_output: Dict, ground_truth: Dict) -> MetricResult:
        """Evaluate relevance of retrieved content"""
        time.sleep(1.05)  # Rate limiting
        question = ground_truth["question"]
        pres = run_output["contexts"][0]

        contexts_used = (
            pres["contexts"]
            if self.n_contexts <= 0
            else pres["contexts"][: self.n_contexts]
        )
        pres_context = "\n\n---\n\n".join(contexts_used)

        llm_out = self.chain.invoke(dict(query=question, context=pres_context))
        return MetricResult(
            name=self.name,
            score=float(llm_out["relevance_score"]),
            explanation=llm_out["explanation"],
        )


class MetricsRegistry:
    """Factory for creating metric instances"""

    _metrics = {
        "presentationmatch": PresentationMatch,
        "presentationfound": PresentationFound,
        "pagematch": PageMatch,
        "llmrelevance": LLMRelevance,
    }

    @classmethod
    def create(cls, metric_name: str, **kwargs) -> BaseMetric:
        """Create metric instance by name"""
        metric_cls = cls._metrics.get(metric_name.lower())
        if metric_cls is None:
            raise ValueError(f"Unknown metric: {metric_name}")
        return metric_cls(**kwargs)


class EvaluationConfig(BaseModel):
    """Configuration for RAG evaluation"""

    experiment_name: str = "RAG_test"
    tracking_uri: str = f"sqlite:///{Config().navigator.eval_runs / 'mlruns.db'}"
    artifacts_uri: str = f"file:////{Config().navigator.eval_artifacts}"

    scorers: List[ScorerTypes]
    n_contexts: int = 2
    metrics: List[str] = ["presentation_match", "page_match"]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RAGEvaluator:
    """MLFlow-based evaluator for RAG pipeline"""

    def __init__(
        self,
        storage: ChromaSlideStore,
        config: EvaluationConfig,
        llm: Optional[ChatOpenAI] = None,
    ):
        # Setup logging
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Setup Evaluation
        self.storage = storage
        self.config = config
        self.llm = llm or Config().model_config.load_vsegpt(model="openai/gpt-4o-mini")

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
                kwargs = dict(llm=self.llm, n_contexts=config.n_contexts)
            self.metrics.append(MetricsRegistry.create(metric_name, **kwargs))
            self._logger.info(f"Initialized metric: {metric_name}")

    @staticmethod
    def load_questions_from_sheet(*args, **kwargs) -> pd.DataFrame:
        """Load evaluation questions from spreadsheet"""
        df = load_spreadsheet(*args, **kwargs)
        df.fillna(dict(page=""), inplace=True)
        return df

    def evaluate_single(
        self, output: Dict[str, Any], question: str, ground_truth: Dict
    ) -> Dict[str, MetricResult]:
        """Evaluate single query"""
        # Logging
        self._logger.info(f"Evaluating question: {question}")

        results = {}

        for metric in self.metrics:
            result = metric.calculate(output, ground_truth)
            results[metric.name] = result

            self._logger.info(f"Metric {metric.name}: {result.score}")

        return results

    def run_evaluation(self, questions_df: pd.DataFrame) -> None:
        """Run evaluation for all configured scorers"""
        self._logger.info(f"Starting evaluation with {len(questions_df)} questions")

        # Load the existing experiment or create a new one
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
                # Log scorer parameters
                mlflow.log_params(scorer.model_dump())
                self._logger.debug(f"Logged scorer parameters: {scorer.model_dump()}")

                # Initialize retriever
                retriever = PresentationRetriever(
                    storage=self.storage,
                    scorer=scorer,
                    n_pages=self.config.n_contexts,
                )

                # Run evaluation for each question
                results_log = []
                metric_values = {m.name: [] for m in self.metrics}

                for idx, row in questions_df.iterrows():
                    self._logger.info(
                        f"Processing question {idx+1}/{len(questions_df)}: {row['question'][:50]}..."  # pyright: ignore[reportOperatorIssue]
                    )

                    ground_truth = {
                        "question": row["question"],
                        "pres_name": row["pres_name"],
                        "pages": [int(x) if x else -1 for x in row["page"].split(",")],
                    }

                    output = retriever(dict(question=row["question"]))

                    self._logger.info(
                        f"Retrieved {len(output['contexts'])} presentations"
                    )

                    results = self.evaluate_single(
                        output=output,
                        question=row["question"],  # pyright: ignore[reportArgumentType]
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

                    # Add metrics results and explanations
                    for metric_name, metric_result in results.items():
                        result_row[f"metric_{metric_name}_score"] = metric_result.score
                        if metric_result.explanation:
                            result_row[f"metric_{metric_name}_explanation"] = (
                                metric_result.explanation
                            )
                        metric_values[metric_name].append(metric_result.score)

                    results_log.append(result_row)

                # Save detailed results
                results_df = pd.DataFrame(results_log)
                with NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                    results_df.to_csv(f.name, index=False)
                    fpath = str("detailed_results")
                    mlflow.log_artifact(f.name, fpath)
                    self._logger.info(f"Saved detailed results to {fpath}")

                # Log average metrics
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
    eval_config = EvaluationConfig(
        experiment_name="PresRetrieve_speed_eval",
        metrics=["presentationmatch", "pagematch"],
        scorers=[MinScorer(), HyperbolicScorer()],
        tracking_uri=f"sqlite:////{db_path}",
        artifacts_uri=f"file:////{artifacts_path}",
    )
    logger.info("Created evaluation config")

    evaluator = RAGEvaluator(storage=storage, config=eval_config, llm=llm)
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
