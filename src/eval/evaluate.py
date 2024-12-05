from functools import partial
import os
from typing import Dict, List, Optional

import pandas as pd
from langchain_community.chat_models import ChatOpenAI
from langchain_core import outputs
from langsmith import Client, evaluate, evaluation
from langsmith.evaluation import EvaluationResult, run_evaluator
from langsmith.evaluation.evaluator import DynamicRunEvaluator
from langsmith.schemas import Dataset
from pydantic import BaseModel, ConfigDict
from ragas import SingleTurnSample
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (AnswerCorrectness, AnswerRelevancy,
                           ContextPrecision, ContextRecall, Faithfulness)

from src.config import Config, load_spreadsheet
from src.rag import (ChromaSlideStore, HyperbolicScorer, MinScorer,
                     ScorerTypes, SlideRetriever)


@run_evaluator
def presentation_match(run, example) -> EvaluationResult:
    """Evaluator for checking if retrieved presentation matches ground truth
    Scoring: 1 if match else 0
    """
    prediction = run.outputs["pres_info"]["pres_name"]
    match = int(prediction == example.outputs["pres_name"])
    return EvaluationResult(key="presentation_match", score=match)


@run_evaluator
def page_match(run, example) -> EvaluationResult:
    """Evaluator for checking if retrieved pages match ground truth
    Scoring:
        - 1: retrieved all the ground truth pages and ranked them the highest
        - 0.5: retrieved all the ground truth pages but did not rank them the highest
        # - <score>: retrieved only some of ground truth pages
        - 0: wrong presentation
    """
    pres_info = run.outputs["pres_info"]
    pres_match = bool(pres_info["pres_name"] == example.outputs["pres_name"])

    # page_eval_result = partial(EvaluationResult, key="page_match")
    # if not pres_match:
    #     return page_eval_result(score=0.0)

    pages_found = pres_info["pages"]
    best_page_found = pages_found[0]
    if example.outputs["pages"] and pres_match:
        best_page_reference = example.outputs["pages"][0]
        best_page_match = bool(best_page_found == best_page_reference)
        best_page_found = bool(best_page_reference in pages_found)

        if best_page_match:
            score = 1.0
        elif best_page_found:
            score = 0.75
        else:
            score = 0.5
    elif pres_match:
        score = 1.0
    else:
        score = 0.0

    return EvaluationResult(key="page_match", score=score)


def create_ragas_evaluator(metric):
    """Factory function for RAGAS metric evaluators

    Args:
        metric: Initialized RAGAS metric with LLM

    Returns:
        Evaluator function compatible with LangSmith
    """

    @run_evaluator
    async def evaluate(run, example) -> EvaluationResult:
        sample = SingleTurnSample(
            user_input=example.inputs["question"],
            response=run.outputs["answer"],
            retrieved_contexts=run.outputs["contexts"],
        )
        score = await metric.single_turn_ascore(sample)
        return EvaluationResult(key=metric.name, score=score)

    return evaluate


class EvaluationConfig(BaseModel):
    """Configuration for RAG evaluation"""

    dataset_name: str = "RAG_test"

    # Configure Retrieval
    scorers: List[ScorerTypes] = [MinScorer(), HyperbolicScorer()]

    # Setup Evaluators
    evaluators: List[DynamicRunEvaluator] = [presentation_match, page_match]

    # Configure RAGAS
    ragas_metrics: List[type] = [Faithfulness]  # List of metric classes
    n_contexts: int = 2

    # Configure evaluation
    max_concurrency: int = 2
    experiment_prefix: Optional[str] = None
    sheet_id: Optional[str] = os.environ.get("BENCHMARK_SPREADSHEET_ID")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RAGEvaluator:
    """Evaluator for RAG pipeline using LangSmith"""

    def __init__(
        self,
        storage: ChromaSlideStore,
        config: EvaluationConfig,
        llm: ChatOpenAI = Config().model_config.load_vsegpt(model="openai/gpt-4o-mini"),
    ):
        # Enable LangSmith tracing
        os.environ["LANGCHAIN_TRACING_V2"] = os.environ.get(
            "LANGCHAIN_TRACING_V2", "true"
        )
        os.environ["LANGCHAIN_ENDPOINT"] = os.environ.get(
            "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
        )
        os.environ["LANGCHAIN_PROJECT"] = os.environ.get(
            "LANGCHAIN_PROJECT", "presentation_rag"
        )

        # Setup class
        self.storage = storage
        self.client = Client()
        self.config = config
        llm_unwrapped = llm
        self.llm = LangchainLLMWrapper(llm_unwrapped)

    @classmethod
    def load_questions_from_sheet(cls, sheet_id: str) -> pd.DataFrame:
        """Load evaluation questions from Google Sheets and preprocess dataset"""
        df = load_spreadsheet(sheet_id)
        df.fillna(dict(page=""), inplace=True)
        return df

    def create_dataset(self, df: pd.DataFrame, dataset_name: str) -> Dataset:
        dataset = self.client.create_dataset(dataset_name=self.config.dataset_name)

        examples = dict(inputs=[], outputs=[])
        for _, row in df.iterrows():
            self.client.create_example(
                inputs=dict(question=row["question"]),
                outputs=dict(
                    ground_truth="",
                    pres_name=row["pres_name"],
                    pages=[int(x) if x else -1 for x in row["page"].split(",")],
                ),
                dataset_id=dataset.id,
            )

        self.client.create_examples(
            inputs=examples["inputs"],
            outputs=examples["outputs"],
            dataset_name=dataset_name,
        )
        return dataset

    def create_or_load_dataset(self, df: Optional[pd.DataFrame] = None) -> Dataset:
        """Create or load evaluation dataset in LangSmith"""
        try:
            self.dataset = self.client.read_dataset(
                dataset_name=self.config.dataset_name
            )
            print(f"Using existing dataset: {self.dataset.name}")
            return self.dataset
        except:
            if df is not None:
                self.dataset = self.create_dataset(
                    df, dataset_name=self.config.dataset_name
                )
                print(f"Created new dataset: {self.dataset.name}")
                return self.dataset
            raise ValueError("No dataset provided")

    def _build_evaluator_chains(self) -> Dict:
        chains = {e._name: e for e in self.config.evaluators}

        # For ragas metrics
        embedding_model = self.storage._embeddings
        for metric_cls in self.config.ragas_metrics:
            metric = metric_cls(llm=self.llm, embeddings=embedding_model)
            evaluator = create_ragas_evaluator(metric)
            chains[metric_cls.name] = evaluator

        return chains

    def run_evaluation(self) -> None:
        """Run evaluation for all configured scorers"""
        chains = self._build_evaluator_chains()
        # exp_suffix = str(uuid.uuid4())[:6]

        # NOTE Not sure whether it is only for notebooks
        # import nest_asyncio
        # nest_asyncio.apply()

        for scorer in self.config.scorers:
            if self.config.experiment_prefix:
                experiment_prefix = f"{self.config.experiment_prefix}_{scorer.id}"
            else:
                experiment_prefix = f"{scorer.id}"

            retriever = SlideRetriever(
                storage=self.storage, scorer=scorer, n_contexts=self.config.n_contexts
            )
            evaluate(
                retriever,
                experiment_prefix=experiment_prefix,
                data=self.config.dataset_name,
                evaluators=list(chains.values()),
                metadata=dict(scorer=scorer.id),
                max_concurrency=self.config.max_concurrency,
            )


def main():
    from dotenv import load_dotenv

    from src.rag.score import (ExponentialScorer, ExponentialWeightedScorer,
                               HyperbolicScorer, HyperbolicWeightedScorer,
                               MinScorer)

    # Load env variables
    load_dotenv()

    # Setup llm and embeddings
    project_config = Config()
    llm = project_config.model_config.load_vsegpt(model="openai/gpt-4o-mini")
    embeddings = project_config.embedding_config.load_vsegpt()

    # Initialize components
    storage = ChromaSlideStore(collection_name="pres0", embedding_model=embeddings)
    eval_config = EvaluationConfig(
        dataset_name="RAGAS_5",
        ragas_metrics=[AnswerRelevancy],
        scorers=[MinScorer(), ExponentialScorer()],
    )
    evaluator = RAGEvaluator(storage=storage, config=eval_config, llm=llm)

    # Load questions if needed
    sheet_id = os.environ["BENCHMARK_SPREADSHEET_ID"]
    questions_df = evaluator.load_questions_from_sheet(sheet_id)

    # Create or load dataset
    evaluator.create_or_load_dataset(questions_df)

    # Run evaluation
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
