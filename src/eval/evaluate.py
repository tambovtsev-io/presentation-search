import asyncio
import os
import time
from collections import OrderedDict
from functools import partial
from textwrap import dedent
from typing import ClassVar, Dict, List, Optional, Union

import pandas as pd
from langchain_core import outputs
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import Client, evaluate, evaluation
from langsmith.evaluation import EvaluationResult, aevaluate, run_evaluator
from langsmith.evaluation.evaluator import DynamicRunEvaluator, EvaluationResults
from langsmith.schemas import Dataset
from langsmith.utils import LangSmithError
from pandas._libs.tslibs.np_datetime import py_td64_to_tdstruct
from pandas.core.dtypes.dtypes import re
from pydantic import BaseModel, ConfigDict, Field
from ragas import SingleTurnSample
from ragas.llms.base import LangchainLLMWrapper

from src.config import Config, load_spreadsheet
from src.rag import (
    ChromaSlideStore,
    HyperbolicScorer,
    MinScorer,
    PresentationRetriever,
    ScorerTypes,
)
from src.rag.storage import LLMPresentationRetriever


@run_evaluator
def presentation_match(run, example) -> EvaluationResult:
    """Evaluator for checking if top-1 retrieved presentation matches ground truth
    Scoring: 1 if match else 0
    """
    best_pres_info = run.outputs["contexts"][0]
    best_pres_name = best_pres_info["pres_name"]
    match = int(best_pres_name == example.outputs["pres_name"])
    return EvaluationResult(key="presentation_match", score=match)


@run_evaluator
def presentation_found(run, example) -> EvaluationResult:
    """Evaluator for checking whether ground truth presentation
    is present in top-k retrieved.

    Scoring: 1 if present else 0
    """
    found_pres_names = [c["pres_name"] for c in run.outputs["contexts"]]
    score = int(example.outputs["pres_name"] in found_pres_names)
    return EvaluationResult(key="presentation_found", score=score)


@run_evaluator
def page_match(run, example) -> EvaluationResult:
    """Evaluator for checking if retrieved pages match ground truth
    Scoring: 1 if best page matches the specified else 0
    """
    score = 0
    for pres_info in run.outputs["contexts"]:
        best_page_found = pres_info["pages"][0]
        if pres_info["pres_name"] == example.outputs["pres_name"]:
            reference_pages = example.outputs["pages"]
            if not reference_pages:  # Length is 0
                score = 1
            elif best_page_found in reference_pages:
                score = 1
    return EvaluationResult(key="page_match", score=score)


@run_evaluator
def page_found(run, example) -> EvaluationResult:
    """Evaluator for checking whether ground truth presentation
    is present in top-k retrieved.

    Scoring: 1 if present else 0
    """
    score = 0
    for pres_info in run.outputs["contexts"]:
        pages_found = pres_info["pages"]

        # Count for the presentation which matches ground truth. Even if it is not top-1
        if pres_info["pres_name"] == example.outputs["pres_name"]:
            reference_pages = example.outputs["pages"]
            if not reference_pages:  # Length is 0
                score = 1
            elif not set(reference_pages) - set(pages_found):
                score = 1
    return EvaluationResult(key="page_found", score=score)


@run_evaluator
def n_pages(run, example) -> EvaluationResult:
    pres_info = run.outputs["contexts"][0]
    n_pgs = len(pres_info["pages"])
    return EvaluationResult(key="n_pages", score=n_pgs)


@run_evaluator
def n_pres(run, example) -> EvaluationResult:
    n = len(run.outputs["contexts"])
    return EvaluationResult(key="n_pres", score=n)


def create_llm_relevance_evaluator(llm, n_contexts: int = -1):
    class RelevanceOutput(BaseModel):
        explanation: str = Field(description="Explanation for the relevance score")
        relevance_score: int = Field(description="Relevance score (0 or 1)")

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

    llm = Config().model_config.load_vsegpt(model="openai/gpt-4o-mini")
    chain = (
        prompt_template
        | llm
        | StrOutputParser()
        | JsonOutputParser(pydantic_object=RelevanceOutput)
    )

    @run_evaluator
    def llm_relevance(run, example) -> EvaluationResult:
        # print(run.inputs)
        time.sleep(1.05)
        question = run.inputs["inputs"]["question"]
        pres = run.outputs["contexts"][0]

        contexts_used = (
            pres["contexts"] if n_contexts <= 0 else pres["contexts"][:n_contexts]
        )
        pres_context = "\n\n---\n".join(contexts_used)
        llm_out = chain.invoke(dict(query=question, context=pres_context))
        return EvaluationResult(
            key="llm_relevance",
            score=llm_out["relevance_score"],
            comment=llm_out["explanation"],
        )

    return llm_relevance


def create_ragas_evaluator(metric):
    """Factory function for RAGAS metric evaluators

    Args:
        metric: Initialized RAGAS metric with LLM

    Returns:
        Evaluator function compatible with LangSmith

    Example:
      >>> from ragas.metric import AnswerCorrectness, AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness,
      >>> llm = LangchainLLMWrapper(Config().load_vsegpt())
      >>> metric = AnswerRelevancy(llm=llm, embeddings=embedding_model)
      >>> evaluator = create_ragas_evaluator(metric)
      >>> evaluate(dataset_id=..., evaluators=[evaluator])
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
    retriever: Union[PresentationRetriever, LLMPresentationRetriever]

    # Setup Evaluators
    evaluators: List[DynamicRunEvaluator] = [presentation_match, page_match]

    # Configure RAGAS
    # ragas_metrics: List[type] = [Faithfulness]  # List of metric classes
    n_contexts: int = 10
    n_pages: int = 3

    # Configure evaluation
    max_concurrency: int = 2
    experiment_prefix: Optional[str] = None
    sheet_id: Optional[str] = os.environ.get("BENCHMARK_SPREADSHEET_ID")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __post_init__(self):
        self.retriever.n_contexts = self.n_contexts
        self.retriever.n_pages = self.n_pages

    def get_scored_retriever(self, scorer: ScorerTypes):
        self.retriever.set_scorer(scorer)
        return self.retriever


class RAGEvaluatorLangsmith:
    """Evaluator for RAG pipeline using LangSmith"""

    def __init__(
        self,
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
        self.client = Client()
        self.config = config
        self.llm = llm
        self.llm_wrapped = LangchainLLMWrapper(self.llm)

    @classmethod
    def load_questions_from_sheet(cls, *args, **kwargs) -> pd.DataFrame:
        """Load evaluation questions from Google Sheets and preprocess dataset"""
        df = load_spreadsheet(*args, **kwargs)
        df.fillna(dict(page=""), inplace=True)
        return df

    def create_dataset(self, dataset_name: str, df: pd.DataFrame) -> Dataset:
        dataset = self.client.create_dataset(dataset_name=dataset_name)
        self.fill_dataset(dataset_name, df)
        return dataset

    def fill_dataset(self, dataset_name, df: pd.DataFrame):
        examples = dict(inputs=[], outputs=[], metadata=[])
        for _, row in df.iterrows():
            examples["inputs"].append(dict(question=row["question"]))
            examples["outputs"].append(
                dict(
                    pres_name=row["pres_name"],
                    pages=[int(x) if x else -1 for x in row["page"].split(",")],
                )
            )
            examples["metadata"].append(dict(content=row["content"]))

        self.client.create_examples(
            inputs=examples["inputs"],
            outputs=examples["outputs"],
            metadata=examples["metadata"],
            dataset_name=dataset_name,
        )

    def load_dataset(self, dataset_name: str):
        return self.client.read_dataset(dataset_name=dataset_name)

    def create_or_load_dataset(self, df: Optional[pd.DataFrame] = None) -> Dataset:
        """Create or load evaluation dataset in LangSmith"""
        # See if dataset with this name already exists
        dataset_names = [d.name for d in self.client.list_datasets()]
        if self.config.dataset_name in dataset_names:
            self.dataset = self.load_dataset(self.config.dataset_name)
            print(f"Using existing dataset: {self.dataset.name}")
            return self.dataset
        else:  # Create new dataset otherwise
            if df is not None:
                self.dataset = self.create_dataset(
                    dataset_name=self.config.dataset_name, df=df
                )
                print(f"Created new dataset: {self.dataset.name}")
                return self.dataset
            raise ValueError("No dataset provided")

    def _build_evaluator_chains(self) -> Dict:
        chains = {e._name: e for e in self.config.evaluators}

        # For ragas metrics
        # embedding_model = self.storage._embeddings
        # for metric_cls in self.config.ragas_metrics:
        #     metric = metric_cls(llm=self.llm, embeddings=embedding_model)
        #     evaluator = create_ragas_evaluator(metric)
        #     chains[metric_cls.name] = evaluator

        return chains

    def run_evaluation(self) -> None:
        """Run evaluation for all configured scorers"""
        chains = self._build_evaluator_chains()
        # exp_suffix = str(uuid.uuid4())[:6]

        for scorer in self.config.scorers:
            if self.config.experiment_prefix:
                experiment_prefix = f"{self.config.experiment_prefix}_{scorer.id}"
            else:
                experiment_prefix = f"{scorer.id}"

            retriever = self.config.get_scored_retriever(scorer)

            # async def do_retrieve(*args, **kwargs):
            #     return await retriever.aretrieve(*args, **kwargs)

            evaluate(
                retriever,
                experiment_prefix=experiment_prefix,
                data=self.config.dataset_name,
                evaluators=list(chains.values()),
                metadata=dict(
                    scorer=scorer.id,
                    retriever=self.config.retriever.__class__.__name__,
                ),
                max_concurrency=self.config.max_concurrency,
            )


def main():
    from dotenv import load_dotenv

    from src.rag.score import (
        ExponentialScorer,
        ExponentialWeightedScorer,
        HyperbolicScorer,
        HyperbolicWeightedScorer,
        MinScorer,
    )

    # Load env variables
    load_dotenv()
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    # Setup llm and embeddings
    project_config = Config()
    llm = project_config.model_config.load_vsegpt(model="openai/gpt-4o-mini")
    embeddings = project_config.embedding_config.load_vsegpt()

    # Initialize components
    storage = ChromaSlideStore(collection_name="pres0", embedding_model=embeddings)
    eval_config = EvaluationConfig(
        dataset_name="PresRetrieve_5",
        retriever_cls=LLMPresentationRetriever,
        evaluators=[
            presentation_match,
            presentation_found,
            page_match,
            page_found,
            # create_llm_relevance_evaluator(llm),
        ],
        scorers=[MinScorer(), ExponentialScorer()],
        max_concurrency=1,
    )
    evaluator = RAGEvaluatorLangsmith(storage=storage, config=eval_config, llm=llm)

    # Load questions if needed
    # sheet_id = os.environ["BENCHMARK_SPREADSHEET_ID"]
    # questions_df = evaluator.load_questions_from_sheet(sheet_id)

    # Create or load dataset
    # evaluator.create_or_load_dataset(questions_df)
    # evaluator.load_dataset(self.config.dataset_name)

    # Run evaluation
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
