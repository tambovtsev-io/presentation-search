import os
from tempfile import NamedTemporaryFile
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import mlflow
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from src.config import Config, load_spreadsheet
from src.rag import (
    ChromaSlideStore,
    HyperbolicScorer,
    MinScorer,
    PresentationRetriever,
    ScorerTypes,
)


class RetrievalMetrics:
    """Metrics calculators for retrieval evaluation"""

    @staticmethod
    def presentation_match(run_output: Dict, ground_truth: Dict) -> float:
        """Check if top-1 retrieved presentation matches ground truth"""
        best_pres_info = run_output["contexts"][0]
        best_pres_name = best_pres_info["pres_name"]
        return float(best_pres_name == ground_truth["pres_name"])

    @staticmethod
    def presentation_found(run_output: Dict, ground_truth: Dict) -> float:
        """Check if ground truth presentation is in top-k"""
        found_pres_names = [c["pres_name"] for c in run_output["contexts"]]
        return float(ground_truth["pres_name"] in found_pres_names)

    @staticmethod
    def page_match(run_output: Dict, ground_truth: Dict) -> float:
        """Check if best page matches ground truth"""
        score = 0.0
        for pres_info in run_output["contexts"]:
            best_page_found = pres_info["pages"][0]
            if pres_info["pres_name"] == ground_truth["pres_name"]:
                reference_pages = ground_truth["pages"]
                if not reference_pages:
                    score = 1.0
                elif best_page_found in reference_pages:
                    score = 1.0
        return score

    @staticmethod
    def page_found(run_output: Dict, ground_truth: Dict) -> float:
        """Check if ground truth pages are found"""
        score = 0.0
        for pres_info in run_output["contexts"]:
            pages_found = pres_info["pages"]
            if pres_info["pres_name"] == ground_truth["pres_name"]:
                reference_pages = ground_truth["pages"]
                if not reference_pages:
                    score = 1.0
                elif not set(reference_pages) - set(pages_found):
                    score = 1.0
        return score

    @staticmethod
    def n_pages(run_output: Dict, ground_truth: Dict) -> float:
        """Count number of pages returned"""
        pres_info = run_output["contexts"][0]
        return float(len(pres_info["pages"]))

    @staticmethod
    def n_pres(run_output: Dict, ground_truth: Dict) -> float:
        """Count number of presentations returned"""
        return float(len(run_output["contexts"]))


class LLMRelevanceEvaluator:
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

    def evaluate(self, run_output: Dict, ground_truth: Dict) -> Dict[str, Union[float, str]]:
        """Evaluate relevance of retrieved content"""
        time.sleep(1.05)  # Rate limiting
        question = ground_truth["question"]
        pres = run_output["contexts"][0]

        contexts_used = pres["contexts"] if self.n_contexts <= 0 else pres["contexts"][:self.n_contexts]
        pres_context = "\n\n---\n\n".join(contexts_used)

        llm_out = self.chain.invoke(dict(query=question, context=pres_context))
        return {
            "llm_relevance_score": float(llm_out["relevance_score"]),
            "llm_relevance_explanation": llm_out["explanation"]
        }


class EvaluationConfig(BaseModel):
    """Configuration for RAG evaluation"""

    experiment_name: str = "RAG_test"
    tracking_uri: str = "sqlite:///data/processed/eval/mlruns.db"

    scorers: List[ScorerTypes] = [MinScorer(), HyperbolicScorer()]
    n_contexts: int = 2
    metrics: List[str] = ["presentation_match", "page_match"]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RAGEvaluator:
    """MLFlow-based evaluator for RAG pipeline"""

    def __init__(
        self,
        storage: ChromaSlideStore,
        config: EvaluationConfig,
        llm: Optional[ChatOpenAI] = None
    ):
        self.storage = storage
        self.config = config
        self.llm = llm or Config().model_config.load_vsegpt(model="openai/gpt-4o-mini")

        # Setup MLFlow
        mlflow.set_tracking_uri(config.tracking_uri)

        # Initialize metrics calculators
        self.metrics = {
            name: getattr(RetrievalMetrics, name)
            for name in self.config.metrics
        }

        if llm:
            self.llm_evaluator = LLMRelevanceEvaluator(
                llm=self.llm,
                n_contexts=self.config.n_contexts
            )

    @staticmethod
    def load_questions_from_sheet(sheet_id: str) -> pd.DataFrame:
        """Load evaluation questions from spreadsheet"""
        df = load_spreadsheet(sheet_id)
        df.fillna(dict(page=""), inplace=True)
        return df

    def evaluate_single(
        self,
        retriever: PresentationRetriever,
        question: str,
        ground_truth: Dict
    ) -> Dict:
        """Evaluate single query"""
        # Run retrieval
        output = retriever(dict(question=question))

        # Calculate metrics
        results = {}
        for name, metric_fn in self.metrics.items():
            results[name] = metric_fn(output, ground_truth)

        # Add LLM evaluation if configured
        if hasattr(self, "llm_evaluator"):
            llm_results = self.llm_evaluator.evaluate(output, ground_truth)
            results.update(llm_results)

        return results

    def run_evaluation(self, questions_df: pd.DataFrame) -> None:
        """Run evaluation for all configured scorers"""
        mlflow.set_experiment(self.config.experiment_name)

        for scorer in self.config.scorers:
            with mlflow.start_run(run_name=f"scorer_{scorer.id}"):
                # Log scorer parameters
                mlflow.log_params(scorer.model_dump())

                # Initialize retriever
                retriever = PresentationRetriever(
                    storage=self.storage,
                    scorer=scorer,
                    n_contexts=self.config.n_contexts
                )

                # Run evaluation for each question
                results_log = []
                metric_values = {name: [] for name in self.metrics.keys()}
                if hasattr(self, "llm_evaluator"):
                    metric_values["llm_relevance_score"] = []

                for _, row in questions_df.iterrows():
                    ground_truth = {
                        "question": row["question"],
                        "pres_name": row["pres_name"],
                        "pages": [int(x) if x else -1 for x in row["page"].split(",")]
                    }

                    output = retriever(dict(question=row["question"]))
                    results = self.evaluate_single(
                        retriever=retriever,
                        question=row["question"],
                        ground_truth=ground_truth
                    )

                    for name, value in results.items():
                        if isinstance(value, (int, float)):
                            metric_values[name].append(value)

                    # Prepare row for results log
                    result_row = {
                        "question": row["question"],
                        "expected_presentation": row["pres_name"],
                        "expected_pages": row["page"],
                        "retrieved_presentations": [
                            p["pres_name"] for p in output["contexts"]
                        ],
                        "retrieved_pages": [
                            ",".join(map(str, p["pages"]))
                            for p in output["contexts"]
                        ],
                        **{
                            f"metric_{name}": value
                            for name, value in results.items()
                            if isinstance(value, (int, float))
                        }
                    }

                    # Add LLM explanation if available
                    if "llm_relevance_explanation" in results:
                        result_row["llm_explanation"] = results["llm_relevance_explanation"]

                    results_log.append(result_row)

                # Save metrics results
                results_df = pd.DataFrame(results_log)

                # Save whith file
                with NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                    results_df.to_csv(f.name, index=False)
                    fpath = str(Config().navigator.eval / "detailed_results.csv")
                    mlflow.log_artifact(f.name, fpath)

                # Log average metrics
                for name, values in metric_values.items():
                    if values:  # Skip empty lists
                        mlflow.log_metric(f"mean_{name}", sum(values) / len(values))




def main():
    from dotenv import load_dotenv

    # Load environment
    load_dotenv()

    # Mlflow setup logging
    mlflow.langchain.autolog()

    # Setup components
    project_config = Config()
    llm = project_config.model_config.load_vsegpt(model="openai/gpt-4o-mini")
    embeddings = project_config.embedding_config.load_vsegpt()

    storage = ChromaSlideStore(collection_name="pres0", embedding_model=embeddings)

    eval_config = EvaluationConfig(
        experiment_name="PresRetrieve_mlflow",
        metrics=["presentation_match", "page_match"],
        scorers=[MinScorer(), HyperbolicScorer()],
    )

    evaluator = RAGEvaluator(
        storage=storage,
        config=eval_config,
        llm=llm
    )

    # Load questions
    sheet_id = os.environ["BENCHMARK_SPREADSHEET_ID"]
    questions_df = evaluator.load_questions_from_sheet(sheet_id)

    questions_df.sample(5)

    # Run evaluation
    evaluator.run_evaluation(questions_df)


if __name__ == "__main__":
    main()
