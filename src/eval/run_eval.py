import os

from dotenv import load_dotenv
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from src.config import Config
from src.eval.evaluate import EvaluationConfig, RAGEvaluator
from src.rag import ChromaSlideStore
from src.rag.score import (
    ExponentialScorer,
    ExponentialWeightedScorer,
    HyperbolicScorer,
    HyperbolicWeightedScorer,
    MinScorer,
)


def main():
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
