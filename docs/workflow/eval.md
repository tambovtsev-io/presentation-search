# Измерение качества Retrieval
Для измерения качества я использую Langsmith.

## Измерение качества через разметку
В [таблице](https://docs.google.com/spreadsheets/d/1qWRF_o-RY1x-o-3z08iVb2akh0HS3ZNxVkZi6yoVsI4/edit?gid=0#gid=0) разменные презентации. Формат презентация-вопрос-слайды. Если в ответ выдали нужную презу, скор 1, иначе 0.

## RAGAS - оценка RAG без разметки
- [arxiv](http://arxiv.org/abs/2309.15217)
- [Ноутбук с туториалом](https://colab.research.google.com/github/langfuse/langfuse-docs/blob/main/cookbook/evaluation_of_rag_with_ragas.ipynb)

RAGAS - метод оценки генерации ответов RAG-системами.

Идея: хотим оценивать ответы системы без разметки.

Решение: придумаем метрики, которые 'self-contained and referece-free'.

Что я не понял:
- Почему Answer Relevance считается через эмбеддинги? Можно же гптшку спросить..
- Зачем в Faithfulness разбиение на key-points? Опять же гптшка и без этого поймет


### Входные данные
В метриках RAGAS используются
- Question - Исходный запрос
- Answer - Ответ модели
- Contexts - Документы, которые выдал Retrieval

Метрики:
- Faithfulness - ответ основывается на найденном контексте.
- Answer Relevance - ответ соответствует вопросу.
- Context Relevance - ответ модели строго по делу, без нерелевантной информации.

### RAGAS для поиска презентаций
В этом проекте ответом на запрос является слайды из конкретной презентации. У нас есть Contexts, но нет Answer. Метрики Faithfulness и Сontext Relevance отпадают сразу.

Для метрики Answer Relevance можно попробовать Answer=`лучший слайд`. Идея метрики:
- Для каждого ответа RAG-системы попросим LLM сгенерировать $n$ вопросов $q_i$.
- Получим эмбеддинги этих вопросов $e_i$
- Вычислим Similarity с эмбеддингом исходного запроса $e$: $AR = Mean(Sim(e, e_i))$

Вот, что из этого вышло:
LLM Генерила примерно одинаковые вопросы. Они были на английском, и не похожи на исходный. Получились рандомные скоры.

Возможная причина: У них в примерах короткие вопросы и короткие ответы. LLM не знает что делать с большим описанием слайда. Примеры из их промпта:

```
--------EXAMPLES-----------
Example 1
Input: {
    "response": "Albert Einstein was born in Germany."
}
Output: {
    "question": "Where was Albert Einstein born?",
    "noncommittal": 0
}

Example 2
Input: {
    "response": "I don't know about the  groundbreaking feature of the smartphone invented in 2023 as am unaware of information beyond 2022. "
}
Output: {
    "question": "What was the groundbreaking feature of the smartphone invented in 2023?",
    "noncommittal": 1
}
-----------------------------
```

## Вырезки из кода RAGAS
```python
class CorrectnessClassifier(
    PydanticPrompt[QuestionAnswerGroundTruth, ClassificationWithReason]
):
    instruction = "Given a ground truth and an answer statements, analyze each statement and classify them in one of the following categories: TP (true positive): statements that are present in answer that are also directly supported by the one or more statements in ground truth, FP (false positive): statements present in the answer but not directly supported by any statement in ground truth, FN (false negative): statements found in the ground truth but not present in answer. Each statement can only belong to one of the categories. Provide a reason for each classification."


class ResponseRelevancePrompt(
    PydanticPrompt[ResponseRelevanceInput, ResponseRelevanceOutput]
):
    instruction = """Generate a question for the given answer and Identify if answer is noncommittal. Give noncommittal as 1 if the answer is noncommittal and 0 if the answer is committal. A noncommittal answer is one that is evasive, vague, or ambiguous. For example, "I don't know" or "I'm not sure" are noncommittal answers"""


class ContextPrecisionPrompt(PydanticPrompt[QAC, Verification]):
    name: str = "context_precision"
    instruction: str = (
        'Given question, answer and context verify if the context was useful in arriving at the given answer. Give verdict as "1" if useful and "0" if not with json output.'
    )


class ContextRecallClassificationPrompt(
    PydanticPrompt[QCA, ContextRecallClassifications]
):
    name: str = "context_recall_classification"
    instruction: str = (
        "Given a context, and an answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not. Use only 'Yes' (1) or 'No' (0) as a binary classification. Output json with reason."
    )


# Faithfulness
class NLIStatementPrompt(PydanticPrompt[NLIStatementInput, NLIStatementOutput]):
    instruction = "Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context."
```

Ragas integration with langchain
```python
class EvaluatorChain(Chain, RunEvaluator):
    """
    Wrapper around ragas Metrics to use them with langsmith.
    """

    metric: Metric

```

