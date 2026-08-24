import asyncio
import time
from collections.abc import Callable

import mlflow
import nest_asyncio
from datasets import Dataset
from langchain_core.documents import Document
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

from hr_rag.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from hr_rag.formatting import format_docs

__all__ = ["evaluate_rag", "format_docs", "log_to_mlflow", "measure_latency"]


def measure_latency(chain, test_question: str = "What is the leave policy?") -> float:
    start = time.time()
    chain.invoke(test_question)
    return round(time.time() - start, 3)


def evaluate_rag(chain, get_docs_fn: Callable[[str], list[Document]], dataset: dict):
    nest_asyncio.apply()
    answers, contexts = [], []

    for question in dataset["question"]:
        try:
            answer = chain.invoke(question)
            docs = get_docs_fn(question)
            answers.append(str(answer))
            contexts.append([str(d.page_content) for d in docs])
        except Exception as e:  # noqa: BLE001 - one bad Q shouldn't blank the whole eval run
            print(f"  error on: {question[:60]!r} -> {e}")
            answers.append("error")
            contexts.append(["error"])

    eval_dataset = Dataset.from_dict({
        "question": dataset["question"],
        "answer": answers,
        "contexts": contexts,
        "ground_truth": dataset["answer"],
    })

    from hr_rag.embeddings import get_embeddings
    from hr_rag.llm import get_llm

    ragas_llm = LangchainLLMWrapper(get_llm(), run_config=RunConfig(max_workers=1))

    # RAGAS 0.1.x reuses the current loop inside a worker thread. In Jupyter,
    # that loop is already running, so provide a temporary non-running loop.
    previous_loop = asyncio.get_event_loop()
    evaluation_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(evaluation_loop)
    try:
        return evaluate(
            eval_dataset,
            metrics=[faithfulness, context_precision, context_recall],
            llm=ragas_llm,
            embeddings=get_embeddings(),
            run_config=RunConfig(max_workers=1, timeout=120),
            is_async=False,
            raise_exceptions=False,  # one bad question shouldn't blank the whole run
        )
    finally:
        evaluation_loop.close()
        asyncio.set_event_loop(previous_loop)


def log_to_mlflow(run_name: str, result, latency: float, retriever_type: str, top_k: int = 3, extra_params: dict | None = None) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):
        if hasattr(result, "to_pandas"):
            scores = result.to_pandas().mean(numeric_only=True).to_dict()
        else:
            scores = dict(result)

        for k, v in scores.items():
            try:
                mlflow.log_metric(k, float(v))
            except (TypeError, ValueError):
                pass

        mlflow.log_metric("latency_seconds", latency)
        mlflow.log_param("retriever_type", retriever_type)
        mlflow.log_param("top_k", top_k)

        if extra_params:
            for k, v in extra_params.items():
                mlflow.log_param(k, str(v))
