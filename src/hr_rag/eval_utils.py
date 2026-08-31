import asyncio
import contextlib
import io
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

from hr_rag.config import LLM_PROVIDER, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from hr_rag.formatting import format_docs

__all__ = ["evaluate_rag", "format_docs", "log_to_mlflow", "measure_latency"]


def measure_latency(chain, test_question: str = "What is the leave policy?") -> float:
    start = time.time()
    chain.invoke(test_question)
    return round(time.time() - start, 3)


# Free-tier LLM calls are slow and ragas' per-batch asyncio.timeout() would
# otherwise CANCEL those in-flight calls and dump an ugly TimeoutError /
# "Runner in Executor raised an exception" traceback (CancelledError is a
# BaseException, so raise_exceptions=False can't suppress it).
# Using a very generous timeout keeps evaluation clean AND robust.
EVAL_TIMEOUT_SECONDS = 3600  # effectively disable the asyncio timeout


def evaluate_rag(chain, get_docs_fn: Callable[[str], list[Document]], dataset: dict):
    nest_asyncio.apply()
    questions, answers, contexts = [], [], []

    # Skip failed questions entirely instead of recording a fake "error" row:
    # a literal "error" answer + ["error"] context scores 0 on every RAGAS
    # metric and silently drags the whole run's averages down.
    for question, ground_truth in zip(dataset["question"], dataset["answer"]):
        answer, docs = None, None
        # one retry before giving up -- transient 429/timeouts are common on free tiers
        for attempt in range(2):
            try:
                answer = chain.invoke(question)
                docs = get_docs_fn(question)
                break
            except Exception as e:  # noqa: BLE001 - one bad Q shouldn't blank the whole eval run
                if attempt == 0:
                    print(f"  retrying in 20s ({e.__class__.__name__}): {question[:50]!r}")
                    time.sleep(20)
                else:
                    print(f"  skipped (chain failed twice): {question[:60]!r} -> {e}")
        if not docs or answer is None:
            continue
        questions.append(question)
        answers.append(str(answer))
        contexts.append([str(d.page_content) for d in docs])

    if not questions:
        raise RuntimeError("Every eval question failed — check API quota/connectivity before trusting any numbers.")

    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": [dataset["answer"][dataset["question"].index(q)] for q in questions],
    })

    from hr_rag.embeddings import get_embeddings
    from hr_rag.llm import get_llm

    # Judge = the configured primary provider (LLM_PROVIDER). LiteLLM's
    # built-in retries absorb transient 429s; if the provider is truly down
    # the run fails loudly instead of producing quietly-wrong numbers.
    judge_llm = get_llm()
    print(f"RAGAS judge LLM: {LLM_PROVIDER}")
    ragas_llm = LangchainLLMWrapper(
        judge_llm, run_config=RunConfig(max_workers=1, timeout=EVAL_TIMEOUT_SECONDS),
    )

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
            run_config=RunConfig(max_workers=1, timeout=EVAL_TIMEOUT_SECONDS),
            is_async=False,
            raise_exceptions=False,  # one bad question shouldn't blank the whole run
        )
    finally:
        evaluation_loop.close()
        asyncio.set_event_loop(previous_loop)


def log_to_mlflow(run_name: str, result, latency: float, retriever_type: str, top_k: int = 3, extra_params: dict | None = None) -> None:
    # mlflow (and especially the DagsHub remote) prints noise like
    # "🏃 View run baseline at: ..." and "🧪 View experiment at: ..." to stdout
    # on every set_experiment / start_run. Silence it while still logging.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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
