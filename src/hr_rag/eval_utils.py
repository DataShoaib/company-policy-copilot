"""RAG evaluation helpers (Ragas metrics, MLflow logging, latency).

Import-time stderr/stdout are temporarily redirected while third-party
libraries (ragas/litellm/langchain/mlflow) are imported. Some of them write
deprecation notices straight to sys.stderr and ignore filterwarnings, so this
is the only way to guarantee zero warning noise in notebook output.
"""
import asyncio
import contextlib
import io
import logging
import os
import sys
import time
import warnings
from collections.abc import Callable

# --- suppress import-time noise from heavy deps BEFORE importing them ---
warnings.filterwarnings("ignore")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("LITELLM_LOG", "21")
for _logger_name in (
    "litellm",
    "LiteLLM",
    "httpx",
    "httpcore",
    "urllib3",
    "openai",
    "ragas",
    "datasets",
    "qdrant_client",
    "tenacity",
    "langchain",
    "langchain_core",
    "langchain_community",
    "mlflow",
    "mlflow_sflog",
):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

_import_out, _import_err = io.StringIO(), io.StringIO()
_stdout_fd, _stderr_fd = sys.stdout, sys.stderr
sys.stdout, sys.stderr = _import_out, _import_err  # swallow anything written during import
try:
    import mlflow
    import nest_asyncio
    from datasets import Dataset
    from langchain_community.chat_models import ChatLiteLLM
    from langchain_core.documents import Document
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    from hr_rag.config import (
        GROQ_API_KEY,
        GROQ_MODEL,
        MLFLOW_EXPERIMENT_NAME,
        MLFLOW_TRACKING_URI,
    )
    from hr_rag.formatting import format_docs
finally:
    sys.stdout, sys.stderr = _stdout_fd, _stderr_fd  # restore real streams

__all__ = ["evaluate_rag", "format_docs", "log_to_mlflow", "measure_latency"]


def measure_latency(chain, test_question: str = "What is the leave policy?") -> float:
    start = time.time()
    chain.invoke(test_question)
    return round(time.time() - start, 3)




def evaluate_rag(chain, get_docs_fn: Callable[[str], list[Document]], dataset: dict):
    nest_asyncio.apply()
    questions, answers, contexts = [], [], []
    _messages = []  # collected silently; only surfaced if every question fails

    for question in dataset["question"]:
        answer, docs = None, None
        for attempt in range(2):
            try:
                answer = chain.invoke(question)
                docs = get_docs_fn(question)
                break
            except Exception as e:  # noqa: BLE001 - one bad Q shouldn't blank the whole eval run
                if attempt == 0:
                    _messages.append(f"retrying in 20s ({e.__class__.__name__}): {question[:50]!r}")
                    time.sleep(20)
                else:
                    _messages.append(f"skipped (chain failed twice): {question[:60]!r} -> {e}")
        if not docs or answer is None:
            continue
        questions.append(question)
        answers.append(str(answer))
        contexts.append([str(d.page_content) for d in docs])

    if not questions:
        raise RuntimeError(
            "Every eval question failed — check API quota/connectivity before trusting any numbers.\n"
            + "\n".join(_messages)
        )

    truth_map = dict(zip(dataset["question"], dataset["answer"], strict=True))

    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": [truth_map[q] for q in questions],
    })

    from hr_rag.embeddings import get_embeddings

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY not set — add it to .env (Ragas judge needs it). "
            "Get a free key at https://console.groq.com"
        )
    judge_llm = ChatLiteLLM(
        model_name=f"groq/{GROQ_MODEL}",
        api_key=GROQ_API_KEY,
        temperature=0,
        max_retries=5,
        timeout=600,
        model_kwargs={"max_tokens": 4096},
    )
    print(f"RAGAS judge LLM: groq/{GROQ_MODEL} (Groq API)")
    ragas_llm = LangchainLLMWrapper(
        judge_llm, run_config=RunConfig(max_workers=1, timeout=300),
    )

    previous_loop = asyncio.get_event_loop()
    evaluation_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(evaluation_loop)
    try:
        # Redirect stdout+stderr so tqdm progress bars, LiteLLM prints and any
        # deprecation chatter never leak into the notebook output.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return evaluate(
                eval_dataset,
                metrics=[
                    faithfulness,
                    context_precision,
                    context_recall,
                    answer_relevancy,
                    answer_correctness,
                ],
                llm=ragas_llm,
                embeddings=get_embeddings(),
                run_config=RunConfig(max_workers=1, timeout=300),  # per-item timeout cut from 3600s
                is_async=False,
                raise_exceptions=False,  # skip a bad question rather than aborting the whole run
            )
    finally:
        evaluation_loop.close()
        asyncio.set_event_loop(previous_loop)


def log_to_mlflow(run_name: str, result, latency: float, retriever_type: str, top_k: int | None = None, extra_params: dict | None = None) -> None:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
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
                if top_k is not None:
                    mlflow.log_param("top_k", top_k)

                if extra_params:
                    for k, v in extra_params.items():
                        mlflow.log_param(k, str(v))
        except Exception as e:  # noqa: BLE001 - MLflow is best-effort; never let logging kill an eval run
            logging.getLogger("hr_rag.eval_utils").warning(
                "mlflow skipped for %r: %s: %s", run_name, e.__class__.__name__, e
            )
