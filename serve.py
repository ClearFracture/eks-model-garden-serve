import os
import json
from app.factory import (
    # build_chat_app,
    build_embedding_app,
    build_reranker_app,
)

# ---------- shared flags -------------------------------------------------
MODEL_ARGS = {
    "model":               os.environ["MODEL_ID"],
    "tensor-parallel-size": os.environ["TENSOR_PARALLELISM"],
    "pipeline-parallel-size": os.environ["PIPELINE_PARALLELISM"],
    "dtype":               os.environ.get("DTYPE", "float16"),
    # extras for vLLM tools / templates
    "enable_auto_tool":    True,
    "tool_parser":         "llama3_json",
}

# ---------- chat  ---------------------------------------------------
# chat_model      = build_chat_app(MODEL_ARGS)

# ---------- embedding  ---------------------------------------------------
embedding_model = build_embedding_app(MODEL_ARGS)

# ---------- reranker (extra overrides only here) -------------------------
# hf_overrides = {
#     "architectures": ["Qwen3ForSequenceClassification"],
#     "rename_classifier_to_score": True
# }

# RERANKER_ARGS = {
#     **MODEL_ARGS,                           # inherit the common flags
#     "task": "score",                        # make sure task is correct
#     "hf-config-overrides": json.dumps(hf_overrides),
# }

reranker_model = build_reranker_app(MODEL_ARGS)
