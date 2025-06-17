import os
from app.factory import (
    # build_chat_app,
    build_embedding_app,
    build_reranker_app,
)

# ---- minimal user-tunable model flags -----------------------------
MODEL_ARGS = {
    "model":               os.environ["MODEL_ID"],
    "tensor-parallel-size": os.environ["TENSOR_PARALLELISM"],
    "pipeline-parallel-size": os.environ["PIPELINE_PARALLELISM"],
    "dtype":               os.environ.get("DTYPE", "float16"),
    # extras for vLLM tools / templates
    "enable_auto_tool":    True,
    "tool_parser":         "llama3_json",
}
# -------------------------------------------------------------------

# chat_model      = build_chat_app(MODEL_ARGS)
embedding_model = build_embedding_app(MODEL_ARGS)
reranker_model  = build_reranker_app(MODEL_ARGS)
