import logging
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import JSONResponse
from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import EmbeddingRequest
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_models import (
    BaseModelPath,
    OpenAIServingModels,
)

logger = logging.getLogger("ray.serve")
embed_app = FastAPI()

@serve.deployment(name="VLLMEmbeddingDeployment")
@serve.ingress(embed_app)
class VLLMEmbeddingDeployment:
    """OpenAI-compatible /v1/embeddings endpoint."""
    def __init__(self, engine_args: AsyncEngineArgs):
        engine_args.task = "embed"
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._openai = None

    @embed_app.post("/v1/embeddings")
    async def create_embedding(self, request: EmbeddingRequest, raw_request: Request):
        if not self._openai:
            cfg = await self.engine.get_model_config()
            model_name = engine_args.served_model_name or engine_args.model
            models = OpenAIServingModels(
                engine_client=self.engine,
                model_config=cfg,
                base_model_paths=[BaseModelPath(name=model_name, model_path=model_name)],
            )
            self._openai = OpenAIServingEmbedding(
                engine_client=self.engine,
                model_config=cfg,
                models=models,
                request_logger=None,
                chat_template=None,
                chat_template_content_format="auto",
            )

        resp = await self._openai.create_embedding(request, raw_request)
        return JSONResponse(content=resp.model_dump())
