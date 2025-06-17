import logging
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import JSONResponse
from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import RerankRequest, ErrorResponse
from vllm.entrypoints.openai.serving_score import ServingScores
from vllm.entrypoints.openai.serving_models import (
    BaseModelPath,
    OpenAIServingModels,
)

logger = logging.getLogger("ray.serve")
rerank_app = FastAPI()

@serve.deployment(name="VLLMRerankerDeployment")
@serve.ingress(rerank_app)
class VLLMRerankerDeployment:
    """OpenAI-compatible /v1/rerank endpoint."""
    def __init__(self, engine_args: AsyncEngineArgs):
        engine_args.task = "rerank"        # set task for the worker pool
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._openai: ServingScores | None = None

    # ---------- route ----------
    @rerank_app.post("/v1/rerank")
    async def rerank(self, request: RerankRequest, raw_request: Request):
        if not self._openai:
            cfg        = await self.engine.get_model_config()
            model_name = engine_args.served_model_name or engine_args.model
            models     = OpenAIServingModels(
                engine_client=self.engine,
                model_config=cfg,
                base_model_paths=[BaseModelPath(name=model_name,
                                                model_path=model_name)],
            )
            self._openai = ServingScores(
                engine_client=self.engine,
                model_config=cfg,
                models=models,
                request_logger=None,
            )

        resp = await self._openai.do_rerank(request, raw_request)

        if isinstance(resp, ErrorResponse):
            return JSONResponse(content=resp.model_dump(),
                                status_code=resp.code)

        return JSONResponse(content=resp.model_dump())
