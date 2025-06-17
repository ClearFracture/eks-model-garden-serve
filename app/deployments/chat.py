import logging
from typing import Optional, List

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse
from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import (
    BaseModelPath,
    LoRAModulePath,
    PromptAdapterPath,
    OpenAIServingModels,
)
from vllm.entrypoints.logger import RequestLogger

logger = logging.getLogger("ray.serve")
chat_app = FastAPI()

@serve.deployment(name="VLLMDeployment")
@serve.ingress(chat_app)
class VLLMDeployment:
    """OpenAI-compatible /v1/chat/completions endpoint."""
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        prompt_adapters: Optional[List[PromptAdapterPath]] = None,
        request_logger: Optional[RequestLogger] = None,
        chat_template: Optional[str] = None,
    ):
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.prompt_adapters = prompt_adapters
        self.request_logger = request_logger
        self.chat_template = chat_template
        self._openai = None   # lazy-init

    # ---------- route ----------
    @chat_app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        if not self._openai:
            model_cfg  = await self.engine.get_model_config()
            model_name = engine_args.served_model_name or engine_args.model
            models = OpenAIServingModels(
                engine_client=self.engine,
                model_config=model_cfg,
                base_model_paths=[BaseModelPath(name=model_name, model_path=model_name)],
                lora_modules=self.lora_modules,
                prompt_adapters=self.prompt_adapters,
            )
            self._openai = OpenAIServingChat(
                engine_client=self.engine,
                model_config=model_cfg,
                models=models,
                response_role=self.response_role,
                request_logger=self.request_logger,
                chat_template=self.chat_template,
                chat_template_content_format="auto",
                enable_auto_tools=True,
                tool_parser="llama3_json",
            )

        result = await self._openai.create_chat_completion(request, raw_request)
        if isinstance(result, ErrorResponse):
            return JSONResponse(content=result.model_dump(), status_code=result.code)
        if request.stream:
            return StreamingResponse(content=result, media_type="text/event-stream")
        return JSONResponse(content=result.model_dump())
