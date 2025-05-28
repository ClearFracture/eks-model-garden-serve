import os

from typing import Dict, Optional, List
import logging

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_models import (
    LoRAModulePath,
    PromptAdapterPath,
    OpenAIServingModels,
    BaseModelPath
)
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.logger import RequestLogger

logger = logging.getLogger("ray.serve")

chat_app = FastAPI()
embed_app = FastAPI()


def parse_vllm_args(cli_args: Dict[str, str]):
    """Parses vLLM args based on CLI inputs.

    Currently uses argparse because vLLM doesn't expose Python models for all of the
    config options we want to support.
    """
    arg_parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )

    parser = make_arg_parser(arg_parser)
    arg_strings = []
    for key, value in cli_args.items():
        if isinstance(value, bool) and value:
            arg_strings.append(f"--{key}")
        elif isinstance(value, str) and value.lower() == "true":
            arg_strings.append(f"--{key}")
        elif value is not None:
            arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args


# Chat Completion Application
@serve.deployment(name="VLLMDeployment")
@serve.ingress(chat_app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        prompt_adapters: Optional[List[PromptAdapterPath]] = None,
        request_logger: Optional[RequestLogger] = None,
        chat_template: Optional[str] = None,
        # enable_auto_tools: bool = False,
        # tool_parser: Optional[str] = None,
    ):
        logger.info(f"Starting with engine args: {engine_args}")
        self.openai_serving_chat = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.prompt_adapters = prompt_adapters
        self.request_logger = request_logger
        self.chat_template = chat_template
        # self.enable_auto_tools = enable_auto_tools
        # self.tool_parser = tool_parser
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @chat_app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()

            if self.engine_args.served_model_name is not None:
                base_model_paths = [BaseModelPath(name=self.engine_args.served_model_name,
                                                model_path=self.engine_args.served_model_name)]
            else:
                base_model_paths = [BaseModelPath(name=self.engine_args.model,
                                                model_path=self.engine_args.model)]

            models = OpenAIServingModels(
                engine_client=self.engine,
                model_config=model_config,
                base_model_paths=base_model_paths,
                lora_modules=self.lora_modules,
                prompt_adapters=self.prompt_adapters,
            )

            self.openai_serving_chat = OpenAIServingChat(
                engine_client=self.engine,
                model_config=model_config,
                models=models,
                response_role=self.response_role,
                request_logger=self.request_logger,
                chat_template=self.chat_template,
                chat_template_content_format="auto",
                enable_auto_tools=True,
                tool_parser="llama3_json",
            )
        logger.info(f"Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())


# Embedding Application
@serve.deployment(name="VLLMEmbeddingDeployment")
@serve.ingress(embed_app)
class VLLMEmbeddingDeployment:
    def __init__(self, engine_args: AsyncEngineArgs):
        logger.info(f"Starting embedding engine with args: {engine_args}")
        self.engine_args = engine_args
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.openai_serving_embedding = None

    @embed_app.post("/v1/embeddings")
    async def create_embedding(self, request: EmbeddingRequest, raw_request: Request):
        if not self.openai_serving_embedding:
            model_config = await self.engine.get_model_config()

            # Create base model paths similar to chat deployment
            if self.engine_args.served_model_name is not None:
                base_model_paths = [BaseModelPath(name=self.engine_args.served_model_name,
                                                model_path=self.engine_args.served_model_name)]
            else:
                base_model_paths = [BaseModelPath(name=self.engine_args.model,
                                                model_path=self.engine_args.model)]
            
            # Create models instance
            models = OpenAIServingModels(
                engine_client=self.engine,
                model_config=model_config,
                base_model_paths=base_model_paths,
                lora_modules=None,
                prompt_adapters=None,
            )

            self.openai_serving_embedding = OpenAIServingEmbedding(
                engine_client=self.engine,
                model_config=model_config,
                models=models,
                request_logger=None,
                chat_template=None,
                chat_template_content_format="auto",
            )

        logger.info(f"Embedding Request: {request}")
        response = await self.openai_serving_embedding.create_embedding(
            request, raw_request
        )
        return JSONResponse(content=response.model_dump())


def build_chat_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Chat Serve application."""
    temp_cli_args = cli_args.copy()
    temp_cli_args["max-model-len"] = "30000"
    parsed_args = parse_vllm_args(temp_cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True

    return VLLMDeployment.bind(
        engine_args,
        parsed_args.response_role,
        parsed_args.lora_modules,
        parsed_args.prompt_adapters,
        cli_args.get("request_logger"),
        parsed_args.chat_template,
        # parsed_args.enable_auto_tools,
        # parsed_args.tool_parser
    )


def build_embedding_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Embedding Serve application."""
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True
    engine_args.task = "embed"  # Force task to embed for embedding model

    return VLLMEmbeddingDeployment.bind(engine_args)


# Create the chat model application by default
model_args = {
    "model": os.environ['MODEL_ID'], 
    "tensor-parallel-size": os.environ['TENSOR_PARALLELISM'], 
    "pipeline-parallel-size": os.environ['PIPELINE_PARALLELISM'],
    "dtype": os.environ.get('DTYPE', 'float16'),
    "enable_auto_tool": True,
    "tool_parser": "llama3_json",
    # "chat-template": "tool_chat_template_llama3.1_json.jinja"
}

# For backwards compatibility, keep the default 'model' export
# model = build_chat_app(model_args)

# Also export named applications for direct import in RayService config
chat_model = build_chat_app(model_args)
embedding_model = build_embedding_app(model_args)