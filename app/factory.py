from typing import Dict
from ray import serve

from app.config import parse_vllm_args
from app.deployments.chat import VLLMDeployment
from app.deployments.embedding import VLLMEmbeddingDeployment
from app.deployments.reranker import VLLMRerankerDeployment

def _base_engine_args(cli_args: Dict[str, str]):
    args = parse_vllm_args(cli_args)
    args.worker_use_ray = True
    return args

def build_chat_app(cli_args: Dict[str, str]) -> serve.Application:
    args = _base_engine_args({**cli_args, "max-model-len": "30000"})
    return VLLMDeployment.bind(
        args,
        args.response_role,
        args.lora_modules,
        args.prompt_adapters,
        cli_args.get("request_logger"),
        args.chat_template,
    )

def build_embedding_app(cli_args: Dict[str, str]) -> serve.Application:
    args = _base_engine_args(cli_args)
    args.task = "embed"
    return VLLMEmbeddingDeployment.bind(args)

def build_reranker_app(cli_args: Dict[str, str]) -> serve.Application:
    args = _base_engine_args(cli_args)
    args.task = "score"
    return VLLMRerankerDeployment.bind(args)
