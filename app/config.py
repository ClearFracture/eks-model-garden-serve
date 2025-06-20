import logging
from typing import Dict, List

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser

logger = logging.getLogger("ray.serve")

def parse_vllm_args(cli_args: Dict[str, str]):
    """
    Parse a plain-Python dict of CLI-style key/value pairs into vLLM's
    `AsyncEngineArgs`.  Booleans are handled automatically.
    """
    arg_strings: List[str] = []
    for key, value in cli_args.items():
        if isinstance(value, bool) and value:
            arg_strings.append(f"--{key}")
        elif isinstance(value, str) and value.lower() == "true":
            arg_strings.append(f"--{key}")
        elif value is not None:
            arg_strings += [f"--{key}", str(value)]

    logger.info("vLLM CLI args: %s", arg_strings)

    parser = make_arg_parser(
        FlexibleArgumentParser("vLLM OpenAI-Compatible RESTful API server.")
    )
    parsed = parser.parse_args(args=arg_strings)
    return AsyncEngineArgs.from_cli_args(parsed)
