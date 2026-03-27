import argparse
import asyncio
import os

import uvicorn

from chainlit.auth import ensure_jwt_secret
from chainlit.cache import init_lc_cache
from chainlit.config import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    config,
    load_module,
)
from chainlit.markdown import init_markdown
from chainlit.server import app


def _assert_app_callbacks() -> None:
    if (
        not config.code.on_chat_start
        and not config.code.on_message
        and not config.code.on_audio_chunk
    ):
        raise RuntimeError(
            "Define at least one callback: on_chat_start, on_message, or on_audio_chunk."
        )


async def _serve(host: str, port: int, headless: bool, target: str) -> None:
    config.run.host = host
    config.run.port = port
    config.run.headless = headless
    config.run.module_name = target

    load_module(target)
    ensure_jwt_secret()
    _assert_app_callbacks()

    init_markdown(config.root)
    init_lc_cache()

    server_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        ws="auto",
        log_level="error",
    )
    server = uvicorn.Server(server_config)
    await server.serve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chainlit app without nest_asyncio patch")
    parser.add_argument("--target", default="app.py", help="Path to Chainlit module file")
    parser.add_argument("--host", default=os.getenv("CHAINLIT_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("CHAINLIT_PORT", DEFAULT_PORT)))
    parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(_serve(args.host, args.port, args.headless, args.target))
