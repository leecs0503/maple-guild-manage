import asyncio

import uvicorn
from fastapi import FastAPI
from fastapi.routing import APIRoute

from ..errors import InvalidInput, invalid_input_handler
from .protocol import RestProtocol


class WebServer:
    def __init__(
        self,
        protocol: RestProtocol,
    ):
        self.protocol = protocol

    def create_application(self) -> FastAPI:
        return FastAPI(
            title="MapleGuildManageWebServer",
            routes=[
                # Server Liveness API returns 200 if server is alive.
                APIRoute("/", self.protocol.live),
                # Metrics
                APIRoute("/api/page_info", self.protocol.post_page_info, methods=["POST"]),
            ],
            exception_handlers={
                InvalidInput: invalid_input_handler,
            },
        )


class UvicornServer:
    def __init__(
        self,
        web_server: WebServer,
        http_port: int,
    ):
        self.web_server = web_server
        app = web_server.create_application()
        self.cfg = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=http_port,
        )

    def run_sync(self):
        server = uvicorn.Server(config=self.cfg)
        asyncio.run(server.serve())
