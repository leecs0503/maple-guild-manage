import fastapi
from fastapi import FastAPI
from fastapi.routing import APIRoute
from ..errors import InvalidInput, invalid_input_handler

class DataPlane:
    def __init__(self):
        pass

class WebServer:
    def __init__(
        self,
        dataplane: DataPlane,
    ):
        self.dataplane = dataplane

    def create_application(self) -> FastAPI:
        return FastAPI(
            title="MapleGuildManageWebServer",
              routes=[
                # Server Liveness API returns 200 if server is alive.
                APIRoute(r"/", self.dataplane.live),
                # Metrics

            ], exception_handlers={
                InvalidInput: invalid_input_handler,
            }
        )
