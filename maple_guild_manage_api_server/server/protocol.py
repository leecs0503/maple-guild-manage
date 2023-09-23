from typing import Generic, TypeVar

from pydantic import BaseModel

from .dataplane import DataPlane


class Image(BaseModel):
    b64: str


DATAPLANE = TypeVar("DATAPLANE", bound=DataPlane)


class RestProtocol(Generic[DATAPLANE]):
    def __init__(
        self,
        dataplane: DATAPLANE,
    ):
        self.dataplane = dataplane

    def live(self):
        return "200 OK"

    def post_page_info(self, image: Image):
        return self.dataplane.inference(b64_str=image.b64)
