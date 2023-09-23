import pytest

from maple_guild_manage_api_server.server.dataplane import DataPlane
from maple_guild_manage_api_server.server.protocol import Image, RestProtocol


class DataPlaneForTest(DataPlane):
    def __init__(self):
        self.inference_count = 0

    def inference(self, b64_str: str) -> dict:
        self.inference_count += 1
        return {"result": "good"}


@pytest.fixture
def rest_protocol():
    dataplane_for_test = DataPlaneForTest()
    rest_protocol = RestProtocol(dataplane=dataplane_for_test)
    return rest_protocol


def test_live(rest_protocol: RestProtocol):
    assert rest_protocol.live() == "200 OK"


def test_inference(rest_protocol: RestProtocol[DataPlaneForTest]):
    test_image = Image(b64="test")
    assert rest_protocol.post_page_info(image=test_image) == {"result": "good"}
    assert rest_protocol.dataplane.inference_count == 1
