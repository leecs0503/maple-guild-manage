from .model.roi_extractor_model import RoiExtractorModel, RoiExtractorModelConfig
from .server.dataplane import DataPlane
from .server.protocol import RestProtocol
from .server.server import UvicornServer, WebServer


def main():
    roi_model_config = RoiExtractorModelConfig(
        standard_guild_ui_img_path="./resource/img/standard-ui-guild-ui.png",
        standard_guild_contents_img_path="./resource/img/standard-ui-guild-contents.png",
    )
    roi_model = RoiExtractorModel(config=roi_model_config)
    dataplane = DataPlane(roi_model=roi_model)
    rest_protocol = RestProtocol(dataplane=dataplane)
    web_server = WebServer(protocol=rest_protocol)
    uvicorn_server = UvicornServer(web_server=web_server, http_port=8080)
    uvicorn_server.run_sync()


if __name__ == "__main__":
    main()
