from pathlib import Path

from maple_guild_manage_api_server.model.roi_extractor_model import (
    RoiExtractorModel,
    RoiExtractorModelConfig,
)


def test_roi_extractor_model():
    TEST_IMG_PATH = Path(__file__).parent / "test_data" / "01_image.png"

    model_config = RoiExtractorModelConfig(
        standard_guild_ui_img_path="./resource/img/standard-ui-guild-ui.png",
        standard_guild_contents_img_path="./resource/img/standard-ui-guild-contents.png",
    )
    model = RoiExtractorModel(config=model_config)

    result = model.inference(TEST_IMG_PATH)
    print(result)
