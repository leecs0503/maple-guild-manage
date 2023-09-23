from pathlib import Path
from PIL import Image

from maple_guild_manage_api_server.model.roi_extractor_model import (
    RoiExtractorModel,
    RoiExtractorModelConfig,
)


def test_roi_extractor_model():
    TEST_DATA_PATH = Path(__file__).parent / "test_data"

    expected_result_num = 17
    (
        expected_result_num,
        expected_name_imgs,
    ) = _get_expected_data(TEST_DATA_PATH / "expected_result")

    model_config = RoiExtractorModelConfig(
        standard_guild_ui_img_path="./resource/img/standard-ui-guild-ui.png",
        standard_guild_contents_img_path="./resource/img/standard-ui-guild-contents.png",
    )
    model = RoiExtractorModel(config=model_config)

    result = model.inference(str(TEST_DATA_PATH / "01_image.png"))


    assert result.num == expected_result_num

    for result_name_img, expected_name_img in zip(result.names, expected_name_imgs):
        assert list(result_name_img.getdata()) == list(expected_name_img.getdata())

def _get_expected_data(path: Path):
    expected_result_num = 17
    expected_result_names = [
        Image.open(path / f"{i:02d}_name.png")
        for i in range(0, expected_result_num)
    ]
    return (
        expected_result_num,
        expected_result_names,
    )
