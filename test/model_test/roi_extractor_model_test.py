from pathlib import Path
from typing import List

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
        expected_result_names,
        expected_result_jobs,
        expected_result_levels,
        expected_result_authorities,
        expected_result_week_mission_points,
        expected_result_suro_points,
        expected_result_flag_points,
    ) = _get_expected_data(TEST_DATA_PATH / "expected_result")

    model_config = RoiExtractorModelConfig(
        standard_guild_ui_img_path="./resource/img/standard-ui-guild-ui.png",
        standard_guild_contents_img_path="./resource/img/standard-ui-guild-contents.png",
    )
    model = RoiExtractorModel(config=model_config)

    target_image = Image.open(str(TEST_DATA_PATH / "01_image.png"))
    result = model.inference(target_image)

    assert result.num == expected_result_num

    _assert_equal_of(result.names, expected_result_names, "names")
    _assert_equal_of(result.jobs, expected_result_jobs, "jobs")
    _assert_equal_of(result.levels, expected_result_levels, "levels")
    _assert_equal_of(result.authorities, expected_result_authorities, "authorities")
    _assert_equal_of(
        result.week_mission_points,
        expected_result_week_mission_points,
        "week_mission_points",
    )
    _assert_equal_of(result.suro_points, expected_result_suro_points, "suro_points")
    _assert_equal_of(result.flag_points, expected_result_flag_points, "flag_points")


def _assert_equal_of(
    result_imgs: List[Image.Image],
    expected_imgs: List[Image.Image],
    msg: str,
):
    for result_img, expected_name_img in zip(result_imgs, expected_imgs):
        assert list(result_img.getdata()) == list(expected_name_img.getdata()), msg


def _get_expected_data(path: Path):
    expected_result_num = 17
    expected_result_names = [
        Image.open(path / f"{i:02d}_name.png") for i in range(0, expected_result_num)
    ]
    expected_result_jobs = [
        Image.open(path / f"{i:02d}_job.png") for i in range(0, expected_result_num)
    ]
    expected_result_levels = [
        Image.open(path / f"{i:02d}_level.png") for i in range(0, expected_result_num)
    ]
    expected_result_authorities = [
        Image.open(path / f"{i:02d}_authority.png")
        for i in range(0, expected_result_num)
    ]
    expected_result_week_mission_points = [
        Image.open(path / f"{i:02d}_week_mission_point.png")
        for i in range(0, expected_result_num)
    ]
    expected_result_suro_points = [
        Image.open(path / f"{i:02d}_suro_point.png")
        for i in range(0, expected_result_num)
    ]
    expected_result_flag_points = [
        Image.open(path / f"{i:02d}_flag_point.png")
        for i in range(0, expected_result_num)
    ]
    return (
        expected_result_num,
        expected_result_names,
        expected_result_jobs,
        expected_result_levels,
        expected_result_authorities,
        expected_result_week_mission_points,
        expected_result_suro_points,
        expected_result_flag_points,
    )
