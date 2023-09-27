# TODO: guild_ui와 guild_contents중 하나에 마우스가 올라가 있거나 인식 못할 때 처리

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import List, Tuple, Union

import numpy as np
from PIL import Image

from ..base.base_model import BaseModel

PRIME_NUMBER = np.uint64(1000000007)


def pow_uint64(base, exponent: int):
    if exponent == 0:
        return np.uint64(1)
    val = pow_uint64(base, exponent >> 1)
    if exponent & 1:
        return val * val * base
    return val * val


@dataclass
class RoiExtractorModelConfig:
    standard_guild_ui_img_path: str
    standard_guild_contents_img_path: str


@dataclass
class RegionOnInterest:
    num: int
    names: List[Image.Image]
    jobs: List[Image.Image]
    levels: List[Image.Image]
    authorities: List[Image.Image]
    week_mission_points: List[Image.Image]
    suro_points: List[Image.Image]
    flag_points: List[Image.Image]


class RoiExtractorModel(BaseModel):
    def __init__(
        self,
        config: RoiExtractorModelConfig,
    ):
        self.config = config
        np.seterr(over="ignore")
        self.standard_guild_ui_img = Image.open(self.config.standard_guild_ui_img_path)
        self.standard_guild_contents_img = Image.open(
            self.config.standard_guild_contents_img_path
        )

        self.ready = True

    def b64_str_to_PIL_image(self, b64_str: str) -> Image.Image:
        image_data = base64.b64decode(b64_str)
        image_io = BytesIO(image_data)
        return Image.open(image_io)
    
    def PIL_image_to_b64_str(self, img: Image.Image) -> str:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


    def inference(
        self,
        target_image: Image.Image,
    ) -> RegionOnInterest:
        (guild_ui_x, guild_ui_y) = self._template_matching(
            target_img=target_image,
            template_img=self.standard_guild_ui_img,
        )
        (guild_contents_x, guild_contents_y) = self._template_matching( # FIXME: 여기에 마우스가 올라가 있는 경우 버그가 발생
            target_img=target_image,
            template_img=self.standard_guild_contents_img,
        )
        (standard_x, standard_y) = self._get_standard_cordinate(
            guild_ui_x=guild_ui_x,
            guild_ui_y=guild_ui_y,
            guild_contents_x=guild_contents_x,
            guild_contents_y=guild_contents_y,
        )
        roi = self._get_region_on_interest(
            target_image=target_image,
            std_x=standard_x,
            std_y=standard_y,
        )

        return roi

    def _get_region_on_interest(
        self,
        target_image: Image.Image,
        std_x: int,
        std_y: int,
    ) -> RegionOnInterest:
        # TODO: black 이미지 제거 (*현재는 무조건 17장이 추출됨.)
        USER_NUM = 17

        accumulated_w_sum = std_x
        default_kwargs = {
            "target_image": target_image,
            "std_y": std_y,
            "user_num": USER_NUM,
        }

        # extract name from taget_image.
        NAME_BOX_W = 70
        accumulated_w_sum, names = self._get_images(
            **default_kwargs,
            accumulated_w_sum=accumulated_w_sum,
            region_box_w=NAME_BOX_W,
        )
        # extract job from taget_image.
        JOB_BOX_W = 84
        accumulated_w_sum, jobs = self._get_images(
            **default_kwargs,
            accumulated_w_sum=accumulated_w_sum,
            region_box_w=JOB_BOX_W,
        )
        # extract level from taget_image.
        LEVEl_BOX_W = 23
        accumulated_w_sum, levels = self._get_images(
            **default_kwargs,
            accumulated_w_sum=accumulated_w_sum,
            region_box_w=LEVEl_BOX_W,
        )
        # extract authorities from taget_image.
        AUTHORITY_BOX_W = 78
        accumulated_w_sum, authorities = self._get_images(
            **default_kwargs,
            accumulated_w_sum=accumulated_w_sum,
            region_box_w=AUTHORITY_BOX_W,
        )

        # extract week_mission_point from taget_image.
        WEEK_MISSION_POINT_BOX_W = 36
        accumulated_w_sum, week_mission_points = self._get_images(
            **default_kwargs,
            accumulated_w_sum=accumulated_w_sum,
            region_box_w=WEEK_MISSION_POINT_BOX_W,
        )

        # extract suro_point from taget_image.
        SURO_POINT_BOX_W = 80
        accumulated_w_sum, suro_points = self._get_images(
            **default_kwargs,
            accumulated_w_sum=accumulated_w_sum,
            region_box_w=SURO_POINT_BOX_W,
        )

        # extract flag_point from taget_image.
        FLAG_POINT_BOX_W = 80
        accumulated_w_sum, flag_points = self._get_images(
            **default_kwargs,
            accumulated_w_sum=accumulated_w_sum,
            region_box_w=FLAG_POINT_BOX_W,
        )

        return RegionOnInterest(
            num=USER_NUM,
            names=names,
            jobs=jobs,
            levels=levels,
            authorities=authorities,
            week_mission_points=week_mission_points,
            suro_points=suro_points,
            flag_points=flag_points,
        )

    def _get_images(
        self,
        target_image: Image.Image,
        accumulated_w_sum: int,
        std_y: int,
        region_box_w: int,
        user_num: int,
    ) -> Tuple[int, List[Image.Image]]:
        DELTA_X, DELTA_Y = 205, 125
        BOX_H = 24
        nxt_accumulated_w_sum = accumulated_w_sum + region_box_w
        images = [
            target_image.crop(
                (
                    accumulated_w_sum + DELTA_X,
                    std_y + DELTA_Y + BOX_H * i,
                    nxt_accumulated_w_sum + DELTA_X,
                    std_y + DELTA_Y + BOX_H * i + BOX_H,
                )
            )
            for i in range(user_num)
        ]
        return (nxt_accumulated_w_sum, images)

    def _get_standard_cordinate(
        self,
        guild_ui_x: int,
        guild_ui_y: int,
        guild_contents_x: int,
        guild_contents_y: int,
    ) -> Tuple[int, int]:
        EXPECTED_DELTA_X, EXPECTED_DELTA_Y = (21, 370)

        if guild_ui_x < 0 and guild_contents_x < 0:
            return -1, -1
        if guild_ui_x < 0:
            std_x = guild_contents_x - EXPECTED_DELTA_X
            std_y = guild_contents_y - EXPECTED_DELTA_Y
            return std_x, std_y
        return guild_ui_x, guild_ui_y

    def _template_matching(
        self,
        target_img: Image.Image,
        template_img: Image.Image,
    ) -> Tuple[int, int]:
        """Get target image's left upper corner coordinate of template image by hashing. if not exist, return (-1, -1)."""
        # TODO: functionalize
        i64_prime = np.uint64(PRIME_NUMBER)

        target_w, target_h = target_img.size
        template_w, template_h = template_img.size
        if target_w < template_w or target_h < template_h:
            return (-1, -1)

        target_arr = self._hash_arr_of(target_img, std_h=template_h, prime=i64_prime)
        template_arr = self._hash_arr_of(
            template_img, std_h=template_h, prime=i64_prime
        )

        sum_target_arr = self._make_sum_arr(target_arr)
        sum_template_arr = self._make_sum_arr(template_arr)
        template_sum = sum_template_arr[template_h][template_w]

        now_x = np.uint64(1)
        std_h = pow_uint64(i64_prime, template_h)
        np.uint64(1)
        for x in range(target_w - template_w + 1):
            now_y = now_x
            for y in range(target_h - template_h + 1):
                target_sum = self._get_sum(
                    sum_target_arr, x, y, x + template_w - 1, y + template_h - 1
                )
                template_value = template_sum * now_y
                now_y *= i64_prime
                if target_sum == template_value:
                    return (x, y)
            now_x = now_x * std_h
        return (-1, -1)

    def _hash_arr_of(
        self, img: Image.Image, std_h: int, prime: Union[int, np.uint64] = PRIME_NUMBER
    ) -> np.ndarray:
        w, h = img.size
        i64_prime = np.uint64(prime)
        arr = np.array(img, dtype=np.uint64)
        arr_2d = (arr[:, :, 0] << 16) | (arr[:, :, 1] << 8) | (arr[:, :, 2])

        pow_arr = np.zeros((h, w), dtype=np.uint64)
        now_x = np.uint64(1)
        k = pow_uint64(prime, std_h)
        for j in range(w):
            now_y = now_x
            for i in range(h):
                pow_arr[i, j] = now_y
                now_y = now_y * i64_prime
            now_x = now_x * k
        return arr_2d * pow_arr

    def _make_sum_arr(self, arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape
        sum_arr = np.zeros((h + 1, w + 1), dtype=np.uint64)
        for i in range(1, h + 1):
            for j in range(1, w + 1):
                sum_arr[i, j] = (
                    sum_arr[i - 1, j]
                    + sum_arr[i, j - 1]
                    - sum_arr[i - 1, j - 1]
                    + arr[i - 1, j - 1]
                )
        return sum_arr

    def _get_sum(
        self, sum_arr: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> np.uint64:
        return (
            sum_arr[y2 + 1][x2 + 1]
            - sum_arr[y1][x2 + 1]
            - sum_arr[y2 + 1][x1]
            + sum_arr[y1][x1]
        )
