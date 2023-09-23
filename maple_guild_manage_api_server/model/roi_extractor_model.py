# TODO: guild_ui와 guild_contents중 하나에 마우스가 올라가 있거나 인식 못할 때 처리

from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
from PIL import Image

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


class RoiExtractorModel:
    def __init__(
        self,
        config: RoiExtractorModelConfig,
    ):
        self.config = config

        self.standard_guild_ui_img = Image.open(self.config.standard_guild_ui_img_path)
        self.standard_guild_contents_img = Image.open(
            self.config.standard_guild_contents_img_path
        )

        self.ready = True

    def inference(
        self,
        img_path: str,
    ) -> RegionOnInterest:
        target_image = Image.open(img_path)
        (guild_ui_x, guild_ui_y) = self._template_matching(
            target_img=target_image,
            template_img=self.standard_guild_ui_img,
        )
        (guild_contents_x, guild_contents_y) = self._template_matching(
            target_img=target_image,
            template_img=self.standard_guild_contents_img,
        )
        (standard_x, standard_y) = self._get_standard_cordinate(
            guild_ui_x=guild_ui_x,
            guild_ui_y=guild_ui_y,
            guild_contents_x=guild_contents_x,
            guild_contents_y=guild_contents_y,
        )
        ROI = self._get_region_on_interest(
            target_image=target_image,
            std_x=standard_x,
            std_y=standard_y,
        )

        return ROI
    
    def _get_region_on_interest(
        self,
        target_image: Image.Image,
        std_x: int,
        std_y: int,
    ) -> RegionOnInterest:
        USER_NUM = 17
        DELTA_X, DELTA_Y = 205, 125
        BOX_H = 24

        accumulated_w_sum = std_x
        # extract name from taget_image.
        NAME_BOX_W = 70
        names = [
            target_image.crop((
                accumulated_w_sum + DELTA_X, std_y + DELTA_Y + BOX_H * i,
                accumulated_w_sum + DELTA_X + NAME_BOX_W, std_y + DELTA_Y + BOX_H * (i + 1)
                # h -(std_x + delta_x), w -(std_y + delta_y)
            ))
            for i in range(USER_NUM)
        ]
        accumulated_w_sum += NAME_BOX_W

        # TODO: black 이미지 제거
        return RegionOnInterest(
            num=USER_NUM,
            names=names,
        )

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
