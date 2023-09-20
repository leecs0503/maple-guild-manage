from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image

PRIME_NUMBER = np.int64(1000000007)


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
    ):
        target_image = Image.open(img_path)
        (guild_ui_x, guild_ui_y) = self._template_matching(
            target_img=target_image,
            template_img=self.standard_guild_ui_img,
        )
        (guild_contents_x, guild_contents_x) = self._template_matching(
            target_img=target_image,
            template_img=self.standard_guild_contents_img,
        )
        print("!!", guild_ui_x, guild_ui_y)
        print("!!", guild_contents_x, guild_contents_x)

    def _template_matching(
        self,
        target_img: Image.Image,
        template_img: Image.Image,
    ) -> Tuple[int, int]:
        """Get target image's left upper corner coordinate of template image by hashing. if not exist, return (-1, -1)."""
        target_w, target_h = target_img.size
        template_w, template_h = template_img.size
        i64_prime = np.uint64(PRIME_NUMBER)
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
        k = pow_uint64(i64_prime, template_h)
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
            now_x = now_x * k
        return (-1, -1)

    def _hash_arr_of(
        self, img: Image.Image, std_h: int, prime: int = PRIME_NUMBER
    ) -> np.ndarray:
        w, h = img.size
        i64_prime = np.uint64(prime)
        arr = np.array(img, dtype=np.uint64)
        arr_2d = (arr[:, :, 0] << 16) | (arr[:, :, 1] << 8) | (arr[:, :, 2])

        pow_arr = np.zeros((h, w), dtype=np.uint64)
        now_x = np.uint64(1)
        k = pow_uint64(prime, std_h)
        np.uint64(1)
        for j in range(w):
            now_y = now_x
            for i in range(h):
                pow_arr[i][j] = now_y
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
