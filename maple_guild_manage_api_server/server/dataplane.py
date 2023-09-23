from dataclasses import asdict

from ..model.roi_extractor_model import RoiExtractorModel


class DataPlane:
    def __init__(
        self,
        roi_model: RoiExtractorModel,
    ):
        self.roi_model = roi_model

    def inference(
        self,
        b64_str: str,
    ) -> dict:
        target_image = self.roi_model.b64_str_to_PIL_image(b64_str=b64_str)
        roi = self.roi_model.inference(target_image=target_image)
        return asdict(roi)
