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
        result = {
            "num": roi.num,
            "names": [self.roi_model.PIL_image_to_b64_str(img) for img in roi.names],
            "jobs": [self.roi_model.PIL_image_to_b64_str(img) for img in roi.jobs],
            "levels": [self.roi_model.PIL_image_to_b64_str(img) for img in roi.levels],
            "authorities": [self.roi_model.PIL_image_to_b64_str(img) for img in roi.authorities],
            "week_mission_points": [self.roi_model.PIL_image_to_b64_str(img) for img in roi.week_mission_points],
            "suro_points": [self.roi_model.PIL_image_to_b64_str(img) for img in roi.suro_points],
            "flag_points": [self.roi_model.PIL_image_to_b64_str(img) for img in roi.flag_points],
        }
        return result
