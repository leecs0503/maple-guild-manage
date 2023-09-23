from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def inference(self, *args, **kwargs):
        raise NotImplementedError("method 'inference' is not implemented.")
