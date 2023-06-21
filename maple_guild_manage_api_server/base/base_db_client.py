from abc import ABC, abstractmethod

class BaseDBClient(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def insert_query(
        self,
        sql: str,
        args: dict,
    ):
        pass

    @abstractmethod
    def select_query(
        self,
        sql: str,
        args: dict,
    ):
        pass

    @abstractmethod
    def update_query(
        self,
        sql: str,
        args: dict,
    ):
        pass
