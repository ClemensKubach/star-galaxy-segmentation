from enum import Enum


class BaseEnum(Enum):
    @classmethod
    def values(cls) -> list:
        return [i.value for i in cls.items()]

    @classmethod
    def items(cls) -> list:
        return sorted([i for i in cls])
