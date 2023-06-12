from enum import auto
from strenum import StrEnum


class ModelTypes(StrEnum):
    CUSTOM = auto()
    UNET = auto()
    FCN = auto()
    SEGNET = auto()
