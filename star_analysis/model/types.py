from enum import auto
from strenum import StrEnum


class ModelTypes(StrEnum):
    CUSTOM = auto()
    FCN = auto()
    SEGNET = auto()
    UNET = auto()
