from enum import auto

from strenum import StrEnum


class ModelTypes(StrEnum):
    CUSTOM = auto()
    FCN = auto()
    DLV3 = auto()
    UNET = auto()
