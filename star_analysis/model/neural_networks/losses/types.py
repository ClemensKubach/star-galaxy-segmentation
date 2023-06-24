from enum import auto

from strenum import StrEnum


class LossType(StrEnum):
    FOCAL = auto()
    DICE = auto()
    DA_MSE = auto()
    DA_FOCAL = auto()
    DA_DICE = auto()
    CORNERNET = auto()
    CUSTOM = auto()
