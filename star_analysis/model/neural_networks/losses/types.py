from enum import auto

from strenum import StrEnum


class LossType(StrEnum):
    MSE = auto()
    FOCAL = auto()
    DICE = auto()
    DA_MSE = auto()
    DA_FOCAL = auto()
    DA_DICE = auto()
    CORNERNET = auto()
    CUSTOM = auto()
