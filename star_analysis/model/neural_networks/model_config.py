from dataclasses import dataclass

from lightning import LightningModule
from torch.nn import Module

from star_analysis.model.neural_networks.losses.cornernet_loss import CornerNetLoss
from star_analysis.model.neural_networks.losses.da_dice_loss import DADiceLoss
from star_analysis.model.neural_networks.losses.da_focal_loss import DAFocalLoss
from star_analysis.model.neural_networks.losses.da_mse_loss import DAMseLoss
from star_analysis.model.neural_networks.losses.dice_loss import DiceLoss
from star_analysis.model.neural_networks.losses.focal_loss import FocalLoss
from star_analysis.model.neural_networks.losses.mse_loss import MseLoss
from star_analysis.model.neural_networks.losses.types import LossType
from star_analysis.model.types import ModelTypes


@dataclass
class ModelConfig:
    learning_rate: float
    batch_size: int
    image_shape: tuple[int, int]
    num_classes: int
    model_type: ModelTypes = ModelTypes.UNET
    model_module: LightningModule | None = None
    loss_type: LossType = LossType.DA_FOCAL
    loss_mode: str | None = None
    loss_module: Module | None = None

    def get_model(self, loss: Module, run_id: int | None = None):
        match self.model_type:
            case ModelTypes.FCN:
                raise NotImplementedError("FCN is not implemented yet.")
            case ModelTypes.DLV3:
                from star_analysis.model.neural_networks.deep_lab_v3 import DeepLabV3Model
                return DeepLabV3Model(loss=loss, config=self, run_id=run_id)
            case ModelTypes.UNET:
                from star_analysis.model.neural_networks.unet import UNetModel
                return UNetModel(loss=loss, config=self, run_id=run_id)
            case ModelTypes.CUSTOM:
                raise NotImplementedError("Custom model is not implemented yet.")
            case _:
                raise ValueError(f"Unknown model type {self}")

    def get_loss(self):
        match self.loss_type:
            case LossType.MSE:
                return MseLoss(
                    mode=self.loss_mode
                )
            case LossType.FOCAL:
                return FocalLoss(
                    mode=self.loss_mode,
                    num_classes=self.num_classes,
                )
            case LossType.DICE:
                return DiceLoss(
                    mode=self.loss_mode,
                    num_classes=self.num_classes,
                )
            case LossType.DA_MSE:
                return DAMseLoss(
                    mode=self.loss_mode
                )
            case LossType.DA_FOCAL:
                return DAFocalLoss(
                    mode=self.loss_mode,
                    num_classes=self.num_classes,
                    image_shape=self.image_shape
                )
            case LossType.DA_DICE:
                return DADiceLoss(
                    mode=self.loss_mode,
                    num_classes=self.num_classes
                )
            case LossType.CORNERNET:
                return CornerNetLoss(
                    mode=self.loss_mode,
                )
