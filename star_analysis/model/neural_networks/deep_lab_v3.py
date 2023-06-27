import segmentation_models_pytorch as smp
from torch.nn import Module

from star_analysis.model.neural_networks.lightning import BaseLightningModule
from star_analysis.model.neural_networks.model_config import ModelConfig


class DeepLabV3Model(BaseLightningModule):
    def __init__(
            self,
            loss: Module,
            config: ModelConfig,
            run_id: int | None = None
    ):
        deeplab_v3 = smp.DeepLabV3(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=5,
            classes=config.num_classes,
        )
        super().__init__(
            architecture=deeplab_v3,
            loss=loss,
            config=config,
            run_id=run_id
        )
