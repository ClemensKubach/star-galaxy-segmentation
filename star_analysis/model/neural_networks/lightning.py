import segmentation_models_pytorch as smp
import torch
import torchmetrics
from lightning import LightningModule
from lightning.pytorch.cli import ReduceLROnPlateau
from segmentation_models_pytorch.utils.metrics import Accuracy

from star_analysis.model.neural_networks.model_config import ModelConfig
from star_analysis.utils.conversions import vectorize_image, relocate_channels


class BaseLightningModule(LightningModule):
    def __init__(
            self,
            architecture: torch.nn.Module,
            loss: torch.nn.Module,
            config: ModelConfig,
    ):
        super().__init__()

        assert config.loss_mode is not None

        # necessary for lightning tuner
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.optimizer = None

        self.loss_fn = loss
        self.architecture = architecture
        self.__config = config

        # for later use
        self._outputs_train = []
        self._outputs_val = []
        self._outputs_test = []

    @property
    def config(self) -> ModelConfig:
        return self.__config

    def forward(self, x):
        return self.architecture(x)

    def shared_step(self, batch, stage):
        inputs, labels = batch
        image, mask = relocate_channels(inputs), relocate_channels(labels)

        # Shape of the image should be (batch_size, num_channels, height, width)
        assert image.ndim == 4
        # Check that image dimensions are divisible by 32
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        # Shape of the mask should be [batch_size, num_classes, height, width]
        assert mask.ndim == 4
        # Check that mask values in between 0 and 1
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(
            output=vectorize_image(pred_mask, self.config.num_classes).long(),
            target=vectorize_image(mask, self.config.num_classes).long(),
            mode=self.config.loss_mode,
            num_classes=self.config.num_classes,
        )
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        tp_class0 = torch.sum(tp[:, 0])
        tp_class1 = torch.sum(tp[:, 1])

        fp_class0 = torch.sum(fp[:, 0])
        fp_class1 = torch.sum(fp[:, 1])

        fn_class0 = torch.sum(fn[:, 0])
        fn_class1 = torch.sum(fn[:, 1])

        tn_class0 = torch.sum(tn[:, 0])
        tn_class1 = torch.sum(tn[:, 1])

        f1_class0 = smp.metrics.f1_score(tp_class0, fp_class0, fn_class0, tn_class0, reduction="micro")
        f1_class1 = smp.metrics.f1_score(tp_class1, fp_class1, fn_class1, tn_class1, reduction="micro")

        console_metrics = {
            f"{stage}_loss": loss,
            f"{stage}_f1": f1,
            f"{stage}_f1_class0": f1_class0,
            f"{stage}_f1_class1": f1_class1,
        }
        additional_metrics = {
            f"{stage}_tp": torch.sum(tp),
            f"{stage}_fp": torch.sum(fp),
            f"{stage}_fn": torch.sum(fn),
            f"{stage}_tn": torch.sum(tn),
        }
        for k, v in console_metrics.items():
            self.log(k, v, prog_bar=True, logger=True)
        for k, v in additional_metrics.items():
            self.log(k, v, prog_bar=False, logger=True)
        # self.log(console_metrics, prog_bar=True, logger=True)
        # self.log(additional_metrics, prog_bar=False, logger=True)

    # def shared_epoch_end(self, outputs, stage):
    #     tp = torch.cat([x["tp"] for x in outputs])
    #     fp = torch.cat([x["fp"] for x in outputs])
    #     fn = torch.cat([x["fn"] for x in outputs])
    #     tn = torch.cat([x["tn"] for x in outputs])
    #
    #     dataset_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    #
    #     metrics = {
    #         f"{stage}_epoch_loss": torch.stack([x["loss"] for x in outputs]).mean(),
    #         f"{stage}_epoch_f1": dataset_f1,
    #     }
    #
    #     self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     return metrics

    def training_step(self, batch, batch_idx):
        self.shared_step(batch, stage="train")
        # self._outputs_train.append(result)

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, stage="val")
        # self._outputs_val.append(result)

    def test_step(self, batch, batch_idx):
        self.shared_step(batch, stage="test")
        # self._outputs_test.append(result)

    # def on_train_epoch_end(self):
    #     metrics = self.shared_epoch_end(self._outputs_train, "train")
    #     self._outputs_train = []
    #     #return metrics
    #
    # def on_validation_epoch_end(self):
    #     metrics = self.shared_epoch_end(self._outputs_val, "val")
    #     self._outputs_val = []
    #     #return metrics

    # def on_test_epoch_end(self):
    #     metrics = self.shared_epoch_end(self._outputs_test, "test")
    #     self._outputs_test = []
    #     return metrics

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(self.optimizer, monitor='train_loss')
        return {'optimizer': self.optimizer,
                'scheduler': scheduler,
                'monitor': "train_loss"
                }
