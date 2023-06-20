import torch
import torchvision
from lightning import Callback
from lightning.pytorch.loggers import TensorBoardLogger


class PlottingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0
    ) -> None:
        self.plotting(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0
    ) -> None:
        self.plotting(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def plotting(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0) -> None:
        if batch_idx == 0 or True:
            inputs, labels = batch
            predictions = pl_module(inputs)

            inputs = inputs[0]
            labels = labels[0]
            predictions = predictions[0]

            irg_image = inputs[:, :, 0:3]
            labels = labels[:, :, 0:2]

            #print(predictions)

            shape = list(labels.shape[:2]) + [3]
            label_img = torch.full(shape, 255, dtype=torch.uint8, device=labels.device)
            label_img[labels[:, :, 0] == 1] = torch.tensor([0, 0, 255], dtype=torch.uint8, device=labels.device)
            label_img[labels[:, :, 1] == 1] = torch.tensor([0, 0, 255], dtype=torch.uint8, device=labels.device)
            label_img[predictions[:, :, 0] >= 0.5] = torch.tensor([255, 0, 0], dtype=torch.uint8, device=labels.device)
            label_img[predictions[:, :, 1] >= 0.5] = torch.tensor([0, 255, 0], dtype=torch.uint8, device=labels.device)

            tb_logger = None
            for logger in trainer.loggers:
                if isinstance(logger, TensorBoardLogger):
                    tb_logger = logger.experiment
                    break
            if tb_logger is None:
                raise ValueError('TensorBoard Logger not found')

            tb_logger.add_image(f'Predictions', label_img, global_step=trainer.global_step, dataformats='WHC')
