from typing import Literal

import numpy as np
import torch
import torchvision
from PIL import Image
from lightning import Callback
from lightning.pytorch.loggers import TensorBoardLogger
import io
import matplotlib.pyplot as plt
from star_analysis.utils.conversions import relocate_channels


def _plotting(state: str, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0, only_idx: int = 0) -> None:
    def plot_obj(image, labels, predictions, obj: Literal["Galaxies", "Stars"]):
        image = torch.clamp(image, min=0, max=1)

        if obj == "Galaxies":
            obj_idx = 0
        elif obj == "Stars":
            obj_idx = 1
        else:
            raise ValueError("Not supported object type")

        # coords_obj_pred = torch.nonzero(predictions[:, :, obj_idx])
        # coords_obj_true = torch.nonzero(labels[:, :, obj_idx])

        #shape = list(labels.shape[:2]) + [3]
        new_image = torch.clone(image) #torch.full(shape, 255, dtype=torch.uint8, device=labels.device)
        new_image[labels[:, :, obj_idx] == 1] = torch.tensor([0, 0, 1], dtype=torch.float32, device=labels.device)
        new_image[predictions[:, :, obj_idx] > 0.5] = torch.tensor([1, 0, 0], dtype=torch.float32, device=labels.device)
        new_image[(predictions[:, :, obj_idx] > 0.5) & (labels[:, :, obj_idx] == 1)] = torch.tensor([0, 1, 0], dtype=torch.float32, device=labels.device)

        tb_logger.add_image(f'{obj}-{state}', new_image, global_step=trainer.global_step, dataformats='WHC')

    def plot_objects(inputs, labels, predictions):
        image = inputs[:, :, [1, 2, 0]]
        plot_obj(image, labels, predictions, "Galaxies")
        plot_obj(image, labels, predictions, "Stars")

    inputs_batch, labels_batch = batch
    logits_mask = relocate_channels(pl_module(relocate_channels(inputs_batch)))

    prob_mask = logits_mask.sigmoid()
    predictions_batch = (prob_mask > 0.5).float()

    tb_logger = None
    for logger in trainer.loggers:
        if isinstance(logger, TensorBoardLogger):
            tb_logger = logger.experiment
            break
    if tb_logger is None:
        raise ValueError('TensorBoard Logger not found')

    if only_idx < 0:
        for inputs, labels, predictions in zip(inputs_batch, labels_batch, predictions_batch):
            plot_objects(inputs.cpu(), labels.cpu(), predictions.cpu())
    else:
        plot_objects(inputs_batch[only_idx].cpu(), labels_batch[only_idx].cpu(), predictions_batch[only_idx].cpu())


class PlottingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        if batch_idx == 0:
            _plotting('val', trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, only_idx=0)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        _plotting('test', trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, only_idx=-1)
