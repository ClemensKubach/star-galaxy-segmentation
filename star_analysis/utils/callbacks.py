from typing import Literal

import numpy as np
import torch
from PIL import Image
from lightning import Callback
from lightning.pytorch.loggers import TensorBoardLogger
import io
import matplotlib.pyplot as plt
from star_analysis.utils.conversions import relocate_channels


def _plotting(state: str, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0, only_idx: int = 0) -> None:
    def plot_obj(image, labels, predictions, obj: Literal["Galaxies", "Stars"]):
        image = torch.clamp(image, min=0, max=1).numpy()

        if obj == "Galaxies":
            obj_idx = 0
        elif obj == "Stars":
            obj_idx = 1
        else:
            raise ValueError("Not supported object type")

        coords_obj_pred = torch.nonzero(predictions[:, :, obj_idx])
        coords_obj_true = torch.nonzero(labels[:, :, obj_idx])

        fig, ax = plt.subplots()
        ax.imshow(image, origin='upper')
        ax.scatter(coords_obj_true[:, 0],
                   coords_obj_true[:, 1], c='Blue', label="True", s=3)
        ax.scatter(coords_obj_pred[:, 0],
                   coords_obj_pred[:, 1], c='Red', label="Pred", s=1)
        ax.legend()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close(fig)
        buffer.seek(0)
        # loaded_image = plt.imread(buffer, format='png') #np.flipud(plt.imread(buffer, format='png'))
        # print(loaded_image.shape)
        pil_image = Image.open(buffer)
        loaded_image = np.array(pil_image)

        tb_logger.add_image(f'{obj}-{state}', loaded_image,
                            global_step=trainer.global_step, dataformats='WHC')

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
