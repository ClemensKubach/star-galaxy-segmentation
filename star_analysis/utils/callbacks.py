import torch
from lightning import Callback
from lightning.pytorch.loggers import TensorBoardLogger
import io
import matplotlib.pyplot as plt
from star_analysis.utils.conversions import relocate_channels
import torchvision.io as tio


def _plotting(state: str, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0) -> None:
    inputs, labels = batch
    logits_mask = relocate_channels(pl_module(relocate_channels(inputs)))

    prob_mask = logits_mask.sigmoid() #.to('cpu', non_blocking=True)
    predictions = (prob_mask > 0.5).float()

    tb_logger = None
    for logger in trainer.loggers:
        if isinstance(logger, TensorBoardLogger):
            tb_logger = logger.experiment
            break

    if tb_logger is None:
        raise ValueError('TensorBoard Logger not found')

    inputs = inputs[0]
    labels = labels[0]
    predictions = predictions[0]

    # irg_image = inputs[:, :, 0:3]
    labels = labels[:, :, 0:2]

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

    for image, label, pred_ in zip(inputs, labels, prob_mask > 0.5):
        label = label[:, :, 0:2].to('cpu', non_blocking=True)
        image = torch.clip(image[:, :, (1, 2, 0)]).to(
            'cpu', dtype=int, non_blocking=True)  # irg order!

        coords_galaxies = torch.nonzero(pred_[:, :, 0])
        coords_stars = torch.nonzero(pred_[:, :, 1])

        plt.figure()
        plt.imshow(image.numpy())
        plt.scatter(coords_galaxies[:, 0],
                    coords_galaxies[:, 1], c='red', label="Galaxy", s=2)
        plt.scatter(coords_stars[:, 0],
                    coords_stars[:, 1], c='Blue', label="Star", s=2)
        plt.legend()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')

        pytorch_image = tio.decode_png(buffer)
        tb_logger.add_image(f'Predictions {state}', pytorch_image,
                            global_step=trainer.global_step, dataformats='WHC')


class PlottingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        # _plotting('train', trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        pass

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if batch_idx == 0:
            _plotting('val', trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0
    ) -> None:
        _plotting('test', trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
