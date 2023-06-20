import segmentation_models_pytorch as smp
import torch
from lightning import LightningModule
from segmentation_models_pytorch.encoders import get_preprocessing_fn

from star_analysis.model.neural_networks.loss import FocalLoss


class UNetLightningModule(LightningModule):
    def __init__(self, image_shape: tuple[int, int], num_classes: int):
        super().__init__()
        self.image_shape = image_shape
        self.num_classes = num_classes

        # Instantiate the FCN model
        self.model = smp.Unet(
            encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=5,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=self.num_classes,                      # model output channels (number of classes in your dataset)
        )
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True, log_loss=False)
        self.preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')
        self.outputs_train = []
        self.outputs_val = []

    def forward(self, x):
        #x = self.preprocess_input(x)
        return self.model(x)

    def shared_step(self, batch, stage):
        inputs, labels = batch
        image, mask = inputs.permute(0, 3, 2, 1), labels.permute(0, 3, 2, 1)

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(flatten_image(logits_mask, self.num_classes), flatten_image(mask, self.num_classes))

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(flatten_image(pred_mask, self.num_classes).long(), flatten_image(mask, self.num_classes).long(), mode="multiclass")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        result = self.shared_step(batch, stage="train")
        self.outputs_train.append(result)
        #self.log_dict(result)
        return result

    def validation_step(self, batch, batch_idx):
        result = self.shared_step(batch, stage="val")
        self.outputs_val.append(result)
        #self.log_dict(result)
        return result

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.outputs_train, "train")
        self.outputs_train = []

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.outputs_val, "val")
        self.outputs_val = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def flatten_image(image: torch.Tensor, num_classes) -> torch.Tensor:
    return image.contiguous().view(image.size(0), num_classes, -1)
