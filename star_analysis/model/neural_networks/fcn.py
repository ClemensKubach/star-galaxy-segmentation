import torch
from lightning import LightningModule
from segmentation_models_pytorch.losses import FocalLoss, MULTILABEL_MODE
from torch import nn
from torchvision import models
from torchvision.models.segmentation.fcn import FCNHead


class FCN(nn.Module):
    def __init__(self, image_shape: tuple[int, int], num_classes: int):
        super(FCN, self).__init__()
        self.image_shape = image_shape
        self.num_classes = num_classes

        # Load the pretrained ResNet50 model
        self.fcn = models.segmentation.fcn_resnet50(
            pretrained=True
        )
        self.fcn.backbone.requires_grad_(False)
        self.fcn.backbone.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fcn.backbone.conv1.requires_grad_(True)

        # Modify the classifier (head) to match the number of output classes
        self.fcn.classifier = FCNHead(2048, num_classes)
        self.class_activation = nn.Sigmoid()
        # num_features = self.resnet.fc.in_features
        # self.resnet.fc = nn.Sequential(
        #     #nn.Linear(num_features, 256),
        #     #nn.ReLU(inplace=True),
        #     #nn.Linear(256, image_shape[0] * image_shape[1] * num_classes),
        #     nn.Linear(num_features, image_shape[0] * image_shape[1] * num_classes),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.fcn(x)['out']
        x = self.class_activation(x)
        x = x.view(x.size(0), self.image_shape[0], self.image_shape[1], self.num_classes)
        return x


class FCNLightningModule(LightningModule):
    def __init__(self, image_shape: tuple[int, int], num_classes: int):
        super(FCNLightningModule, self).__init__()

        # Instantiate the FCN model
        self.model = FCN(image_shape, num_classes)
        self.loss = FocalLoss(mode=MULTILABEL_MODE, reduction='mean')  # nn.BCELoss(reduction='mean')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)

        logits = logits.permute(0, 3, 1, 2).contiguous().view(2, -1)  # Shape: (num_classes, w*h)
        labels = labels.permute(0, 3, 1, 2).contiguous().view(2, -1)  # Shape: (num_classes, w*h)

        loss = self.loss(logits, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)

        logits = logits.permute(0, 3, 1, 2).contiguous().view(2, -1)  # Shape: (num_classes, w*h)
        labels = labels.permute(0, 3, 1, 2).contiguous().view(2, -1)  # Shape: (num_classes, w*h)

        loss = self.loss(logits, labels)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer
