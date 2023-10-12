import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import lightning.pytorch as pl
from torchmetrics.classification import BinaryJaccardIndex

from download import download_file

# in RGB
PLANT_COLOR = [5, 158, 79]

UNET_PATH = "models/unet_efficientnet-b0_best-epoch=196.ckpt"
UNETPLUS_PATH = "models/unetplus_efficientnet-b0_best-epoch=189.ckpt"
DEEPLAB_PATH = "models/deeplab_efficientnet-b0_best-epoch=199.ckpt"

UNET_URL = "https://cloud.ilabt.imec.be/index.php/s/Sr2aZPzBqskbcky/download/unet_efficientnet-b0_best-epoch=196.ckpt"
UNETPLUS_URL = "https://cloud.ilabt.imec.be/index.php/s/RJwpz3qLGGo3X5N/download/unetplus_efficientnet-b0_best-epoch=189.ckpt"
DEEPLAB_URL = "https://cloud.ilabt.imec.be/index.php/s/64J3k6mAs672LGM/download/deeplab_efficientnet-b0_best-epoch=199.ckpt"


class BinarySegmentationDataset(Dataset):
    """
    Binary plant segmentation dataset.
    """

    def __init__(self, dataset_dict, transform=None):
        self.images = dataset_dict["images"]
        self.masks = dataset_dict["masks"]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        # apply transforms (need to be applied on RGB values)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        return image, mask


class BinaryFileSegmentationDataset(Dataset):
    """
    Binary plant segmentation dataset from file paths.
    """

    def __init__(self, dataset_dict, transform=None):
        self.image_paths = dataset_dict["images"]
        self.mask_paths = dataset_dict["masks"]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], 0).astype(np.float32) / 255
        # apply transforms (need to be applied on RGB values)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        return image, mask


class InferenceDataset(Dataset):
    """
    For inference from list of filepaths
    """

    def __init__(self, filepaths, transform=None):
        # filepaths = list of img paths
        """
        Args:
            dataset
        """
        self.filepaths = filepaths
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.filepaths[idx]), cv2.COLOR_BGR2RGB)

        # apply transforms (need to be applied on RGB values)
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image


# Define the LightningModule
class BinaryModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.validation_step_outputs = []
        self.train_step_outputs = []
        self.validation_ious = []
        self.iou_fn = BinaryJaccardIndex(threshold=0.5)
        # self.loss_fn = smp.losses.FocalLoss(mode="binary")
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, img):
        return self.model(img)

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) and len(self.train_step_outputs):
            epoch_average = torch.stack(self.validation_step_outputs).mean()
            train_epoch_average = torch.stack(self.train_step_outputs).mean()
            val_iou = torch.stack(self.validation_ious).mean()
            print(
                "Validation loss:",
                epoch_average,
                "Validation IoU:",
                val_iou,
                "Train loss:",
                train_epoch_average,
            )
        self.validation_step_outputs.clear()  # free memory
        self.train_step_outputs.clear()
        self.validation_ious.clear()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        # Forward pass
        logits_mask = self.model(batch["img"])
        mask = batch["mask"]
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)
        # Backward propagation
        loss = loss
        self.log("train_loss", loss, prog_bar=False, on_epoch=True)
        self.train_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        logits_mask = self.model(batch["img"])
        mask = batch["mask"]
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        val_loss = self.loss_fn(logits_mask, mask)

        # prob_mask = logits_mask.sigmoid()
        prob_mask = logits_mask
        pred_mask = (prob_mask > 0.5).float()

        iou_score = self.iou_fn(pred_mask.squeeze(), mask)

        self.validation_step_outputs.append(val_loss)
        self.validation_ious.append(iou_score)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)
        self.log("val_iou", iou_score, prog_bar=True, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer


# Define custom collate function which defines how to batch examples together
def binary_collate_fn(batch):
    pixel_values = torch.stack([example[0] for example in batch])
    pixel_mask = torch.stack([example[1] for example in batch])

    return {"img": pixel_values, "mask": pixel_mask}


def inference_collate_fn(batch):
    return torch.stack([img for img in batch])


def get_transforms(img_width=608, img_height=800):
    image_transform = A.Compose(
        [
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            A.Rotate(),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=0.75
            ),
            A.Blur(blur_limit=3),
            A.GridDistortion(),
            # Resize and normalize last!!
            A.Resize(width=img_width, height=img_height),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    validation_transform = A.Compose(
        [
            A.Resize(width=img_width, height=img_height),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    return image_transform, validation_transform


def download_model(model_path, model_name="unetplus"):
    encoder = "efficientnet-b0"
    weights = "imagenet"
    activation = "sigmoid"
    in_channels = 3
    classes = 1
    print("Loading base model from SMP")
    if model_name == "unetplus":
        model_url = UNETPLUS_URL
        base_model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )
    elif model_name == "unet":
        model_url = UNET_URL
        base_model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )
    elif model_name == "deeplab":
        model_url = DEEPLAB_URL
        base_model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )
    else:
        raise NotImplementedError(
            f'Model: {model_name}, not found, choose one of:["unet", "unetplus", "deeplab"]'
        )
    if not os.path.exists(model_path):
        print("Downloading model weights")
        download_file(model_url, model_path)
    else:
        print("Loading model weights")

    model = BinaryModel.load_from_checkpoint(model_path, model=base_model)

    return model


def visualize_prediction(img, pred):
    out = img.copy()
    out[pred.astype(bool)] = PLANT_COLOR

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("Image")

    plt.subplot(1, 3, 2)
    plt.imshow(pred, cmap="gray")
    plt.axis("off")
    plt.title("Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(out, cmap="gray")
    plt.axis("off")
    plt.title("Prediction")


def get_dataloader(dataset, **kwargs):
    return DataLoader(
        dataset,
        shuffle=False,
        collate_fn=inference_collate_fn,
        **kwargs,
    )


def get_images_and_masks(
    img_dir, mask_dir, img_extension=".jpg", img_shape=(800, 608), load_images=True
):
    images = []
    masks = []
    img_h, img_w = img_shape
    for img_fname in os.listdir(img_dir):
        if img_fname.endswith(img_extension):
            img_path = os.path.join(img_dir, img_fname)

            mask_path1 = os.path.join(
                mask_dir, img_fname.replace(img_extension, "_plant_0.png")
            )
            mask_path2 = os.path.join(
                mask_dir, img_fname.replace(img_extension, ".png")
            )

            if os.path.exists(mask_path1):
                mask_path = mask_path1
            elif os.path.exists(mask_path2):
                mask_path = mask_path2
            else:
                print("No mask found for image:", img_path)
                continue

            if load_images:
                img = Image.open(img_path).resize((img_w, img_h), Image.NEAREST)
                label = Image.open(mask_path).resize((img_w, img_h), Image.NEAREST)

                images.append(np.array(img, dtype=np.uint8))
                masks.append(np.array(label, dtype=np.float32) / 255)
            else:
                images.append(img_path)
                masks.append(mask_path)

    if load_images:
        images = np.array(images, dtype=np.uint8)
        masks = np.array(masks, dtype=np.float32)

    return images, masks


def get_train_datasets(
    img_dir,
    mask_dir,
    img_dir_val,
    mask_dir_val,
    img_extension=".jpg",
    load_images=True,
    img_shape=(800, 608),
):
    train_dict = {}
    val_dict = {}
    img_h, img_w = img_shape
    print("Loading train dataset")
    images_train, masks_train = get_images_and_masks(
        img_dir=img_dir,
        mask_dir=mask_dir,
        img_extension=img_extension,
        load_images=True,
        img_shape=(800, 608),
    )

    train_dict["images"] = images_train
    train_dict["masks"] = masks_train
    print("Number of images train:", len(images_train))
    print("Loading validation dataset")
    images_val, masks_val = get_images_and_masks(
        img_dir=img_dir_val,
        mask_dir=mask_dir_val,
        img_extension=img_extension,
        load_images=True,
        img_shape=(800, 608),
    )
    val_dict["images"] = images_val
    val_dict["masks"] = masks_val
    print("Number of images validation:", len(images_val))

    image_transform, validation_transform = get_transforms(
        img_width=img_w, img_height=img_h
    )

    if load_images:
        train_dataset = BinarySegmentationDataset(train_dict, transform=image_transform)
        val_dataset = BinarySegmentationDataset(
            val_dict, transform=validation_transform
        )

    else:
        train_dataset = BinaryFileSegmentationDataset(
            train_dict, transform=image_transform
        )
        val_dataset = BinaryFileSegmentationDataset(
            val_dict, transform=validation_transform
        )

    return train_dataset, val_dataset, image_transform, validation_transform
