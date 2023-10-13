import os
import json
from collections import defaultdict

import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from PIL import Image

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers.image_transforms import rgb_to_id, id_to_rgb
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import lightning.pytorch as pl

from download import download_file

PANOPTIC_URL = "https://cloud.ilabt.imec.be/index.php/s/3oiAJ5A52ZZDyiG/download/mask2former-pan-best-epoch=196.ckpt"
INSTANCE_URL = "https://cloud.ilabt.imec.be/index.php/s/3N59LiykEYXJn9t/download/mask2former-ins-best-epoch=195.ckpt"


class InstanceSegmenationDataset(Dataset):
    """Instance segmentation dataset."""

    def __init__(self, dataset, processor, transform=None):
        """
        Args:
            dataset
        """
        self.dataset = dataset
        self.images = dataset["images"]
        self.panoptic_labels = dataset["panoptic_labels"]
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        panoptic_seg_gt = self.panoptic_labels[idx]

        # apply transforms (need to be applied on RGB values)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=panoptic_seg_gt)
            image, panoptic_seg_gt = transformed["image"], transformed["mask"]
            # convert to C, H, W
            image = image.transpose(2, 0, 1)

        panoptic_seg_gt = rgb_to_id(panoptic_seg_gt)
        inst2class = {
            segment["id"]: segment["category_id"]
            for segment in self.dataset["segments"][idx]
        }

        inputs = self.processor(
            [image],
            [panoptic_seg_gt],
            instance_id_to_semantic_id=inst2class,
            return_tensors="pt",
        )
        inputs = {
            k: v.squeeze() if isinstance(v, torch.Tensor) else v[0]
            for k, v in inputs.items()
        }

        return inputs


class Mask2FormerModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.validation_step_outputs = []
        self.train_step_outputs = []

    def forward(self, **inputs):
        return self.model(**inputs)

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) and len(self.train_step_outputs):
            epoch_average = torch.stack(self.validation_step_outputs).mean()
            train_epoch_average = torch.stack(self.train_step_outputs).mean()
            print("Validation loss:", epoch_average, "Train loss:", train_epoch_average)
        self.validation_step_outputs.clear()  # free memory
        self.train_step_outputs.clear()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        # Forward pass
        outputs = self.model(
            pixel_values=batch["pixel_values"],
            mask_labels=batch["mask_labels"],
            class_labels=batch["class_labels"],
        )

        # Backward propagation
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=False, on_epoch=True)
        self.train_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        outputs = self.model(
            pixel_values=batch["pixel_values"],
            mask_labels=batch["mask_labels"],
            class_labels=batch["class_labels"],
        )

        # Backward propagation
        val_loss = outputs.loss
        self.validation_step_outputs.append(val_loss)
        self.log("val_loss", val_loss, prog_bar=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        return optimizer


def download_model(model_path, id2label, model_name="panoptic"):
    print("Loading base model from HuggingFace")
    if model_name == "panoptic":
        model_url = PANOPTIC_URL
        base_model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-tiny-coco-panoptic",
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )

    elif model_name == "instance":
        model_url = INSTANCE_URL
        base_model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-tiny-coco-instance",
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )
    else:
        raise NotImplementedError(
            f'Model: {model_name}, not found, choose one of:["panoptic", "instance"]'
        )
    if not os.path.exists(model_path):
        print("Downloading model weights")
        download_file(model_url, model_path)
    else:
        print("Loading model weights")

    model = Mask2FormerModel.load_from_checkpoint(model_path, model=base_model)

    return model


def get_transforms(img_width=600, img_height=800):
    image_transform = A.Compose(
        [
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.50,
                rotate_limit=45,
                p=0.75,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.Blur(blur_limit=3),
            A.GridDistortion(border_mode=cv2.BORDER_CONSTANT),
            # Resize and normalize last!!
            A.Resize(width=img_width, height=img_height),
            A.Normalize(),
        ]
    )

    validation_transform = A.Compose(
        [
            A.Resize(width=img_width, height=img_height),
            A.Normalize(),
        ]
    )

    return image_transform, validation_transform


def get_processor(model_name="panoptic", reduce_labels=True):

    if model_name == "panoptic":
        processor = AutoImageProcessor.from_pretrained(
            "facebook/mask2former-swin-tiny-coco-panoptic",
            ignore_index=0,
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            reduce_labels=reduce_labels,
        )
    elif model_name == "instance":
        processor = AutoImageProcessor.from_pretrained(
            "facebook/mask2former-swin-tiny-coco-instance",
            ignore_index=0,
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            reduce_labels=reduce_labels,
        )
    else:
        raise NotImplementedError(
            f'Model: {model_name}, not found, choose one of:["panoptic", "instance"]'
        )
    return processor


def draw_panoptic_segmentation(segmentation, segments, id2label):
    # get the used color map
    viridis = cm.get_cmap('viridis', torch.max(segmentation).numpy().astype(int))
    
    plt.imshow(segmentation)
    instances_counter = defaultdict(int)
    handles = []
    # for each segment, draw its legend
    for segment in segments:
        segment_id = segment['id']
        segment_label_id = segment['label_id']
        segment_label = id2label[segment_label_id]
        label = f"{segment_label}-{instances_counter[segment_label_id]}"
        instances_counter[segment_label_id] += 1
        color = viridis(segment_id)
        handles.append(mpatches.Patch(color=color, label=label))
        
    plt.legend(handles=handles,loc='center left', bbox_to_anchor=(1, 0.5))
