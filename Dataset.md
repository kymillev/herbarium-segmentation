## Dataset info

Download link: [Main link](https://cloud.ilabt.imec.be/index.php/s/6zr8GcEBgJQqqzj/download/plantclef.zip) - [Google Drive mirror](https://drive.google.com/file/d/1WIaipfAyLMvhUf-VYsVHJWlI9Njg66PE/view?usp=sharing)

- 220 train images (1779 instances)
- 30 validation images (250 instances)
- ~60 MB when unzipped

Overview of the number of labeled object instances in the dataset and the percentage of images on which they occur.

| **Class** | **Train (%)** | **Validation (%)** |
|:------------------:|:-----------------------:|:----------------------------:|
| **Plant**              | **282 (100)**      | **34 (100)**                        |
| Note               | 551 (99.5)              | 83 (100)          |
| Barcode            | 231 (95.9)              | 34 (96.7)                    |
| Stamp              | 221 (80.5)              | 30 (80.0)                    |
| Ruler              | 216 (92.3)              | 29 (86.7)                    |
| Color card         | 145 (59.5)              | 21 (60.0)                    |
| Attachment         | 125 (56.8)              | 19 (63.3)                    |
| Other              | 8 (1.4)      | 0 (0)                        |


### Dataset structure
*The dataset will likely be expanded & restructured in the future*

Unzip and move to datasets/plantclef

- train/val => Images used for training & validation

**Binary**
- plant_masks_{train,val} => Binary plant label masks

**Instance**
- instance_herbaria_categories.json => Categories and ids per class (plant as instance class)
- instances_{train,val}_split => Binary instance masks
- instances_{train,val}_rle_split.json => Instances used for training & validation (COCO RLE format)

**Panoptic**

See [rgb_to_id](https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py#L73) and [MaskFormer tutorial](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/MaskFormer/Fine-tuning/Fine_tuning_MaskFormer_on_a_panoptic_dataset.ipynb) for an explanation on the panoptic RGB values. 

- panoptic_herbaria_categories.json => Categories and ids per class (plant as semantic class)
- panoptic_{train,val} => Panoptic label masks (plant semantic class)
- panoptic_{train,val}.json => Accompanying panoptic json files
- panoptic_{train,val}_split => Panoptic label masks (with plants as instance, needed for Mask2Former instance)
- panoptic_{train,val}_split.json => Accompanying panoptic json files
