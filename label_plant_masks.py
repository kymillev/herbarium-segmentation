import os
import json

import numpy as np
import cv2


def image_resize(image, width=None, height=None):
    # initialize the dimensions of the image to be resized and
    # grab the image size

    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim)

    # return the resized image
    return resized


data_dir = "datasets/sample"
img_dir = os.path.join(data_dir, "img")
out_dir = os.path.join(data_dir, "outputs_split")
mask_dir = os.path.join(data_dir, "plant_masks")

labels_fname = "plant_extraction_labels.json"
try:
    labels = json.load(open(labels_fname, "r"))
except FileNotFoundError:
    labels = {}

n_done = 0
resize = 1
fnames = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

i = 0
previous = None
previous_index = 0
while i < len(fnames):
    fname = fnames[i]
    img_name = fname.replace(".jpg", "")
    mask_path = os.path.join(mask_dir, img_name + ".png")
    output_path = os.path.join(out_dir, fname)

    if fname in labels:
        i += 1
        continue

    print("i:", i)
    print("Current:", fname)

    if os.path.exists(mask_path):
        w = 500
        img = cv2.imread(os.path.join(img_dir, fname))
        orig_img = image_resize(img.copy(), width=w)
        mask = cv2.imread(mask_path)
        output = cv2.imread(output_path)

        mask = image_resize(mask, width=w)
        output = image_resize(output, width=w)

        cv2.imshow("orig", orig_img)
        cv2.imshow("mask", mask)
        cv2.imshow("output", output)
        k = cv2.waitKey()

        # Press enter to label as correct enough for validation
        if k == 13:
            print("Image added to validation set")
            labels[fname] = 1
            previous = fname
            previous_index = i

        # Exit window to exit program
        elif k == -1:
            print("Saving labels and exiting")
            with open(labels_fname, "w") as f:
                json.dump(labels, f)
            exit()
        # Press escape for a mistake, go back to previous img
        elif k == 27:
            print("Mistake, go back to previous")
            if i > 0:
                del labels[previous]
                i = previous_index - 1

        # Press any other key to label as bad
        else:
            print("Bad mask, skip")
            labels[fname] = 0
            previous = fname

        i += 1
        n_done += 1
        with open(labels_fname, "w") as f:
            json.dump(labels, f)
    else:
        i += 1
