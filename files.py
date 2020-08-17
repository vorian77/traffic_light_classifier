import os
import glob # library for loading images from a directory
import matplotlib.image as mpimg
import numpy as np
import cv2
import helpers as h

# This function loads in images and their labels and places them in a list
# The list contains all images and their associated labels
# For example, after data is loaded, im_list[0][:] will be the first image-label pair in the list
def load_dataset(image_dir):

    # Populate this empty image list
    image_idx = 0
    image_list = []
    image_type_red = ["red", 0]
    image_type_yellow = ["yellow", 1]
    image_type_green = ["green", 2]
    image_type_temp = ["temp", 2]
    image_types = [image_type_red, image_type_yellow, image_type_green]
    #image_types = [image_type_temp]

    # Iterate through each color folder
    for im_type in image_types:
        # Iterate through each image file in each image_type folder
        # glob reads in any image with the extension "image_dir/im_type/*"
        label = im_type[0]
        label_idx = im_type[1]

        for file in glob.glob(os.path.join(image_dir, label, "*")):
            # Read in the image
            im = mpimg.imread(file)

            # Check if the image exists/if it's been correctly read-in
            if not im is None:
                # Append the image, and it's type (red, green, yellow) to the image list
                image_item = {}
                image_item["file_name"] = file[file.index(".") - 3: file.index(".")]
                image_item["image_idx"] = image_idx
                image_item["label"] = label
                image_item["label_idx"] = label_idx
                image_item["image"] = im

                image_list.append(image_item)
                image_idx += 1
    return image_list

# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):

    ## TODO: Resize image and pre-process so that all "standard" images are the same size
    standard_im = np.copy(image)

    # resize image to 32x32px
    # https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image
    standard_im = cv2.resize(standard_im, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)

    return standard_im

