import numpy as np
import cv2
import math
import extractor_color as ec
import extractor_brightness as eb
import helpers as h
import utilities as u


def classify(IMAGE_ITEM):
    classification = {}
    classification["data"] = pre_process(IMAGE_ITEM)
    classification["features"] = extract(classification["data"])
    classification["conclusion"] = evaluate(classification["features"])
    return classification

def pre_process(IMAGE_ITEM):
    # define data needed by extractors
    data = {}
    data["file_name"] = IMAGE_ITEM["file_name"]
    data["label"] = IMAGE_ITEM["label"]
    data["source_image"] = IMAGE_ITEM["image"]
    data["prepped_image"] = pre_process_resize(IMAGE_ITEM["image"])
    data["edged_image"] = pre_process_edge(data["prepped_image"])
    data["boundaries"] = pre_process_boundaries(data["edged_image"])
    return data

def extract(data):
    # extract features from data
    features = []
    extractors = [ec.ExtractorColor(), eb.ExtractorBrightness()]
    for e in extractors:
        features.append(e.extract(data))
    return features

def evaluate(features):
    conclusion = {}

    # sum feature results for final classification
    scores_sum = [0, 0, 0]
    for f in features:
        scores_sum = np.add(scores_sum, f["scores"])
    conclusion["scores"] = [round(s, 2) for s in scores_sum]
    if 0 == sum(scores_sum):
        conclusion["aspect"] = "no_conclusion"
        conclusion["label"] = [0, 0, 0]
    else:
        one_hot_encoded = [0, 0, 0]
        one_hot_encoded[int(np.argmax(scores_sum))] = 1
        conclusion["label"] = one_hot_encoded
        conclusion["aspect"] = ("red", "yellow", "green")[int(np.argmax(scores_sum))]
    return conclusion


## pre-process
def pre_process_resize(image):
    # resize image to 32x32px so that all processed images are the same size
    # https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image
    resized_image = cv2.resize(np.copy(image), dsize=(32, 32), interpolation=cv2.INTER_CUBIC)

    # crop horizontally to isolate actionable data
    h_crop_percent = 0.60
    width = len(resized_image[0])
    h_crop_size = int(width * h_crop_percent / 2)
    h_crop_left = h_crop_size
    h_crop_right = width - h_crop_size
    resized_image = resized_image[:, h_crop_left:h_crop_right, :]

    return resized_image

def pre_process_edge(source_image):
    # use Canny edge detection to identify the edges of the
    # traffic light object in the image

    # https: // www.geeksforgeeks.org / find - and -draw - contours - using - opencv - python /
    gray_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    edged_image = cv2.Canny(gray_image, 30, 200)
    return edged_image

def pre_process_boundaries(edged_image):
    boundaries = [0, 0, 0, 0]

    # top & bottom
    boundaries[0] = pre_process_boundaries_most_continuous_row(edged_image, False)
    boundaries[3] = pre_process_boundaries_most_continuous_row(edged_image, True)

    # aspect zones - include "rounding" pixels in upper zones
    boundaries[1] = boundaries[0] + math.ceil((boundaries[3] - boundaries[0]) / 3)
    boundaries[2] = boundaries[1] + math.ceil((boundaries[3] - boundaries[1]) / 2)
    return boundaries

def pre_process_boundaries_most_continuous_row(image, flip_image):
    pixels = [0] * len(image)
    max_depth_first_row_search = 10
    default_row = 3
    top_row = 0
    max_depth_first_edge = 3

    # check: direction of search
    if flip_image:
        image_copy = np.flip(image, axis=0)
    else:
        image_copy = np.copy(image)

    # identify first row with pixels
    for r, image_row in enumerate(image_copy):
        if r > max_depth_first_row_search:
            top_row = default_row
            break
        else:
            pixel_cnt = np.count_nonzero(image_row == 255)
            if pixel_cnt > 0:
                top_row = r
                break

    # determine row between top and max_depth with the most continuous pixels
    max_depth = top_row + max_depth_first_edge
    for r in range(top_row, max_depth + 1):
        cnt = 0
        image_row = image_copy[r]
        prev_pixel_continuous = False
        for pixel in image_row:
            if pixel == 255:
                if prev_pixel_continuous:
                    cnt += 1
                prev_pixel_continuous = True
            else:
                prev_pixel_continuous = False
        pixels[r] = cnt

    # check: direction of search
    if flip_image:
        pixels = np.flip(pixels, axis=0)
        default_row = len(image) - default_row

    # return default row, if no edges detected
    if sum(pixels) == 0:
        return default_row
    else:
        return np.argmax(pixels)



