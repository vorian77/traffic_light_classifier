import numpy as np
import cv2
import extractor


class ExtractorBrightness(extractor.Extractor):
    """
    Extracts the total amount of brightness found in each of the
    estimated abstract regions of a 3 aspect traffic light image.
    """

    def __init__(self):
        super().__init__()
        self.name = "brightness"

    def prep_image(self, sensor_data, aspect_idx):
        image = super().prep_image(sensor_data, aspect_idx)

        # crop to estimated aspect region
        boundaries = sensor_data["boundaries"]
        top_row = boundaries[aspect_idx]
        bottom_row = boundaries[aspect_idx + 1]
        image = image[top_row:bottom_row + 1]

        # convert abstract region to hsv channel v to extract brightness
        aspect_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return aspect_image[:, :, 2]

    def get_score(self, image, aspect_idx):
        # return the total brightness in the specified aspect region
        return np.sum(image)
