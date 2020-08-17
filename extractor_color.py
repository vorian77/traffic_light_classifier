import numpy as np
import cv2
import extractor
import helpers as h
import utilities as u

class ExtractorColor(extractor.Extractor):
    """
    Counts the number of pixels of specified colors found in each
    aspect area of an image of a 3 aspect traffic light
    """

    def __init__(self):
        super().__init__()
        self.name = "color"

        # hue ranges extracted from the testing data in a separate operation
        self.SIGNAL_HUES = {0: (120, 155),
                            1: (60, 119),
                            2: (19, 59)}

    def prep_image(self, sensor_data, aspect_idx):
        # apply masks that filter all but high saturation
        # colors specified for each aspect region
        image = super().prep_image(sensor_data, aspect_idx)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask_brightness = self.prep_image_mask_saturation(hsv, aspect_idx)
        mask_color = self.prep_image_mask_color(hsv, aspect_idx)
        mask_accumulative = np.bitwise_and(mask_brightness, mask_color)

        # return image with all non-masked pixels set to white ([255, 255, 255])
        masked_image = np.copy(image)
        masked_image[mask_accumulative != 255] = [255, 255, 255]
        return masked_image

    def prep_image_mask_saturation(self, hsv_image, aspect_idx):
        # return a mask for pixels with very high saturation
        # ie. values at least 1 standard deviation above the mean for the image
        hsv_s = hsv_image[:, :, 1]
        image_saturation_mean = int(np.mean(hsv_s))
        image_saturation_standard_deviation = int(np.std(hsv_s))
        lower = image_saturation_mean + image_saturation_standard_deviation
        upper = 255
        return cv2.inRange(hsv_s, lower, upper)

    def prep_image_mask_color(self, hsv_image, aspect_idx):
        # return mask for pixels with hues in range
        # specified for the aspect color
        range = self.SIGNAL_HUES[aspect_idx]
        lower = np.array([range[0], 0, 0])
        upper = np.array([range[1], 255, 255])
        return cv2.inRange(hsv_image, lower, upper)

    def get_score(self, image, aspect_idx):
        # return a count of the number of colored pixels remaining
        # in the image after the masks have been applied
        return np.count_nonzero(image != [255, 255, 255])
