import numpy as np


class Extractor:
    """Template class for feature extractors"""

    def __init__(self):
        # name of the extractor - set in child class
        self.name = None
        self.signal_aspects = ("red", "yellow", "green")

    def extract(self, sensor_data):
        feature = {"feature_name":self.name,
                   "images":[[], [], []],
                   "scores":[0, 0, 0]}

        # process each signal aspect
        for aspect_idx in range(len(self.signal_aspects)):
            prepped_image = self.prep_image(sensor_data, aspect_idx)
            feature["images"][aspect_idx] = prepped_image
            feature["scores"][aspect_idx] = self.get_score(prepped_image, aspect_idx)
        feature["scores"] = self.normalize_scores(feature["scores"])
        feature["classification"] = self.get_classification(feature["scores"])
        return feature

    def prep_image(self, sensor_data, aspect_idx):
        return sensor_data["prepped_image"]

    def get_score(self, prepped_image, aspect_idx):
        pass

    def normalize_scores(self, scores):
        total_score = sum(scores)

        # edge case
        if total_score == 0:
            return [0, 0, 0]
        else:
            for i, s in enumerate(scores):
                scores[i] = round(s / total_score, 2)
            return scores

    def get_classification(self, scores):
        # edge case
        if sum(scores) == 0:
            return "unknown"
        else:
            return self.signal_aspects[int(np.argmax(scores))]
