import analyze as a
import files as f
import classifier as c


def classify(IMAGE_LIST):
    classifications = []
    for i, IMAGE_ITEM in enumerate(IMAGE_LIST):
        classifications.append(c.classify(IMAGE_ITEM))
    return classifications

def main():
    DATA_SET = "training"
    #DATA_SET = "test"
    IMAGE_DIR = "traffic_light_images/" + DATA_SET
    IMAGE_LIST = f.load_dataset(IMAGE_DIR)

    classifications = classify(IMAGE_LIST)

    display_analysis_images = False
    display_misclassified_data_and_images = False

    a.display_analysis(DATA_SET, IMAGE_LIST, classifications, display_analysis_images)
    a.display_summary(DATA_SET, IMAGE_LIST, classifications, display_misclassified_data_and_images)


if __name__ == "__main__":
    main()

