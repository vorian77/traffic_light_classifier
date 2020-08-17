import numpy as np
import cv2
import math
import helpers as h
import classifier as c


# colors
def get_hue(color):
    BGR = np.uint8([[color]])
    HSV = cv2.cvtColor(BGR, cv2.COLOR_BGR2HSV)
    hue = HSV[0][0][0]
    return hue

def get_hues(image):
    hues = set()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_v = hsv[:, :, 2]
    image_brightness_mean = np.mean(hsv_v)
    image_brightness_std = np.std(hsv_v)

    for row in range(len(image)):
        for col in range(len(image[0])):
            pixel_brightness = hsv_v[row, col]
            if pixel_brightness > image_brightness_mean + image_brightness_std:
                # https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv
                color = image[row, col]
                hue = get_hue(color)
                hues.add(hue)
    return hues

def masked_by_hues(image):
    hues = (60,115)
    lower = np.array([hues[0], 0, 0])
    upper = np.array([hues[1], 255, 255])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    masked_image = np.copy(image)
    masked_image[mask == 255] = [255, 255, 255]
    return masked_image


def extract_hues(IMAGE_LIST):
    aspects = ["red", "yellow", "green"]
    color_maps = [{} for _ in range(len(aspects))]

    for image_item in IMAGE_LIST:
        data = c.pre_process(image_item)
        image = data["prepped_image"]
        file = data["file_name"]
        label = image_item["label"]
        label_idx = aspects.index(label)

        boundaries = data["boundaries"]
        top = boundaries[label_idx]
        bottom = boundaries[label_idx + 1]
        aspect_image = image[top:bottom, :, :]

        title = f"{file} - {label}"

        hues = get_hues(aspect_image)
        # print("\n", file)
        # print(sorted((hues)))
        # figs = []
        # figs.append(h.add_figure(aspect_image, "aspect"))
        # figs.append(h.add_figure(masked_by_hues(aspect_image), "masked"))
        # h.plot_figures(figs, title)

        for hue in hues:
            color_maps[label_idx][hue] = color_maps[label_idx].get(hue, 0) + 1
            #d[c] = d.get(c, 0) + 1

    # used colors
    print(f"Image Count: {len(IMAGE_LIST)}")
    for i, aspect in enumerate(aspects):
        percent_to_use = 0.25

        # sort hues by values (# of occurrences)
        prioritized_hues = sorted(color_maps[i].items(), key=lambda x: x[1], reverse=True)

        # identify a percentage of the most frequently used hues
        last_hue = int(len(prioritized_hues) * percent_to_use)
        #filtered_hues = sorted([h[0] for h in prioritized_hues[0:last_hue]])
        filtered_hues = [h[0] for h in prioritized_hues[0:last_hue]]
        sorted_hues = sorted(filtered_hues)

        print(f"\n{aspect} ({len(filtered_hues)})")
        print(f"frequncy: {filtered_hues}")
        print(f"sorted: {sorted_hues}")


### extract brightness
def extract_brightness(IMAGE_LIST):
    aspects = ["red", "yellow", "green"]
    performance = np.full((3, 3), 0)

    for IMAGE_ITEM in IMAGE_LIST:
        data = c.pre_process(IMAGE_ITEM)
        file = data["file_name"]
        label = data["label"]
        actual_aspect_idx = aspects.index(label)
        prepped_image = data["prepped_image"]
        edged_image = data["edged_image"]
        boundaries = data["boundaries"]
        scores = [0, 0, 0]

        for aspect_idx, aspect in enumerate(aspects):
            top = boundaries[aspect_idx]
            bottom = boundaries[aspect_idx + 1]
            aspect_image = prepped_image[top:bottom, :, :]

            hsv = cv2.cvtColor(aspect_image, cv2.COLOR_BGR2HSV)
            hsv_v = hsv[:, :, 2]

            for row in hsv_v:
                scores[aspect_idx] += sum(row)

        calc_aspect_idx = np.argmax(scores)

        performance[actual_aspect_idx][calc_aspect_idx] += 1


        #if label_idx != calc_idx:
        figs = []
        #figs.append(h.add_figure(source_image, "source"))
        #figs.append(h.add_figure(edged_image, "edged", cmap="gray", axhlines=boundaries))
        #figs.append(h.add_figure(prepped_image, "prepped", axhlines=boundaries))
        #figs.append(h.add_figure(mask, "mask", axhlines=axhlines, cmap="gray"))
        #figs.append(h.add_figure(bright_image, "bright_image", axhlines=boundaries))
        #h.plot_figures(figs, title, nrows=2, ncols=2)
        #h.plot_figures(figs, title)

    print("Brightness Performance")
    print(performance)


def random_images_plot(IMAGE_LIST):
    figs = []
    for i in range(20):
        file = IMAGE_LIST[i]["file_name"]
        label = IMAGE_LIST[i]["label"]
        image = IMAGE_LIST[i]["image"]
        title = f"{file} ({label})"
        figs.append(h.add_figure(image, title))
    h.plot_figures(figs, "Random Samples From Traffic Light Image Data Set", 5, 4)
