import numpy as np
import matplotlib.pyplot as plt


def plot_image(image, title=None, cmap=None, axhlines=None):
    if title:
        plt.title(title)

    if axhlines:
        for l in axhlines:
            plt.axhline(y=l, color="blue", linestyle="dotted")

    if not cmap:
        cmap = "viridis"

    plt.imshow(image, cmap=cmap)
    plt.show()


def add_figure(image, title=None, axhlines=None, cmap=None):
    return {"image":image, "title":title, "axhlines":axhlines, "cmap":cmap}

def plot_figures(figures, plot_title=None, nrows=None, ncols=None):
    if not nrows or not ncols:
        # if grid not specified, plot figures in a single row
        nrows = 1
        ncols = len(figures)
    else:
        # check minimum grid configured
        if len(figures) > nrows * ncols:
            raise ValueError(f"Too few subplots ({nrows*ncols}) specified for ({len(figures)}) figures.")

    # close any open figures
    plt.close("all")

    fig = plt.figure(plot_title)

    # spacing between figures
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    # plot figures
    for idx, fig in enumerate(figures):
        plt.subplot(nrows, ncols, idx + 1)

        # cmap
        cmap = fig["cmap"]
        if not cmap:
            cmap = "viridis"

        # axhlines
        axhlines = fig["axhlines"]
        if axhlines:
            for l in axhlines:
                plt.axhline(y=l, color="blue", linestyle="dotted")

        plt.title(fig["title"])
        plt.imshow(fig["image"], cmap=cmap)
    plt.show()


def main():
    number_of_images = 4
    images = [np.random.randn(100, 100) for i in range(number_of_images)]
    figures = []
    for idx, image in enumerate(images):
        figures.append(add_figure(images[0], title=f"Image: {idx}"))
    plot_figures(figures, "random figures", 2, 2)

    #plot_image(images[0], "Random Image")

if __name__ == "__main__":
    main()