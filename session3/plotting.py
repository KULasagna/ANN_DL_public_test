import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import colorsys


def grid_plot(images, shape=None, cmap="gray"):
    """Plot the given images in a grid.

    Parameters
    ----------
    images : array_like
        List of images to show in the grid. Images should have
        the correct shape, unless the `shape` parameter is
        explicitly set.
    shape : tuple
        Shape of each individual image. If ``None`` the original
        shape is kept, otherwise all images are reshaped to the
        given shape.
    cmap : colormap
        Colormap to use.
    
    Returns
    -------
    R, C : int
        Number of rows and columns in the grid.

    """
    N = len(images)
    C = int(np.ceil(np.sqrt(N)))
    R = int(np.ceil(N/C))

    plt.figure(figsize=(5, 5*R/C))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for i, img in enumerate(images):
        if shape is not None:
            img = np.reshape(img, shape)
        plt.subplot(R, C, i+1)
        plt.imshow(img, cmap=cmap)
        plt.axis('off')

    return R, C


def _adjust_lightness(rgba, lightness):
    r, g, b, a = rgba
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return (*colorsys.hls_to_rgb(h, lightness, s), a)


def confusion_plot(y_true, y_pred):
    """Create and show the confusion plot for the given predictions and ground truths.

    Parameters
    ----------
    y_true : array_like
        The ground-truth class labels. Both the 'class index' (shape Dx1) and 'one-hot'
        encodings (shape DxC) are supported, where D is the size of the dataset and C
        is the number of classes.
    y_pred : array_like
        The predicted class labels by a trained model. Both the 'class index' (shape
        Dx1) and 'one-hot'/'probability' encodings (shape DxC) are supported, where D
        is the size of the dataset and C is the number of classes.

    """
    # Convert one-hot encoding to class labels
    if len(y_true.shape) == 2:
        y_true = tf.argmax(y_true, axis=1)
    if len(y_pred.shape) == 2:
        y_pred = tf.argmax(y_pred, axis=1)
    # Calculate confusion matrix
    cm = tf.math.confusion_matrix(y_true, y_pred)
    N = tf.reduce_sum(cm)
    C = cm.shape[0]
    # Calculate marginal and overall accuracy
    d = np.diag(cm)
    mp = d / np.sum(cm, axis=0)
    mt = d / np.sum(cm, axis=1)
    o = np.sum(d) / N

    # Create plot (based on https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html)
    f, ax = plt.subplots(figsize=(C, C))
    # Plot the heatmap
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "red_yellow_green",  # Custom color map between red, yellow and green
        [(0.98, 0.77, 0.75), (0.98, 0.96, 0.71), (0.74, 0.90, 0.77)]
    )
    im = ax.imshow(cm, cmap=cmap)
    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(C+1))
    ax.set_xticklabels(np.arange(C))
    ax.set_yticks(np.arange(C+1))
    ax.set_yticklabels(np.arange(C))
    ax.minorticks_off()
    ax.set_xlabel("Predicted class", fontsize=18)
    ax.set_ylabel("True class", fontsize=18)
    ax.set_title("Confusion matrix", fontsize=24)
    # Turn spines off and create white grid.
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(C+1)-0.5, minor=True)
    ax.set_yticks(np.arange(C+1)-0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="both", bottom=False, left=False)
    # Create black lines to separate marginals
    ax.plot([-0.5, C+0.5], [C-0.5, C-0.5], color="k", linewidth=2)
    ax.plot([C-0.5, C-0.5], [-0.5, C+0.5], color="k", linewidth=2)
    # Loop over the data and create a `Text` for each "pixel".
    for i in range(C):
        for j in range(C):
            val = f"$\mathbf{{{cm[i, j]}}}$\n${100 * cm[i, j] / N:.1f}\%$"
            im.axes.text(j, i, val, ha="center", va="center")
    # Create a `Text` for every marginal accuracy
    for i in range(C):
        val = f"${100 * mp[i]:.1f}\%$"
        ax.text(i, C, val, color=_adjust_lightness(cmap(mp[i]), 0.3), ha="center", va="center")
        val = f"${100 * mt[i]:.1f}\%$"
        ax.text(C, i, val, color=_adjust_lightness(cmap(mt[i]), 0.3), ha="center", va="center")
    # Create a `Text` for the overall accuracy
    val = f"$\mathbf{{{100 * o:.1f}}}\%$"
    ax.text(C, C, val, color=_adjust_lightness(cmap(o), 0.3), ha="center", va="center")

    # Show the plot
    plt.show()
