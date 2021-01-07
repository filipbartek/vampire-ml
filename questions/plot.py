import io

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


def scatter(x, y, name=None, title=None, xlabel=None, ylabel=None, xscale=None, yscale=None, filename=None, step=None):
    fig = plt.figure()
    sns.scatterplot(x=x, y=y)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xscale is not None:
        plt.xscale(xscale)
    if yscale is not None:
        plt.yscale(yscale)
    if name is not None:
        tf.summary.image(name, plot_to_image(fig), step=step)
    if filename is not None:
        plt.savefig(filename)
    plt.close()


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # https://www.tensorflow.org/tensorboard/image_summaries
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
