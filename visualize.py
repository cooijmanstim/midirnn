import os
import glob
import logging
import numpy as np
import pylab as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_matrices(xs):
    for x in xs:
        plt.matshow(x.T, cmap='gray')
    plt.show()


def load_from_npz(fname):
    samples = np.load(fname)
    xs = samples["xs"]
    return xs


def retrieve_most_recent_fname():
    fname = max(glob.iglob('*.npz'), key=os.path.getctime)
    logger.info("The most recent file is %s." % fname)
    return fname


def plot_sample(fname):
    xs = load_from_npz(fname)
    plot_matrices(xs)


def plot_most_recent_sample():
    fname = retrieve_most_recent_fname()
    xs = load_from_npz(fname)
    plot_matrices(xs)


if __name__ == "__main__":
    plot_most_recent_sample()
