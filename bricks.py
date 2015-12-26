import numpy as np
from blocks.bricks import Activation, Feedforward
from blocks.bricks.base import lazy, application
import util

class Unpad(Feedforward):
    @lazy(allocation=['filter_size'])
    def __init__(self, filter_size, num_channels, **kwargs):
        super(Unpad, self).__init__(**kwargs)
        self.filter_size = filter_size
        self.num_filters = num_channels

    @application(inputs=["input_"], outputs=["output"])
    def apply(self, input_):
        ks = tuple(int((k - 1) / 2) for k in self.filter_size)
        return input_[np.index_exp[:, :]
                      + tuple(slice(k, -k) for k in ks)]

    def get_dim(self, name):
        # blocks conv get_dim contract is an absolute mess
        return (None, None, None)
