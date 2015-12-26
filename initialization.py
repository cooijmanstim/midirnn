import itertools
import numpy
import theano
from blocks.initialization import *

# Initialize convolutional filters by generating an mxn weight matrix
# at each spatial location, with m being the incoming number of channels
# and n the outgoing number of channels.  Allows e.g. Orthogonal
# initialization for convnets.
class ConvolutionalInitialization(NdarrayInitialization):
    def __init__(self, initialization):
        self.initialization = initialization

    def generate(self, rng, shape):
        x = numpy.zeros(shape, dtype=theano.config.floatX)
        for i in itertools.product(*map(range, shape[2:])):
            x[numpy.index_exp[:, :] + i] = self.initialization.generate(rng, shape[:2])
        # divide by spatial fan-in
        x /= numpy.prod(shape[2:])
        return x
