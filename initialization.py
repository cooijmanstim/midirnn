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

def lstm_identity_initialize(lstm):
    # identity initialization for LSTM
    identity = Identity()
    W = lstm.W_state.get_value()
    n = lstm.get_dim("states")
    W[:, 2*n:3*n] = 0.95 * identity.generate(lstm.rng, (W.shape[0], n))
    lstm.W_state.set_value(W)

def lstm_bias_initialize(lstm, bias):
    b = bias.get_value()
    n = lstm.get_dim("states")
    b[1*n:2*n] = 1.
    bias.set_value(b)

class GlorotInitialization(NdarrayInitialization):
    def generate(self, rng, shape):
        # makes sense for matrices and for vectors representing diagonal matrices
        assert(len(shape) in [1, 2])
        d = numpy.sqrt(6. / sum(shape))
        m = rng.uniform(-d, +d, size=shape)
        return m.astype(theano.config.floatX)
