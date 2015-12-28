import logging, itertools
import numpy as np
import theano, theano.tensor as T
import blocks, blocks.bricks, blocks.bricks.base, blocks.bricks.recurrent, blocks.bricks.conv, blocks.initialization, blocks.graph, blocks.extensions, blocks.extensions.saveload, blocks.extensions.monitoring, blocks.model, blocks.main_loop, blocks.algorithms
import dataset, bricks, extensions, initialization

logging.basicConfig()
logger = logging.getLogger(__name__)

def swap_tb(*xs):
    return [x.dimshuffle(1, 0, *range(2, x.ndim)) for x in xs]

def main():
    # shape (batch, time, pitch)
    xs_down = T.tensor3("features")
    # shape (batch, time)
    mask = T.matrix("features_mask")

    theano.config.compute_test_value = "warn"
    test_batch = next(dataset.get_stream("train", max_examples=11).get_epoch_iterator(as_dict=True))
    xs_down.tag.test_value = test_batch["features"][:11]
    mask.tag.test_value = test_batch["features_mask"][:11]

    # mask doesn't have a pitch axis; make it broadcast
    mask = T.shape_padright(mask)

    # truncate sequence for debugging
    xs_down = xs_down[:, :30]
    mask = mask[:, :30]

    pitch_dim = 128

    def make_convnet(name, dims):
        dims = list(dims)
        filter_size = (9, 9)
        return blocks.bricks.conv.ConvolutionalSequence(
            name=name,
            layers=list(itertools.chain.from_iterable(
                # theano conv2d for now takes only border modes "full" or
                # "valid"; we use full and then remove the excess padding with
                # the Unpad brick. the result is like "same" convolution.
                (blocks.bricks.conv.ConvolutionalActivation(
                    name="conv_%i" % i,
                    activation=blocks.bricks.Rectifier().apply,
                    filter_size=filter_size,
                    num_filters=dim,
                    border_mode="full"),
                 bricks.Unpad(
                    name="unpad_%i" % i,
                    filter_size=filter_size,
                    num_channels=dim))
                for i, dim in enumerate(dims[1:]))),
            num_channels=dims[0],
            image_size=(None, None),
            tied_biases=True,
            weights_init=initialization.ConvolutionalInitialization(
                blocks.initialization.Orthogonal()),
            biases_init=initialization.Constant(0))

    # one convnet to 2d convolve the piano rolls, another to be its inverse
    convnet_dims = [1, 16]
    lower_dim = convnet_dims[0] * pitch_dim
    upper_dim = convnet_dims[-1] * pitch_dim
    convnet_up = make_convnet("up", convnet_dims)
    convnet_down = make_convnet("down", reversed(convnet_dims))

    convnet_up.initialize()
    convnet_down.initialize()

    def convapply(xs, indim, outdim, convnet):
        # reinstitute channel axis
        xs = xs.reshape((xs.shape[0], xs.shape[1], indim, pitch_dim)).dimshuffle(0, 2, 1, 3)
        xs = convnet.apply(xs)
        # move channel axis after time and lump it in with the pitch axis
        xs = xs.dimshuffle(0, 2, 1, 3).reshape((xs.shape[0], xs.shape[2], outdim * pitch_dim))
        return xs

    def convup(xs):
        return convapply(xs, convnet_dims[0], convnet_dims[-1], convnet_up)

    def convdown(xs):
        return convapply(xs, convnet_dims[-1], convnet_dims[0], convnet_down)

    intermediate_dim = 256
    recurrent_dim = 256

    x_to_h = blocks.bricks.MLP(
        name="x_to_h",
        dims=[upper_dim, intermediate_dim, 4*recurrent_dim],
        activations=[blocks.bricks.Rectifier(), blocks.bricks.Identity()],
        weights_init=blocks.initialization.Orthogonal(),
        biases_init=blocks.initialization.Constant(0))
    lstm = blocks.bricks.recurrent.LSTM(
        dim=recurrent_dim,
        weights_init=initialization.GlorotInitialization(),
        biases_init=blocks.initialization.Constant(0))
    h_to_y = blocks.bricks.MLP(
        name="h_to_y",
        dims=[recurrent_dim, intermediate_dim, upper_dim],
        activations=[blocks.bricks.Rectifier(), blocks.bricks.Rectifier()],
        weights_init=blocks.initialization.Orthogonal(),
        biases_init=blocks.initialization.Constant(0))

    x_to_h.initialize()
    lstm.initialize()
    h_to_y.initialize()

    initialization.lstm_identity_initialize(lstm)
    initialization.lstm_bias_initialize(lstm, x_to_h.linear_transformations[-1].b)

    def stepfn(x, h, c):
        u = x_to_h.apply(x)
        h, c = lstm.apply(
            inputs=u, states=h, cells=c,
            iterate=False)
        y = h_to_y.apply(h)
        return y, h, c

    def predict(xs):
        [xs] = swap_tb(xs)
        [ys, hs, cs], _ = theano.scan(
            stepfn,
            sequences=[xs],
            outputs_info=[None] + lstm.initial_states(xs.shape[1]))
        [ys, hs, cs] = swap_tb(ys, hs, cs)
        return ys, hs, cs

    def generate(xs):
        [xs] = swap_tb(xs)
        # initialize hidden state based on xs
        [ys, hs, cs], _ = theano.scan(
            stepfn,
            sequences=[xs],
            outputs_info=[None] + lstm.initial_states(xs.shape[1]))
        # let the model extrapolate based on its own predictions
        [ys, _, _], _ = theano.scan(
            stepfn,
            n_steps=128,
            outputs_info=[ys[-1], hs[-1], cs[-1]])
        [ys] = swap_tb(ys)
        return ys

    xs_up = convup(xs_down)

    # never backprop through targets
    ys_down = theano.gradient.disconnected_grad(xs_down)
    ys_up = theano.gradient.disconnected_grad(xs_up)

    yhats_up, hs, cs = predict(xs_up)
    prediction_errors = (yhats_up[:, :-1] - ys_up[:, 1:])**2 * mask[:, 1:]
    prediction_cost = (prediction_errors.sum(axis=1) / mask.sum(axis=1)).mean()
    prediction_cost.name = "prediction_cost"

    # train convdown to reconstruct training examples while keeping convup fixed
    reconstruction_cost = ((convdown(ys_up) - ys_down)**2).mean()
    reconstruction_cost.name = "reconstruction_cost"

    cost = prediction_cost + reconstruction_cost
    cost.name = "cost"

    graph = blocks.graph.ComputationGraph(cost)
    model = blocks.model.Model(cost)
    algorithm = blocks.algorithms.GradientDescent(
        cost=cost,
        parameters=graph.parameters,
        step_rule=blocks.algorithms.Adam())

    step_channels = []
    for key, parameter in model.get_parameter_dict().items():
        step_channels.extend([algorithm.steps[parameter].norm(2)
                              .copy(name="step_norm:%s" % key),
                              algorithm.gradients[parameter].norm(2)
                              .copy(name="gradient_norm:%s" % key)])
    step_channels.extend([algorithm.total_step_norm.copy(name="total_step_norm"),
                          algorithm.total_gradient_norm.copy(name="total_gradient_norm")])

    activations = [
        hs.mean().copy(name="states.mean"),
        cs.mean().copy(name="cells.mean")]

    monitors = []
    monitors.append(blocks.extensions.monitoring.TrainingDataMonitoring(
        step_channels,
        prefix="iteration"))
    monitors.extend(
        blocks.extensions.monitoring.DataStreamMonitoring(
            graph.outputs + activations,
            data_stream=dataset.get_stream(which_set, max_examples=100),
            prefix=which_set,
            after_epoch=True)
        for which_set in "train test".split())

    main_loop = blocks.main_loop.MainLoop(
        data_stream=dataset.get_stream("train"),
        model=model, algorithm=algorithm,
        extensions=(monitors + [
            blocks.extensions.FinishAfter(after_n_epochs=100),
            blocks.extensions.ProgressBar(),
            blocks.extensions.Printing(),
            extensions.Generate(
                path="samples_{epoch}.npz",
                generate_fn=theano.function([xs_down], convdown(generate(convup(xs_down)))),
                pitch_dim=pitch_dim,
                every_n_epochs=1)]))
            #blocks.extensions.saveload.Checkpoint(
            #    path="checkpoint.pkl",
            #    after_epoch=True,
            #    on_interrupt=True)]))
    main_loop.run()

if __name__ == "__main__":
    main()
