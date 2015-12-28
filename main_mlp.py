import logging
import numpy as np
import theano, theano.tensor as T
import blocks, blocks.bricks, blocks.bricks.recurrent, blocks.initialization, blocks.graph, blocks.extensions, blocks.extensions.saveload, blocks.extensions.monitoring, blocks.model, blocks.main_loop, blocks.algorithms
import dataset
import extensions

logging.basicConfig()
logger = logging.getLogger(__name__)


def main():
    # shape (batch, time, pitch)
    xs = T.tensor3("features")
    # shape (batch, time)
    mask = T.matrix("features_mask")

    theano.config.compute_test_value = "warn"
    test_batch = next(dataset.get_stream("train", max_examples=11,
                                         piano_roll=True).get_epoch_iterator(as_dict=True))
    xs.tag.test_value = test_batch["features"][:11]
    mask.tag.test_value = test_batch["features_mask"][:11]

    # mask doesn't have a pitch axis; make it broadcast
    mask = T.shape_padright(mask)

    # move time axis in front of batch axis
    xs = xs.dimshuffle(1, 0, 2)
    mask = mask.dimshuffle(1, 0, 2)

    input_dim = 128
    intermediate_dim = 32

    x_to_y = blocks.bricks.MLP(
        name="x_to_y",
        dims=[input_dim, intermediate_dim, input_dim],
        activations=[blocks.bricks.Rectifier(), blocks.bricks.Rectifier()],
        weights_init=blocks.initialization.Orthogonal(),
        biases_init=blocks.initialization.Constant(0))

    x_to_y.initialize()

    def stepfn(x):
        y = x_to_y.apply(x)
        return y

    def predict(xs):
        ys, _ = theano.scan(
            stepfn,
            sequences=[xs],
            outputs_info=[None])
        return ys

    def generate(xs):
        # initialize hidden state based on xs
        ys, _ = theano.scan(
            stepfn,
            sequences=[xs],
            outputs_info=[None])
        # let the model extrapolate based on its own predictions
        ys, _ = theano.scan(
            stepfn,
            n_steps=128,
            outputs_info=[ys[-1]])
        return ys

    ys = predict(xs)
    errors = (ys[:-1] - xs[1:])**2 * mask[1:]
    cost = (errors.sum(axis=0) / mask.sum(axis=0)).mean()
    cost.name = "cost"

    graph = blocks.graph.ComputationGraph(cost)
    model = blocks.model.Model(cost)
    algorithm = blocks.algorithms.GradientDescent(
        cost=cost,
        parameters=graph.parameters,
        step_rule=blocks.algorithms.Adam())

    step_channels = []
    for key, parameter in model.get_parameter_dict().items():
        step_channels.append(algorithm.steps[parameter].norm(2)
                             .copy(name="step_norm:%s" % key))

    monitors = [
        blocks.extensions.monitoring.DataStreamMonitoring(
            graph.outputs + (step_channels if which_set == "train" else []),
            data_stream=dataset.get_stream(which_set, max_examples=100),
            prefix=which_set,
            after_epoch=True)
        for which_set in "train test".split()]
    main_loop = blocks.main_loop.MainLoop(
        data_stream=dataset.get_stream("train"),
        model=model, algorithm=algorithm,
        extensions=(monitors + [
            blocks.extensions.FinishAfter(after_n_epochs=100),
            blocks.extensions.ProgressBar(),
            blocks.extensions.Printing(),
            extensions.Generate(
                path="samples_{epoch}.npz",
                generate_fn=theano.function([xs], generate(xs)),
                input_dim=input_dim,
                every_n_epochs=1),
            blocks.extensions.saveload.Checkpoint(
                path="checkpoint.pkl",
                after_epoch=True,
                on_interrupt=True)]))
    main_loop.run()

if __name__ == "__main__":
    main()
