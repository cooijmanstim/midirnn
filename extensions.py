import numpy as np
import blocks.extensions

class Generate(blocks.extensions.SimpleExtension):
    def __init__(self, path, generate_fn, pitch_dim, **kwargs):
        super(Generate, self).__init__(**kwargs)
        self.path = path
        self.generate_fn = generate_fn
        self.pitch_dim = pitch_dim

    def do(self, which_callback, *args):
        # generate 10 examples based on two random 4-note chords
        xs = np.random.multinomial(4, [1./self.pitch_dim]*self.pitch_dim,
                                   size=(10, 2))
        # stretch the chords out a bit over time
        # (not needed for JSB piano rolls)
        #xs = np.repeat(xs, 10, axis=1)
        ys = self.generate_fn(xs)
        path = self.path.format(epoch=self.main_loop.status["epochs_done"])
        np.savez(path, xs=np.concatenate([xs, ys], axis=1))
