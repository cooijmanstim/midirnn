import os, glob
import logging
import numpy as np
import fuel.datasets, fuel.streams, fuel.schemes, fuel.transformers
from picklable_itertools import iter_, chain
import pretty_midi

try:
    import cPickle as pickle
except ImportError:
    import six.moves.cPickle as pickle

logger = logging.getLogger(__name__)

class MidiFiles(fuel.datasets.Dataset):
    provides_sources = ('features',)
    example_iteration_scheme = None

    def __init__(self, files):
        self.files = list(files)
        super(MidiFiles, self).__init__()

    def open(self):
        return iter_(self.files)

    def get_data(self, state=None, request=None):
        if request is not None:
            raise ValueError

        for file in state:
            try:
                midi_data = pretty_midi.PrettyMIDI(file)

                # TODO: maybe transpose to canonical key
                # get everything except drums
                x = midi_data.get_piano_roll(fs=1)
                # TODO: create a separate roll with onsets
                # TODO: include drums somehow (e.g. one pitchless channel with onsets only)

                # time axis first
                x = x.T
                return (x,)
            except Exception as e:
                logger.warning("skipping %s: %s" % (os.path.basename(file), e))

        return None

    @property
    def num_examples(self):
        return len(self.files)


class PianoRolls(fuel.datasets.Dataset):
    provides_sources = ('features',)
    example_iteration_scheme = None

    def __init__(self, piano_rolls):
        self.piano_rolls = piano_rolls
        super(PianoRolls, self).__init__()
        self.piano_roll_dim = 128

    def open(self):
        return iter_(self.piano_rolls)

    def get_data(self, state=None, request=None):
        if state is None:
            raise ValueError

        if request is not None:
            raise ValueError

        piano_roll = next(state)
        if piano_roll is None:
            return None

        # shape (time, pitch)
        x = np.zeros((len(piano_roll), self.piano_roll_dim), dtype=np.int8)
        for i, vertical_slice in enumerate(piano_roll):
            # set the on pitches to 1
            x[i, vertical_slice] = 1
        return (x,)

    @property
    def num_examples(self):
        return len(self.piano_rolls)


def get_dataset(which_set):
    #datadir = os.environ["UNIQUE_MID_DIR"]
    datadir = os.environ["JSBCHORALES_DIR"]
    files = glob.glob(os.path.join(datadir, which_set, "*.mid"))
    return MidiFiles(files)


def get_dataset_pianoroll(which_set):
    datadir = os.environ["JSBCHORALES_PIANOROLL_DIR"]
    fname = 'JSB Chorales.pickle'
    pickle_fpath = os.path.join(datadir, fname)
    with open(pickle_fpath, 'rb') as p:
        data = pickle.load(p)
    return PianoRolls(data[which_set])


def get_stream(which_set, shuffle=True, max_examples=None, batch_size=10,
               piano_roll=True):
    if piano_roll:
        dataset = get_dataset_pianoroll(which_set)
    else:
        dataset = get_dataset(which_set)
    num_examples = dataset.num_examples
    if max_examples:
        num_examples = min(max_examples, num_examples)
    stream = fuel.streams.DataStream.default_stream(dataset=dataset)
    stream = fuel.transformers.ForceFloatX(stream)
    stream = fuel.transformers.Batch(
        stream,
        fuel.schemes.ConstantScheme(
            batch_size,
            dataset.num_examples))
    stream = fuel.transformers.Padding(stream)
    #stream = fuel.transformers.BackgroundProcess(stream, max_batches=3)
    return stream
