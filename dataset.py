import os, glob
import logging
import fuel.datasets, fuel.streams, fuel.schemes, fuel.transformers
from picklable_itertools import iter_, chain
import pretty_midi

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
                x = midi_data.get_piano_roll(fs=10)
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

def get_dataset(which_set):
    #datadir = os.environ["UNIQUE_MID_DIR"]
    datadir = os.environ["JSBCHORALES_DIR"]
    files = glob.glob(os.path.join(datadir, which_set, "*.mid"))
    return MidiFiles(files)

def get_stream(which_set, shuffle=True, max_examples=None, batch_size=10):
    dataset = get_dataset(which_set)
    num_examples = min(max_examples, dataset.num_examples)
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
