import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass
