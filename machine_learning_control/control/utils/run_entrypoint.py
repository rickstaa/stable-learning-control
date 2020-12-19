"""Module used for determining the run entrypoint.

This module was cloned from the
`spinningup repository <https://github.com/openai/spinningup/blob/master/spinup/utils/run_utils.py>`_.
"""

import base64
import pickle
import zlib

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("encoded_thunk")
    args = parser.parse_args()
    thunk = pickle.loads(zlib.decompress(base64.b64decode(args.encoded_thunk)))
    thunk()