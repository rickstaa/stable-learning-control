"""Module that can be used for determining the run entrypoint.

.. note::
    This module was based on
    `Spinning Up repository <https://github.com/openai/spinningup/tree/master/spinup/utils/run_utils.py>`__.

Source code
-----------
.. literalinclude:: /../../stable_learning_control/utils/run_entrypoint.py
   :language: python
   :linenos:
   :lines: 14-
"""  # noqa
import base64
import pickle
import zlib

# NOTE: This is needed because of https://github.com/pytorch/pytorch/issues/99637
import torch  # noqa: F401

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("encoded_thunk")
    args = parser.parse_args()
    thunk = pickle.loads(zlib.decompress(base64.b64decode(args.encoded_thunk)))
    thunk()
