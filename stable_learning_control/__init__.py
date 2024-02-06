"""Module that Initialises the stable_learning_control package."""

# Put algorithms in main namespace.
from stable_learning_control.algos.pytorch.lac.lac import lac as lac_pytorch
from stable_learning_control.algos.pytorch.latc.latc import latc as latc_pytorch
from stable_learning_control.algos.pytorch.sac.sac import sac as sac_pytorch
from stable_learning_control.utils.import_utils import tf_installed

# Make module version available.
from .version import __version__, __version_tuple__

if tf_installed():
    from stable_learning_control.algos.tf2.lac.lac import lac as lac_tf2
    from stable_learning_control.algos.tf2.latc.latc import latc as latc_tf2
    from stable_learning_control.algos.tf2.sac.sac import sac as sac_tf2
