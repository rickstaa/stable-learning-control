"""Module that initializes the stable_learning_control package."""
# Make module version available.
from .version import __version__  # noqa: F401
from .version import __version_tuple__  # noqa: F401

# Put algorithms in main namespace.
from stable_learning_control.algos.pytorch.lac.lac import lac as lac_pytorch
from stable_learning_control.algos.pytorch.sac.sac import sac as sac_pytorch
from stable_learning_control.utils.import_utils import import_tf

if import_tf(dry_run=True, frail=False):
    from stable_learning_control.algos.tf2.lac.lac import lac as lac_tf2
    from stable_learning_control.algos.tf2.sac.sac import sac as sac_tf2
