"""Contains the BLC Reinforcement Learning (RL) and Imitation learning (IL) algorithms.
"""

from bayesian_learning_control.control.algos.pytorch.lac.lac import lac as lac_pytorch
from bayesian_learning_control.control.algos.pytorch.lac2.lac2 import (
    lac2 as lac2_pytorch,
)
from bayesian_learning_control.control.algos.pytorch.lac3.lac3 import (
    lac3 as lac3_pytorch,
)
from bayesian_learning_control.control.algos.pytorch.lac4.lac4 import (
    lac4 as lac4_pytorch,
)
from bayesian_learning_control.control.algos.pytorch.lac5.lac5 import (
    lac5 as lac5_pytorch,
)
from bayesian_learning_control.control.algos.pytorch.lac6.lac6 import (
    lac6 as lac6_pytorch,
)
from bayesian_learning_control.control.algos.pytorch.sac.sac import sac as sac_pytorch
from bayesian_learning_control.control.algos.pytorch.sac2.sac2 import (
    sac2 as sac2_pytorch,
)

from bayesian_learning_control.utils.import_utils import import_tf
from bayesian_learning_control.version import __version__

if import_tf(dry_run=True, frail=False):
    from bayesian_learning_control.control.algos.tf2.lac.lac import lac as lac_tf2
    from bayesian_learning_control.control.algos.tf2.sac.sac import sac as sac_tf2
