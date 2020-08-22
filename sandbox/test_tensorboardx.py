import sys

from torch.utils.tensorboard import SummaryWriter

try:
    import tensorflow

    # Store whether tensorflow is available
    TF_AVAILABLE = True
except ImportError:
    # Store whether tensorflow is available
    TF_AVAILABLE = False

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    return "\x1b[%sm%s\x1b[0m" % (";".join(attr), string)


tb_writer = SummaryWriter()
if TF_AVAILABLE:
    print("jan")
else:
    print(
        colorize(
            "Warning: Logger.restore_tf_graph method can not be used as tensorflow is "
            "not installed in the current environment. Please install tensorflow if "
            "you want to use tensorflow related logging methods.",
            color="yellow",
            bold=True,
        )
    )
