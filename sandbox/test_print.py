import time
import sys
from machine_learning_control.control.utils.log_utils import colorize

cnt = 0
print("start info")
print("")
while True:
    time.sleep(1)
    cnt += 1
    # sys.stdout.write(
    #     "\r{} {:8.3G}, {} {:8.3g}, {} {:8.3G}".format(
    #         "Epoch", float(cnt), "Steps", float(cnt), "Time", float(cnt),
    #     )
    # )
    # sys.stdout.flush()
    print(
        colorize(
            "\r{}: {:8.3G}, {}: {:8.3g}, {}: {:8.3G}".format(
                "Epoch",
                float(cnt),
                "Step",
                float(cnt),
                "Time",
                float(cnt),
            ),
            "green",
        ),
        end="",
    )
