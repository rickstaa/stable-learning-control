import numpy as np

from is_scalar import is_scalar

print(is_scalar(1), "True")
print(is_scalar([1]), "False")
print(is_scalar(np.array(2)), "True")  # I'm here
pass
