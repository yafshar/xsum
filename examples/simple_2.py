from xsum import *
import numpy as np

# A small superaccumulator
sacc = xsum_small()

a = np.arange(0, 1, 0.1)

# Adding a vector of numbers
sacc.add(a)

print("sum       = {:.20f}".format(np.sum(a)))
print("Exact sum = {:.20f}".format(sacc.round()))