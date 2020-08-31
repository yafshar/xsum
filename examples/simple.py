from xsum import *
import numpy as np

# A small superaccumulator
sacc = xsum_small_accumulator()

a = np.arange(0, 1, 0.1)

# Adding a vector of numbers
xsum_add(sacc, a)

print("sum       = {:.20f}".format(np.sum(a)))
print("Exact sum = {:.20f}".format(xsum_round(sacc)))