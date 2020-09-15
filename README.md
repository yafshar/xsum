# Fast Exact Summation Using Small and Large Superaccumulators (XSUM)

[![Build Status](https://travis-ci.com/yafshar/xsum.svg?token=aY1dW9PfH9SMySdB6Pzy&branch=master)](https://travis-ci.com/yafshar/xsum)
[![Python package](https://github.com/yafshar/xsum/workflows/Python%20package/badge.svg)](https://github.com/yafshar/xsum/actions)
[![Windows Build status](https://ci.appveyor.com/api/projects/status/kd1sksf5t3kdsibn/branch/master?svg=true)](https://ci.appveyor.com/project/yafshar/xsum/branch/master)
[![PyPI](https://img.shields.io/pypi/v/xsum.svg)](https://pypi.python.org/pypi/xsum)
[![Anaconda-Server Badge](https://img.shields.io/conda/vn/conda-forge/xsum.svg)](https://anaconda.org/conda-forge/xsum)
[![License](https://img.shields.io/badge/license-LGPL--v2-blue)](LICENSE)

In applications like optimization or finding the sample mean of data, it is
desirable to use higher accuracy than a simple summation. Where in a simple
summation, rounding happens after each addition.
An exact summation is a way to achieve higher accuracy results.

[XSUM](#neal_2015) is a library for performing exact summation using
super-accumulators. It provides methods for exactly summing a set of
floating-point numbers, where using a simple summation and the rounding which
happens after each addition could be an important factor in many applications.

This library is an easy to use header-only cross-platform C++11 implementation
and also contains Python bindings ([please see the example](#Python)).

The main algorithm is taken from the original C library
[FUNCTIONS FOR EXACT SUMMATION](https://gitlab.com/radfordneal/xsum) described
in the paper
["Fast Exact Summation Using Small and Large Superaccumulators,"](#neal_2015) by
[Radford M. Neal](https://www.cs.toronto.edu/~radford).

The code is rewritten in C++ and amended with more functionalities with the goal
of ease of use. The provided Python bindings provide the *exact summation*
interface in a Python code.

The C++ code also includes extra summation functionalities, (parallel reduction
on multi-core architectures) which are especially useful in high-performance
message passing libraries (like [OpenMPI](https://www.open-mpi.org/) and
[MPICH](https://www.mpich.org/)). Where binding a user-defined global summation
operation to an `op` handle can subsequently be used in `MPI_Reduce,`
`MPI_Allreduce,` `MPI_Reduce_scatter,` and `MPI_Scan` or a similar calls.

- **NOTE:** To see or use or reproduce the results of the original
  implementation reported in the paper
  `Fast Exact Summation Using Small and Large Superaccumulators`, by Radford M.
  Neal, please refer to
  [FUNCTIONS FOR EXACT SUMMATION](https://gitlab.com/radfordneal/xsum).

## Usage

**_XSUM_** library presents two objects, or super-accumulators
`xsum_small_accumulator` and `xsum_large_accumulator`. It also provides methods
for summing floating-point numbers and rounding to the nearest floating-point
number.

A small superaccumulator uses sixty-seven 64-bit chunks, each with 32-bit
overlap with the next one. This accumulator is the preferred method for summing
a moderate number of terms. A large superaccumulator uses 4096 64-bit chunks and
is suitable for big summations. A small superaccumulator is also a component of
the large superaccumulator [1].

**_XSUM_** library provides two interfaces to use superaccumulators. The first
one is a function interface, which takes input and produces output, and in the
second one, supperaccumulators are represented as classes (`xsum_small` and
`xsum_large`.)

### C++

`xsum_small_accumulator` and `xsum_large_accumulator`, both have a default
constructor, thus they do not need to be initialized. Addition operation is
simply a `xsum_add`,

```cpp
// A small superaccumulator
xsum_small_accumulator sacc;

// Adding values to the small accumulator sacc
xsum_add(&sacc, 1.0);
xsum_add(&sacc, 2.0);

// Large superaccumulator
xsum_large_accumulator lacc;

// Adding values to the large accumulator lacc
xsum_add(&lacc, 1.0e-3);
xsum_add(&lacc, 2.0e-3);
```

or `xsum_small`, and `xsum_large` objects can simply be used as,

```cpp
// A small superaccumulator
xsum_small sacc;

// Adding values to the small accumulator sacc
sacc.add(1.0);
sacc.add(2.0);

// Large superaccumulator
xsum_large lacc;

// Adding values to the large accumulator lacc
lacc.add(1.0e-3);
lacc.add(2.0e-3);
```

One can also add a vector of numbers to a superaccumulator.

```cpp
// A small superaccumulator
xsum_small_accumulator sacc;

// Adding a vector of numbers
double vec[] = {1.234e88, -93.3e-23, 994.33, 1334.3, 457.34, -1.234e88, 93.3e-23, -994.33, -1334.3, -457.34};

xsum_add(&sacc, vec, 10);

std::vector<double> v = {1.234e88, -93.3e-23, 994.33, 1334.3, 457.34, -1.234e88, 93.3e-23, -994.33, -1334.3, -457.34};

xsum_add(&sacc, v);
```

the same with `xsum_small`, and `xsum_large` objects as,

```cpp
// A small superaccumulator
xsum_small sacc;

// Adding a vector of numbers
double vec[] = {1.234e88, -93.3e-23, 994.33, 1334.3, 457.34, -1.234e88, 93.3e-23, -994.33, -1334.3, -457.34};

sacc.add(vec, 10);

std::vector<double> v = {1.234e88, -93.3e-23, 994.33, 1334.3, 457.34, -1.234e88, 93.3e-23, -994.33, -1334.3, -457.34};

sacc.add(v);
```

The squared norm of a vector (sum of squares of elements) and the dot product of
two vectors (sum of products of corresponding elements) can be added to a
superaccumulator using `xsum_add_sqnorm` and `xsum_add_dot` respectively.

For exmaple,

```cpp
// A small superaccumulator
xsum_small_accumulator sacc;

// Adding a vector of numbers
double vec1[] = {1.e-2, 2., 3.};
double vec2[] = {1.e-3, 2., 3.};

// Add dot product of vectors to a small superaccumulator
xsum_add_dot(&sacc, vec1, vec2, 3);
```

with `xsum_small`, and `xsum_large` objects as,

```cpp
// A small superaccumulator
xsum_small sacc;

// Adding a vector of numbers
double vec1[] = {1.e-2, 2., 3.};
double vec2[] = {1.e-3, 2., 3.};

// Add dot product of vectors to a small superaccumulator
sacc.add_dot(vec1, vec2, 3);
```

When it is needed, one can simply use the `xsum_init` to reinitilize the
superaccumulator.

```cpp
xsum_small_accumulator sacc;

xsum_add(&sacc, 1.0e-2);
...

// Reinitilize the small accumulator
xsum_init(&sacc);
...
```

or with `xsum_small` object as,

```cpp
xsum_small sacc;

sacc.add(1.0e-2);
...

// Reinitilize the small accumulator
sacc.init();
...
```

The superaccumulator can be rounded as,

```cpp
xsum_small_accumulator sacc;
xsum_add(&sacc, 1.0e-15);

....

double s = xsum_round(&sacc);
```

where, `xsum_round` is used to round the superaccumulator to the nearest
floating-point number.

With `xsum_small`, and `xsum_large` objects we do as,

```cpp
xsum_small sacc;
sacc.add(1.0e-15);

....

double s = sacc.round();
```

where, `round` is used to round the superaccumulator to the nearest
floating-point number.

Two **small** superaccumulators can be added together. `xsum_add` can be used to
add the second superaccumulator to the first one without doing any rounding. Two
**large** superaccumulator can also be added in the same way. In the case of the
two large superaccumulators, the second one internally will be rounded to a
small superaccumulator and then the addition is done.

```cpp
// Small superaccumulators
xsum_small_accumulator sacc1;
xsum_small_accumulator sacc2;

xsum_add(&sacc1, 1.0);
xsum_add(&sacc2, 2.0);

xsum_add(&sacc1, &sacc2);

// Large superaccumulators
xsum_large_accumulator lacc1;
xsum_large_accumulator lacc2;

xsum_add(&lacc1, 1.0);
xsum_add(&lacc2, 2.0);

xsum_add(&lacc1, &lacc2);
```

or as,

```cpp
// Small superaccumulators
xsum_small sacc1;
xsum_small sacc2;

sacc1.add(1.0);
sacc2.add(2.0);

// Add a small accumulator sacc2 to the first accumulator sacc1
sacc1.add(sacc2);

// Large superaccumulators
xsum_large lacc1;
xsum_large lacc2;

lacc1.add(1.0);
lacc2.add(2.0);

// Add a large accumulator lacc2 to the first accumulator lacc1
lacc1.add(lacc2);
```

A **small** superaccumulator can also be added to a **large** one,

```cpp
// Small superaccumulator
xsum_small_accumulator sacc;

xsum_add(&sacc, 1.0e-10);

...

// Large superaccumulator
xsum_large_accumulator lacc;

xsum_add(&lacc, 2.0e-3);

...

// Addition of a small superaccumulator to a large one
xsum_add(&lacc, &sacc);
```

With `xsum_small`, and `xsum_large` objects we do as,

```cpp
// Small superaccumulator
xsum_small sacc;

sacc.add(1.0e-10);

...

// Large superaccumulator
xsum_large lacc;

lacc.add(2.0e-3);

...

// Addition of a small superaccumulator to a large one
lacc.add(sacc);
```

The large superaccumulator can be rounded to a small one as,

```cpp
xsum_large_accumulator lacc;

xsum_small_accumulator sacc = xsum_round_to_small(&lacc);
```

or as,

```cpp
xsum_large lacc;

xsum_small_accumulator sacc = lacc.round_to_small();
```

### Example

Two simple examples on how to use the library:

```cpp
#include <iomanip>
#include <iostream>

#include "xsum/xsum.hpp"

using namespace xsum;

int main() {
  // Large superaccumulator
  xsum_large_accumulator lacc;
  double const a = 0.7209e-5;
  double s = 0;
  for (int i = 0; i < 10000; ++i) {
    xsum_add(&lacc, a);
    s += a;
  }
  std::cout << std::setprecision(20) << xsum_round(&lacc) << "\n"
            << std::setprecision(20) << s << "\n";
}
```

or a `xsum_small` or `xsum_large` objects can simply be used as,

```cpp
#include <iomanip>
#include <iostream>

#include "xsum/xsum.hpp"

using namespace xsum;

int main() {
  // Large superaccumulator
  xsum_large lacc;
  double const a = 0.7209e-5;
  double s = 0;
  for (int i = 0; i < 10000; ++i) {
    lacc.add(a);
    s += a;
  }
  std::cout << std::setprecision(20) << lacc.round() << "\n"
            << std::setprecision(20) << s << "\n";
}
```

One can compile the code as,

```bash
g++ simple.cpp -std=c++11 -O3 -o simple
```

or

```bash
icpc simple.cpp -std=c++11 -O3 -fp-model=double -o simple
```

running the `simple` would result,

```bash
./simple

0.072090000000000001301
0.072089999999998641278
```

### MPI reduction example (`MPI_Allreduce`)

To use a superaccumulator in high-performance message
passing libraries, first we need an MPI datatype of a superaccumulator, and then
we define a user-defined global operation `XSUM` that can subsequently be used
in `MPI_Reduce`, `MPI_Aallreduce`, `MPI_Reduce_scatter`, and `MPI_Scan`.

The below example is a simple demonstration of the use of a superaccumulator on
multiple processors, where the final summation across all processors is desired.

```cpp
#include <mpi.h>

#include <iomanip>
#include <iostream>

#include "xsum/myxsum.hpp"
#include "xsum/xsum.hpp"

using namespace xsum;

int main() {
  // Initialize the MPI environment
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Create the MPI data type of the superaccumulator
  MPI_Datatype acc_mpi = create_mpi_type<xsum_large_accumulator>();

  // Create the XSUM user-op
  MPI_Op XSUM = create_XSUM<xsum_large_accumulator>();

  double const a = 0.239e-3;
  double s(0);

  xsum_large_accumulator lacc;

  for (int i = 0; i < 1000; ++i) {
    s += a;
    xsum_add(&lacc, a);
  }

  MPI_Allreduce(MPI_IN_PLACE, &s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &lacc, 1, acc_mpi, XSUM, MPI_COMM_WORLD);

  if (world_rank == 0) {
    std::cout << "Rank =  " << world_rank
              << ", sum   =  " << std::setprecision(20) << a * 1000 * world_size
              << ", sum 1 =  " << std::setprecision(20) << s
              << ", sum 2 =  " << std::setprecision(20) << xsum_round(&lacc)
              << std::endl;
  }

  // Free the created user-op
  destroy_XSUM(XSUM);

  // Free the created MPI data type
  destroy_mpi_type(acc_mpi);

  // Finalize the MPI environment.
  MPI_Finalize();
}
```

```bash
mpic++ mpi_simple.cpp -std=c++11 -O3 -o simple
```

running the above code using 4 processors would result,

```bash
mpirun -np 4 ./simple

Rank =  0, sum   =  0.95600000000000007194, sum 1 =  0.95599999999998419575, sum 2 =  0.95600000000000007194
```

### Python

The provided Python bindings provide the *exact summation* interface in a
Python code.

### Python requirements

You need Python 3.6 or later to run `xsum`. You can have multiple Python
versions (2.x and 3.x) installed on the same system without problems.

To install Python 3 for different Linux flavors, macOS and Windows, packages
are available at\
[https://www.python.org/getit/](https://www.python.org/getit/)

### Using pip

[![PyPI](https://img.shields.io/pypi/v/xsum.svg)](https://pypi.python.org/pypi/xsum)

**pip** is the most popular tool for installing Python packages, and the one
included with modern versions of Python.

`xsum` can be installed with `pip`:

```sh
pip install xsum
```

**Note:**

Depending on your Python installation, you may need to use `pip3` instead of
`pip`.

```sh
pip3 install xsum
```

Depending on your configuration, you may have to run `pip` like this:

```sh
python3 -m pip install xsum
```

### Using pip (GIT Support)

`pip` currently supports cloning over `git`

```sh
pip install git+https://github.com/yafshar/xsum.git
```

For more information and examples, see the
[pip install](https://pip.pypa.io/en/stable/reference/pip_install/#id18)
reference.

### Using conda

[![Anaconda-Server Badge](https://img.shields.io/conda/vn/conda-forge/xsum.svg)](https://anaconda.org/conda-forge/xsum)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/xsum.svg)](https://anaconda.org/conda-forge/xsum)

**conda** is the package management tool for Anaconda Python installations.

Installing `xsum` from the `conda-forge` channel can be achieved by
adding `conda-forge` to your channels with:

```sh
conda config --add channels conda-forge
```

Once the `conda-forge` channel has been enabled, `xsum` can be
installed with:

```sh
conda install xsum
```

It is possible to list all of the versions of `xsum` available on your platform
with:

```sh
conda search xsum --channel conda-forge
```

### Python examples

```py
from xsum import *
import numpy as np

# A small superaccumulator
sacc = xsum_small_accumulator()

# Adding values to the small accumulator sacc
xsum_add(sacc, 1.0)
xsum_add(sacc, 2.0)

# Large superaccumulator
lacc = xsum_large_accumulator()

# Adding values to the large accumulator lacc
xsum_add(lacc, 1.0e-3)
xsum_add(lacc, 2.0e-3)
```

One can also add a vector of numbers to a superaccumulator.

```py
from xsum import *
import numpy as np

# A small superaccumulator
sacc = xsum_small_accumulator()

a = np.arange(0, 1, 0.1)

# Adding a vector of numbers
xsum_add(sacc, a)

print("sum       = {:.20f}".format(np.sum(a)))
print("Exact sum = {:.20f}".format(xsum_round(sacc)))
```

or a `xsum_small` or `xsum_large` objects can simply be used as,

```py
from xsum import *
import numpy as np

# A small superaccumulator
sacc = xsum_small()

a = np.arange(0, 1, 0.1)

# Adding a vector of numbers
sacc.add(a)

print("sum       = {:.20f}".format(np.sum(a)))
print("Exact sum = {:.20f}".format(sacc.round()))
```

running the `simple.py` script would result,

```bash
python ./simple.py

sum       = 4.50000000000000088818
Exact sum = 4.50000000000000000000
```

## References

<a name="neal_2015"></a>

1. Neal, Radford M., "Fast exact summation using small and large superaccumulators," [arXiv e-prints](https://arxiv.org/abs/1505.05571), (2015)
2. https://www.cs.toronto.edu/~radford/xsum.software.html
3. https://gitlab.com/radfordneal/xsum

## Contributing

Copyright (c) 2020, Regents of the University of Minnesota.\
All rights reserved.

Contributors:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Yaser Afshar

## License

[LGPLv2.1](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html)
