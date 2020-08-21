# Fast Exact Summation Using Small and Large Superaccumulators (XSUM)

[![Build Status](https://travis-ci.com/yafshar/xsum.svg?token=aY1dW9PfH9SMySdB6Pzy&branch=master)](https://travis-ci.com/yafshar/xsum)
[![Python package](https://github.com/yafshar/xsum/workflows/Python%20package/badge.svg)](https://github.com/yafshar/xsum/actions)
[![License](https://img.shields.io/badge/license-LGPL--v2-blue)](LICENSE)

[XSUM](#neal_2015) is a library for performing exact summation using
super-accumulators. It provides methods for exactly summing a set of
floating-point numbers, where using a simple summation and the rounding which
happens after each addition could be an important factor in many applications.

This library is an easy to use header-only cross-platform C++11 implementation
and also contains Python bindings.

The main algorithm is taken from the original C library
[FUNCTIONS FOR EXACT SUMMATION](https://gitlab.com/radfordneal/xsum) described
in the paper
["Fast Exact Summation Using Small and Large Superaccumulators,"](#neal_2015) by
[Radford M. Neal](https://www.cs.toronto.edu/~radford).

The code is rewritten in C++ and amended with more functionality with the goal
of ease of use. The provided Python bindings provide the *exact summation*
interface in a Python code.

The C++ code also includes extra summation functionalities, which are especially
useful in high-performance message passing libraries (like
[OpenMPI](https://www.open-mpi.org/) and [MPICH](https://www.mpich.org/)).
Where binding a user-defined global summation operation to an `op` handle can
subsequently be used in `MPI_Reduce,` `MPI_Allreduce,` `MPI_Reduce_scatter,` and
`MPI_Scan` or a similar calls.

- **NOTE:** To see or use or reproduce the results of the original
  implementation reported in the paper
  `Fast Exact Summation Using Small and Large Superaccumulators`, by Radford M.
  Neal, please refer to
  [FUNCTIONS FOR EXACT SUMMATION](https://gitlab.com/radfordneal/xsum).

## Usage

A small superaccumulator is the preferred method for summing a moderate number
of terms. It is also a component of the large superaccumulator.

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

One can also add a vector of numbers to a superaccumulator.

```cpp
// A small superaccumulator
xsum_small_accumulator sacc;

// Adding a vector of numbers
double vec[] = {1.234e88, -93.3e-23, 994.33, 1334.3, 457.34, -1.234e88, 93.3e-23, -994.33, -1334.3, -457.34};

xsum_add(&sacc, vec, 10);
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

The superaccumulator can be rounded as,

```cpp
xsum_small_accumulator sacc;
xsum_add(&sacc, 1.0e-15);

....

double s = xsum_round(&sacc);
```

where, `xsum_round` is used to round the superaccumulator to the nearest
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

The large superaccumulator can be rounded to a small one as,

```cpp
xsum_large_accumulator lacc;

xsum_small_accumulator sacc = xsum_round_to_small(&lacc);
```

### Example

Two simple examples on how to use the library:

```cpp
#include <iomanip>
#include <iostream>

#include "xsum/xsum.hpp"

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

### MPI example (`MPI_Allreduce`)

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
mpirunt -np 4 ./simple

Rank =  0, sum   =  0.95600000000000007194, sum 1 =  0.95599999999998419575, sum 2 =  0.95600000000000007194
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
