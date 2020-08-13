# XSUM
Fast Exact Summation Using Small and Large Superaccumulators[1].

This library is an easy to use header-only cross-platform C++11 implementation
of the
[Fast Exact Summation Using Small and Large Superaccumulators](#neal_2015).

The current implemenation contains the methods described in the paper "Fast
Exact Summation Using Small and Large Superaccumulators", by Radford M. Neal,
available at [Fast Exact Summation](https://arxiv.org/abs/1505.05571). It has
been rewritten with the goal of ease of use and portability and it also includes
extra classes and summation functionality especially for use in High Performance
Message Passing Libraries. Where an `op` handle can subsequently be used in
`MPI_Reduce`,`MPI_Allreduce`, `MPI_Reduce_scatter`, and `MPI_Scan` calls.

- **NOTE:** To see, use or reproduce the results of the original implementation
  reported in the paper `Fast Exact Summation Using Small and Large Superaccumulators`,
  by Radford M. Neal, please refer to
  [FUNCTIONS FOR EXACT SUMMATION](https://gitlab.com/radfordneal/xsum).

Usage
-----
A small superaccumulator is the preferred method for summing a moderate number
of terms. It is also a component of the large superaccumulator.

`xsum_small_accumulator` and `xsum_large_accumulator`, both have a default
constructor, thus they do not need to be initialized. But when it is needed,
one can simply use the `xsum_init` to initilize them again.

Addition operation is simply a `xsum_add`,
```cpp
// Adding double values to the small accumulator sacc
xsum_small_accumulator sacc;

xsum_add(&sacc, 1.0);
xsum_add(&sacc, 2.0);
```

A small superaccumulator can be added to a large superaccumulator, as well as
two small superaccumulators can be added together. `xsum_add` can be used to add
the second superaccumulator to the first one without doing any rounding.

Two large superaccumulator can also be added in the same way. In the case of the
large superaccumulators, the second one internally will be rounded to a small
superaccumulator and then the addition can be done.

```cpp
// Small acumulators
xsum_small_accumulator sacc1;
xsum_small_accumulator sacc2;

xsum_add(&sacc1, 1.0);
xsum_add(&sacc2, 2.0);

xsum_add(&sacc1, &sacc2);

// Large acumulators
xsum_large_accumulator lacc1;
xsum_large_accumulator lacc2;

xsum_add(&lacc1, 1.0);
xsum_add(&lacc2, 2.0);

xsum_add(&lacc1, &lacc2);

// Addition of a small acumulator to a large one
xsum_add(&lacc1, &sacc1);
```

The large superaccumulator can be rounded to the small one as,
```cpp
xsum_large_accumulator lacc;

xsum_small_accumulator *sacc = xsum_round_to_small(&lacc);
```

The accumulators can be rounded as,
```cpp
double s = xsum_round(&sacc1);
double l = xsum_round(&lacc1);
```

## Example

Two simple examples on how to use the library:

```cpp
#include <iostream>
#include "xsum/xsum.hpp"

int main() {
    xsum_large_accumulator lacc;
    double const a = 0.123e-10;
    for (int i = 0; i < 1000; ++i) {
        xsum_add(&lacc, a);
    }
    std::cout << xsum_round(&lacc) << std::endl;
}
```

or a small/large superaccumulator class can simply be used as,
```cpp
#include <iostream>
#include "xsum/xsum.hpp"

int main() {
    xsum_large lacc;
    double const a = 0.123e-10;
    for (int i = 0; i < 1000; ++i) {
        lacc.add(a);
    }
    std::cout << lacc.round() << std::endl;
}
```

```bash
g++ simple.cpp -std=c++11 -O3 -o simple
```
or
```bash
icpc simple.cpp -std=c++11 -O3 -fp-model=double -o simple
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

#include "xsum/xsum.hpp"
#include "xsum/myxsum.hpp"

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
        xsum_large_add(&lacc, a);
    }

    MPI_Allreduce(MPI_IN_PLACE, &s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &lacc, 1, acc_mpi, XSUM, MPI_COMM_WORLD);

    if (world_rank == 0) {
        std::cout << "Rank =  " << world_rank
                  << ", sum   =  " << std::setprecision(20) << a * 1000 * world_size
                  << ", sum 1 =  " << std::setprecision(20) << s
                  << ", sum 2 =  " << std::setprecision(20) << xsum_large_round(&lacc) << std::endl;
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
1. [Neal, Radford M., "Fast exact summation using small and large superaccumulators," arXiv e-prints, (2015)](https://arxiv.org/abs/1505.05571).
2. https://www.cs.toronto.edu/~radford/xsum.software.html
3. https://gitlab.com/radfordneal/xsum

## Contributing

Copyright (c) 2019--2020, Regents of the University of Minnesota.\
All rights reserved.

Contributors:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Yaser Afshar

## License

[LGPLv2.1](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html)
