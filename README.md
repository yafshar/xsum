# xsum
Fast Exact Summation Using Small and Large Superaccumulators

This library is an easy to use header-only cross-platform C++11 implementation
and updates to the
[Fast Exact Summation Using Small and Large Superaccumulators](https://www.cs.toronto.edu/~radford/xsum.software.html).

It partly contains the methods described in the paper "Fast Exact Summation
Using Small and Large Superaccumulators", by Radford M. Neal, available at
[Fast Exact Summation](https://arxiv.org/abs/1505.05571), and extra summation
functionality especially for use in High Performance Message Passing Libraries.

It has an `op` handle that can subsequently be used in `MPI_Reduce`,
`MPI_Allreduce`, `MPI_Reduce_scatter`, and `MPI_Scan` calls.

## Installation

## Example

```cpp
#include <mpi.h>

#include <iomanip>
#include <iostream>

#include <xsum/xsum.hpp>
#include <xsum/myxsum.hpp>

int main() {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Datatype sacc_mpi = create_small_accumulator_mpi_type();

    /* Create the XSUM user-op */
    MPI_Op XSUM = create_XSUM();

    double const a = 0.239e-3;
    double s(0);

    xsum_large_accumulator lacc;

    for (int i = 0; i < 1000; ++i) {
        s += a;
        xsum_large_add(&lacc, a);
    }

    double ss;
    xsum_small_accumulator sacc;
    xsum_small_accumulator *lacc_sacc = xsum_large_round_to_small_accumulator(&lacc);

    MPI_Allreduce(&s, &ss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(lacc_sacc, &sacc, 1, sacc_mpi, XSUM, MPI_COMM_WORLD);

    if (world_rank == 0) {
        std::cout << "Rank =  " << world_rank
                  << ", sum 1 =  " << std::setprecision(20) << ss
                  << ", sum 2 =  " << std::setprecision(20) << xsum_small_round(&sacc) << std::endl;
    }

    /* Free the created user-op */
    destroy_XSUM(&XSUM);

    destroy_small_accumulator_mpi_type(&sacc_mpi);

    // Finalize the MPI environment.
    MPI_Finalize();
}
```


Contributer:
- @yafshar
