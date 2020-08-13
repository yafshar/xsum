//
// Copyright (c) 2019--2020, Regents of the University of Minnesota.
// All rights reserved.
//
// Contributors:
//    Yaser Afshar
//

#include <mpi.h>

#include <iomanip>
#include <iostream>

#include "../xsum/xsum.hpp"
#include "../xsum/myxsum.hpp"

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
