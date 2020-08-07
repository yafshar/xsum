#ifndef MYXSUM_HPP
#define MYXSUM_HPP

#include <mpi.h>

#include "xsum.hpp"

void create_small_accumulator_mpi_type(MPI_Datatype &small_accumulator_type) {
    int const lengths[4] = {XSUM_SCHUNKS, 1, 1, 1};
    MPI_Aint const displacements[4] = {0,
                                       sizeof(xsum_schunk) * XSUM_SCHUNKS,
                                       sizeof(xsum_schunk) * (XSUM_SCHUNKS + 1),
                                       sizeof(xsum_schunk) * (XSUM_SCHUNKS + 2)};
    MPI_Datatype types[4] = {MPI_INT64_T, MPI_INT64_T, MPI_INT64_T, MPI_INT};

    MPI_Type_create_struct(4, lengths, displacements, types, &small_accumulator_type);
    MPI_Type_commit(&small_accumulator_type);
}

void destroy_small_accumulator_mpi_type(MPI_Datatype &small_accumulator_type) {
    MPI_Type_free(&small_accumulator_type);
}

void myXSUM(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype) {
    xsum_small_accumulator *in = static_cast<xsum_small_accumulator *>(invec);
    xsum_small_accumulator *inout = static_cast<xsum_small_accumulator *>(inoutvec);

    for (int i = 0; i < *len; ++i, ++in, ++inout) {
        xsum_small_add(inout, in);
    }
}

#endif  // MYXSUM_HPP