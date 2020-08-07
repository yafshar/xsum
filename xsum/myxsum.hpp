#ifndef MYXSUM_HPP
#define MYXSUM_HPP

#include <mpi.h>

#include "xsum.hpp"

/*!
 * \brief Create a small accumulator mpi type object
 *
 * \param small_accumulator_type
 */
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

MPI_Datatype create_small_accumulator_mpi_type() {
    MPI_Datatype small_accumulator_type;
    int const lengths[4] = {XSUM_SCHUNKS, 1, 1, 1};
    MPI_Aint const displacements[4] = {0,
                                       sizeof(xsum_schunk) * XSUM_SCHUNKS,
                                       sizeof(xsum_schunk) * (XSUM_SCHUNKS + 1),
                                       sizeof(xsum_schunk) * (XSUM_SCHUNKS + 2)};
    MPI_Datatype types[4] = {MPI_INT64_T, MPI_INT64_T, MPI_INT64_T, MPI_INT};

    MPI_Type_create_struct(4, lengths, displacements, types, &small_accumulator_type);
    MPI_Type_commit(&small_accumulator_type);
    return small_accumulator_type;
}

/*!
 * \brief destroy the created small accumulator mpi type object
 *
 * \param small_accumulator_type
 */
void destroy_small_accumulator_mpi_type(MPI_Datatype &small_accumulator_type) {
    MPI_Type_free(&small_accumulator_type);
}

/*!
 * \brief A user-defined xsum function works on \c xsum_small_accumulator type
 *
 * This is a user-defined global xsum operation on \c xsum_small_accumulator
 * type to an op handle that can subsequently be used in
 * \c MPI_Reduce, \c MPI_Allreduce, \c MPI_Reduce_scatter, and \c MPI_Scan.
 *
 * \param invec arrays of len elements that function is combining.
 * \param inoutvec arrays of len elements that function is combining.
 * \param len length
 * \param datatype a handle to the data type that was passed into the call
 */
void myXSUM(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype) {
    xsum_small_accumulator *in = static_cast<xsum_small_accumulator *>(invec);
    xsum_small_accumulator *inout = static_cast<xsum_small_accumulator *>(inoutvec);

    for (int i = 0; i < *len; ++i, ++in, ++inout) {
        xsum_small_add(inout, in);
    }
}

/* Create the XSUM user-op */
void create_XSUM(MPI_Op &XSUM) {
    MPI_Op_create(&myXSUM, true, &XSUM);
    return XSUM;
}

/* Create the XSUM user-op */
MPI_Op create_XSUM() {
    MPI_Op XSUM;
    MPI_Op_create(&myXSUM, true, &XSUM);
    return XSUM;
}

/* Free the created user-op */
void destroy_XSUM(MPI_Op &XSUM) {
    MPI_Op_free(&XSUM);
}

#endif  // MYXSUM_HPP