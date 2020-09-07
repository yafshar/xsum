//
// MYXSUM.hpp
//
// LGPL Version 2.1 HEADER START
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
//
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
// MA 02110-1301  USA
//
// LGPL Version 2.1 HEADER END
//

//
// Copyright (c) 2020, Regents of the University of Minnesota.
// All rights reserved.
//
// Contributors:
//    Yaser Afshar
//

#ifndef MYXSUM_HPP
#define MYXSUM_HPP

#include <mpi.h>

#include "xsum.hpp"

namespace {

/*!
 * \brief Create a mpi type object
 *
 * \tparam T data type one of \c xsum_small_accumulator or \c
 * xsum_large_accumulator \param datatype
 */
template <typename T>
void create_mpi_type(MPI_Datatype &datatype);

/*!
 * \brief Create a mpi type object
 *
 * \tparam T data type one of \c xsum_small_accumulator or \c
 * xsum_large_accumulator \return MPI_Datatype
 */
template <typename T>
MPI_Datatype create_mpi_type();

/*!
 * \brief destroy the user created mpi type object
 *
 * \param datatype
 */
void destroy_mpi_type(MPI_Datatype &datatype);

/*!
 * \brief A user-defined xsum function
 *
 * This is a user-defined global xsum operation on \c xsum_small_accumulator or
 * \c xsum_large_accumulator type to an op handle that can subsequently be used
 * in \c MPI_Reduce, \c MPI_Allreduce, \c MPI_Reduce_scatter, and \c MPI_Scan.
 *
 * \tparam T data type one of \c xsum_small_accumulator or \c
 * xsum_large_accumulator \param invec arrays of len elements that \c myXSUM
 * function is combining. \param inoutvec arrays of len elements that \c myXSUM
 * function is combining. \param len length \param datatype a handle to the data
 * type that was passed into the call
 */
template <typename T>
void myXSUM(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype);

/* Create the XSUM user-op */
template <typename T>
void create_XSUM(MPI_Op &SSUM);

/* Create the XSUM user-op */
template <typename T>
MPI_Op create_XSUM();

/* Free the created user-op */
void destroy_XSUM(MPI_Op &SSUM);

// Implementation

template <typename T>
void create_mpi_type(MPI_Datatype &datatype) {
  std::cerr << "Not implemented on purpose!" << std::endl;
  int ierr;
  MPI_Abort(MPI_COMM_WORLD, ierr);
}

template <>
void create_mpi_type<xsum_small_accumulator>(
    MPI_Datatype &small_accumulator_type) {
  int const lengths[4] = {XSUM_SCHUNKS, 1, 1, 1};
  MPI_Aint const displacements[4] = {0, sizeof(xsum_schunk) * XSUM_SCHUNKS,
                                     sizeof(xsum_schunk) * (XSUM_SCHUNKS + 1),
                                     sizeof(xsum_schunk) * (XSUM_SCHUNKS + 2)};
  MPI_Datatype const types[4] = {MPI_INT64_T, MPI_INT64_T, MPI_INT64_T,
                                 MPI_INT};
  MPI_Type_create_struct(4, lengths, displacements, types,
                         &small_accumulator_type);
  MPI_Type_commit(&small_accumulator_type);
}

template <>
void create_mpi_type<xsum_large_accumulator>(
    MPI_Datatype &large_accumulator_type) {
  int const lengths[8] = {
      XSUM_LCHUNKS, XSUM_LCHUNKS, XSUM_LCHUNKS / 64, 1, XSUM_SCHUNKS, 1, 1, 1};
  MPI_Aint const d1 = sizeof(xsum_lchunk) * XSUM_LCHUNKS;
  MPI_Aint const d2 = d1 + sizeof(xsum_lcount) * XSUM_LCHUNKS;
  MPI_Aint const d3 = d2 + sizeof(xsum_used) * XSUM_LCHUNKS / 64;
  MPI_Aint const d4 = d3 + sizeof(xsum_used);
  MPI_Aint const d5 = d4 + sizeof(xsum_schunk) * XSUM_SCHUNKS;
  MPI_Aint const d6 = d5 + sizeof(xsum_schunk);
  MPI_Aint const d7 = d6 + sizeof(xsum_schunk);
  MPI_Aint const displacements[8] = {0, d1, d2, d3, d4, d5, d6, d7};
  MPI_Datatype const types[8] = {MPI_INT64_T,  MPI_INT16_T, MPI_UINT64_T,
                                 MPI_UINT64_T, MPI_INT64_T, MPI_INT64_T,
                                 MPI_INT64_T,  MPI_INT};
  MPI_Type_create_struct(8, lengths, displacements, types,
                         &large_accumulator_type);
  MPI_Type_commit(&large_accumulator_type);
}

template <typename T>
MPI_Datatype create_mpi_type() {
  std::cerr << "Not implemented on purpose!" << std::endl;
  int ierr;
  MPI_Abort(MPI_COMM_WORLD, ierr);
  return MPI_DATATYPE_NULL;
}

template <>
MPI_Datatype create_mpi_type<xsum_small_accumulator>() {
  MPI_Datatype small_accumulator_type;
  int const lengths[4] = {XSUM_SCHUNKS, 1, 1, 1};
  MPI_Aint const displacements[4] = {0, sizeof(xsum_schunk) * XSUM_SCHUNKS,
                                     sizeof(xsum_schunk) * (XSUM_SCHUNKS + 1),
                                     sizeof(xsum_schunk) * (XSUM_SCHUNKS + 2)};
  MPI_Datatype const types[4] = {MPI_INT64_T, MPI_INT64_T, MPI_INT64_T,
                                 MPI_INT};
  MPI_Type_create_struct(4, lengths, displacements, types,
                         &small_accumulator_type);
  MPI_Type_commit(&small_accumulator_type);
  return small_accumulator_type;
}

template <>
MPI_Datatype create_mpi_type<xsum_large_accumulator>() {
  MPI_Datatype large_accumulator_type;
  int const lengths[8] = {
      XSUM_LCHUNKS, XSUM_LCHUNKS, XSUM_LCHUNKS / 64, 1, XSUM_SCHUNKS, 1, 1, 1};
  MPI_Aint const d1 = sizeof(xsum_lchunk) * XSUM_LCHUNKS;
  MPI_Aint const d2 = d1 + sizeof(xsum_lcount) * XSUM_LCHUNKS;
  MPI_Aint const d3 = d2 + sizeof(xsum_used) * XSUM_LCHUNKS / 64;
  MPI_Aint const d4 = d3 + sizeof(xsum_used);
  MPI_Aint const d5 = d4 + sizeof(xsum_schunk) * XSUM_SCHUNKS;
  MPI_Aint const d6 = d5 + sizeof(xsum_schunk);
  MPI_Aint const d7 = d6 + sizeof(xsum_schunk);
  MPI_Aint const displacements[8] = {0, d1, d2, d3, d4, d5, d6, d7};
  MPI_Datatype const types[8] = {MPI_INT64_T,  MPI_INT16_T, MPI_UINT64_T,
                                 MPI_UINT64_T, MPI_INT64_T, MPI_INT64_T,
                                 MPI_INT64_T,  MPI_INT};
  MPI_Type_create_struct(8, lengths, displacements, types,
                         &large_accumulator_type);
  MPI_Type_commit(&large_accumulator_type);
  return large_accumulator_type;
}

void destroy_mpi_type(MPI_Datatype &user_type) { MPI_Type_free(&user_type); }

template <typename T>
void myXSUM(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype) {
  std::cerr << "Not implemented on purpose!" << std::endl;
  int ierr;
  MPI_Abort(MPI_COMM_WORLD, ierr);
}

template <>
void myXSUM<xsum_small_accumulator>(void *invec, void *inoutvec, int *len,
                                    MPI_Datatype * /* datatype*/) {
  xsum_small_accumulator *in = static_cast<xsum_small_accumulator *>(invec);
  xsum_small_accumulator *inout =
      static_cast<xsum_small_accumulator *>(inoutvec);

  for (int i = 0; i < *len; ++i, ++in, ++inout) {
    xsum_add<xsum_small_accumulator>(inout, in);
  }
}

template <>
void myXSUM<xsum_large_accumulator>(void *invec, void *inoutvec, int *len,
                                    MPI_Datatype * /* datatype*/) {
  xsum_large_accumulator *in = static_cast<xsum_large_accumulator *>(invec);
  xsum_large_accumulator *inout =
      static_cast<xsum_large_accumulator *>(inoutvec);

  for (int i = 0; i < *len; ++i, ++in, ++inout) {
    xsum_add<xsum_large_accumulator>(inout, in);
  }
}

template <typename T>
void create_XSUM(MPI_Op &XSUM) {
  std::cerr << "Not implemented on purpose!" << std::endl;
  int ierr;
  MPI_Abort(MPI_COMM_WORLD, ierr);
}

template <>
void create_XSUM<xsum_small_accumulator>(MPI_Op &XSUM) {
  MPI_Op_create(&myXSUM<xsum_small_accumulator>, true, &XSUM);
}

template <>
void create_XSUM<xsum_large_accumulator>(MPI_Op &XSUM) {
  MPI_Op_create(&myXSUM<xsum_large_accumulator>, true, &XSUM);
}

template <typename T>
MPI_Op create_XSUM() {
  std::cerr << "Not implemented on purpose!" << std::endl;
  int ierr;
  MPI_Abort(MPI_COMM_WORLD, ierr);
  return MPI_OP_NULL;
}

template <>
MPI_Op create_XSUM<xsum_small_accumulator>() {
  MPI_Op XSUM;
  MPI_Op_create(&myXSUM<xsum_small_accumulator>, true, &XSUM);
  return XSUM;
}

template <>
MPI_Op create_XSUM<xsum_large_accumulator>() {
  MPI_Op XSUM;
  MPI_Op_create(&myXSUM<xsum_large_accumulator>, true, &XSUM);
  return XSUM;
}

void destroy_XSUM(MPI_Op &XSUM) { MPI_Op_free(&XSUM); }
}  // namespace

#endif  // MYXSUM_HPP