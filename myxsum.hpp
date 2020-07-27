#ifndef MYXSUM_HPP
#define MYXSUM_HPP

#include "xsum.hpp"

#include <mpi.h>

void myXSUM(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype)
{
    double *in = static_cast<double *>(invec);
    double *inout = static_cast<double *>(inoutvec);

    for (int i = 0; i < *len; ++i, ++in, ++inout)
    {
        xsum_large lacc;
        lacc.add(*in);
        lacc.add(*inout);
        *inout = static_cast<double>(lacc.round());
    }
}

#endif // MYXSUM_HPP

// MPI_Op XSUM;

// /* create the XSUM user-op */
// MPI_Op_create(myXSUM, true, &XSUM);