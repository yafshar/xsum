#include "xsum.hpp"
#include "myxsum.hpp"

#include <mpi.h>

#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>

int main()
{
    std::vector<double> a(100, 1e-15);
    // std::iota(a.begin(), a.end(), 0.0);

    auto const s1 = std::accumulate(a.begin(), a.end(), 0);
    std::cout << "sum s1 = " << std::setprecision(20) << s1 << std::endl;

    auto const s2 = xsum_sum_double(a.data(), a.size());
    std::cout << "sum s2 = " << std::setprecision(20) << s2 << std::endl;

    xsum_small s3;
    for (int i = 0; i < 100; ++i)
    {
        s3.add(a[i]);
    }
    // s3.add(a.data(), a.size());
    std::cout << "sum s3 = " << std::setprecision(20) << s3.round() << std::endl;

    xsum_large s4;
    for (int i = 0; i < 100; ++i)
    {
        s4.add(a[i]);
    }
    // s4.add(a.data(), a.size());
    std::cout << "sum s4 = " << std::setprecision(20) << s4.round() << std::endl;

    // // Initialize the MPI environment
    // MPI_Init(NULL, NULL);

    // // Get the number of processes
    // int world_size;
    // MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // // Get the rank of the process
    // int world_rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // // MPI_Datatype datatype;
    // // MPI_Type_contiguous(sizeof(xsum_large_accumulator), MPI_BYTE, &datatype);
    // // MPI_Type_commit(&datatype);

    // MPI_Op XSUM;

    // /* Create the XSUM user-op */
    // MPI_Op_create(myXSUM, true, &XSUM);

    // double const a = 1e-15;
    // double s1(0);
    // double s2(0);
    // double s3(0);
    // double s4(0);
    // xsum_large s5;

    // for (int i = 0; i < 100; ++i)
    // {
    //     s3 += a;
    //     s5.add(a);
    // }

    // s4 = s5.round();

    // xsum_small s1;
    // xsum_large s2;
    // double s(0);
    // for (int i = 0; i < 4; ++i)
    // {
    //     double const a = static_cast<double>(i + 1) + 0.0000001;
    //     s1.add(a);
    //     s2.add(a);
    //     s += a;
    // }

    // MPI_Allreduce(&s3, &s1, 1, MPI_DOUBLE, XSUM, MPI_COMM_WORLD);
    // MPI_Allreduce(&s4, &s2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // std::cout << "Rank=" << world_rank
    //           << ", sum 1=" << std::setprecision(20) << s3
    //           << ", sum 2=" << std::setprecision(20) << s4 << std::endl;

    // std::cout << "sum =" << std::setprecision(20) << s
    //           << ", sum 1=" << std::setprecision(20) << s1.round()
    //           << ", sum 2=" << std::setprecision(10) << s2.round() << std::endl;

    // /* Free the created user-op */
    // MPI_Op_free(&XSUM);

    // // Finalize the MPI environment.
    // MPI_Finalize();
}