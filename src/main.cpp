#include <mpi.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "myxsum.hpp"
#include "xsum.hpp"

// int main() {
// xsum_large s1;
// xsum_large s2;
// xsum_large s3;
// xsum_small s4;
// xsum_small s5;
// xsum_small s6;

// double const a = -123.0e-4;
// double const b = 1.0e-3;
// double ss = 0.0;

// long double aa = -123.0e-4;
// long double bb = 1.0e-3;
// long double ssss = 0.0;

// for (int i = 0; i < 10; ++i) {
//     // s1.add(a);
//     // s1.add(b);
//     // s2.add(a);
//     // s3.add(b);
//     s4.add(a);
//     s4.add(b);
//     s5.add(a);
//     s6.add(b);
//     ss += a + b;
//     ssss += aa + bb;
// }

// // s2.add(*s3.get());
// s5.add(*s6.get());

// xsum_large s1(*s4.get());

// // std::cout << "s1 = " << std::setprecision(20) << s1.round() << std::endl;
// // std::cout << "s2 = " << std::setprecision(20) << s2.round() << std::endl;
// std::cout << "ss = " << std::setprecision(20) << ss << std::endl;
// std::cout << "s4 = " << std::setprecision(20) << s4.round() << std::endl;
// std::cout << "s5 = " << std::setprecision(20) << s5.round() << std::endl;
// std::cout << "s1 = " << std::setprecision(20) << s1.round() << std::endl;

// std::vector<double> a(100, 1e-15);
// // std::iota(a.begin(), a.end(), 0.0);

// auto const s1 = std::accumulate(a.begin(), a.end(), 0);
// std::cout << "sum s1 = " << std::setprecision(20) << s1 << std::endl;

// auto const s2 = xsum_sum_double(a.data(), a.size());
// std::cout << "sum s2 = " << std::setprecision(20) << s2 << std::endl;

// xsum_small s3;
// for (int i = 0; i < 100; ++i)
// {
//     s3.add(a[i]);
// }
// // s3.add(a.data(), a.size());
// std::cout << "sum s3 = " << std::setprecision(20) << s3.round() << std::endl;

// xsum_large s4;
// for (int i = 0; i < 100; ++i)
// {
//     s4.add(a[i]);
// }
// // s4.add(a.data(), a.size());
// std::cout << "sum s4 = " << std::setprecision(20) << s4.round() << std::endl;
// }

int main() {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Datatype sacc_mpi;
    create_small_accumulator_mpi_type(sacc_mpi);

    MPI_Op XSUM;
    /* Create the XSUM user-op */
    MPI_Op_create(myXSUM, true, &XSUM);

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
    MPI_Op_free(&XSUM);

    destroy_small_accumulator_mpi_type(sacc_mpi);

    // Finalize the MPI environment.
    MPI_Finalize();
}

// int main() {
//     double const a = 0.0000071;

//     double s(0);
//     xsum_large_accumulator lacc;

//     for (int i = 0; i < 1000; ++i) {
//         s += a;
//         xsum_large_add(&lacc, a);
//     }

//     std::cout << "sum 1 =  " << std::setprecision(20) << s
//               << ", sum 2 =  " << std::setprecision(20) << xsum_large_round(&lacc) << std::endl;
// }
