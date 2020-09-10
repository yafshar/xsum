//
// Copyright (c) 2020, Regents of the University of Minnesota.
// All rights reserved.
//
// Contributors:
//    Yaser Afshar
//

// CORRECTNESS CHECKS FOR FUNCTIONS FOR EXACT SUMMATION ON MULTI PROCESSORS

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#include "../xsum/myxsum.hpp"
#include "../xsum/xsum.hpp"

using namespace xsum;

xsum_flt term1[] = {1.234e88, -93.3e-23, 994.33,  1334.3,  457.34, -1.234e88,
                    93.3e-23, -994.33,   -1334.3, -457.34, 0};
xsum_flt term2[] = {1.,
                    -23.,
                    456.,
                    -78910.,
                    1112131415.,
                    -161718192021.,
                    22232425262728.,
                    -2930313233343536.,
                    373839404142434445.,
                    -46474849505152535455.,
                    -46103918342424313856.};
xsum_flt term3[] = {2342423.3423, 34234.450,  945543.4,          34345.34343,
                    1232.343,     0.00004343, 43423.0,           -342344.8343,
                    -89544.3435,  -34334.3,   2934978.4009734304};
xsum_flt term4[] = {0.9101534, 0.9048397, 0.4036596, 0.1460245,
                    0.2931254, 0.9647649, 0.1125303, 0.1574193,
                    0.6522300, 0.7378597, 5.2826068};
xsum_flt term5[] = {428.366070546, 707.3261930632,  103.29267289,
                    9040.03475821, 36.2121638,      19.307901408,
                    1.4810709160,  8.077159101,     1218.907244150,
                    778.068267017, 12341.0735011012};
xsum_flt term6[] = {1.1e-322,
                    5.3443e-321,
                    -9.343e-320,
                    3.33e-314,
                    4.41e-322,
                    -8.8e-318,
                    3.1e-310,
                    4.1e-300,
                    -4e-300,
                    7e-307,
                    1.0000070031003328e-301};

int different(double const a, double const b) {
  return (std::isnan(a) != std::isnan(b)) ||
         (!std::isnan(a) && !std::isnan(b) && a != b);
}

void result(xsum_small_accumulator *const sacc, double const s, int const rank,
            const char *test) {
  double const r = xsum_round(sacc);
  double const r2 = xsum_round(sacc);

  if (different(r, r2)) {
    std::printf(" \n-- %s on processor %d\n", test, rank);
    std::printf("   ANSWER: %.16le\n", s);
    std::printf("small: Different second time %.16le != %.16le\n", r, r2);
  }

  if (different(r, s)) {
    std::printf(" \n-- %s on processor %d \n", test, rank);
    std::printf("   ANSWER: %.16le\n", s);
    std::printf("small: Result incorrect %.16le != %.16le\n", r, s);
    std::printf("    ");
    print_binary(r);
    std::printf("    ");
    print_binary(s);
  }
}

void result(xsum_large_accumulator *const lacc, double const s, int const rank,
            const char *test) {
  double const r = xsum_round(lacc);
  double const r2 = xsum_round(lacc);

  if (different(r, r2)) {
    std::printf(" \n-- %s on processor %d\n", test, rank);
    std::printf("   ANSWER: %.16le\n", s);
    std::printf("large: Different second time %.16le != %.16le\n", r, r2);
  }
  if (different(r, s)) {
    std::printf(" \n-- %s on processor %d \n", test, rank);
    std::printf("   ANSWER: %.16le\n", s);
    std::printf("large: Result incorrect %.16le != %.16le\n", r, s);
    std::printf("    ");
    print_binary(r);
    std::printf("    ");
    print_binary(s);
  }
}

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
  MPI_Datatype acc_mpi;
  create_mpi_type<xsum_small_accumulator>(acc_mpi);

  // Create the XSUM user-op
  MPI_Op XSUM;
  create_XSUM<xsum_small_accumulator>(XSUM);

  if (world_rank == 0) {
    std::cout << "\nCORRECTNESS MPI TESTS\n";
    std::cout << "SMALL ACCUMULATOR SUM TESTS\n";

    std::cout << "A: SMALL ACCUMULATOR, MPI_Allreduce with/out MPI_IN_PLACE\n";
  }

  {
    xsum_small_accumulator sacc;
    for (int i = 0; i < 10; ++i) {
      if ((i % world_size) == world_rank) {
        xsum_add(&sacc, term1[i]);
      }
    }

    MPI_Allreduce(MPI_IN_PLACE, &sacc, 1, acc_mpi, XSUM, MPI_COMM_WORLD);
    result(&sacc, term1[10], world_rank, "Test 1");
  }

  {
    xsum_small_accumulator ssacc;

    for (int j = 0; j < 1000; ++j) {
      for (int i = 0; i < 10; ++i) {
        if ((i % world_size) == world_rank) {
          xsum_add(&ssacc, term2[i]);
        }
      }
    }

    // double ss;
    xsum_small_accumulator sacc;

    MPI_Allreduce(&ssacc, &sacc, 1, acc_mpi, XSUM, MPI_COMM_WORLD);
    result(&sacc, term2[10] * 1000, world_rank, "Test 2");
  }

  {
    xsum_small ssacc;
    for (int i = 0; i < 10; ++i) {
      if ((i % world_size) == world_rank) {
        ssacc.add(term3[i]);
      }
    }

    xsum_small_accumulator sacc;

    MPI_Allreduce(ssacc.get(), &sacc, 1, acc_mpi, XSUM, MPI_COMM_WORLD);
    result(&sacc, term3[10], world_rank, "Test 3");
  }

  if (world_rank == 0) {
    std::cout << "B: SMALL ACCUMULATOR, LARGE round to SMALL\n";
  }

  {
    xsum_large_accumulator lacc;
    for (int i = 0; i < 10; ++i) {
      if ((i % world_size) == world_rank) {
        xsum_add(&lacc, term4[i]);
      }
    }

    xsum_small_accumulator ssacc = xsum_round_to_small(&lacc);
    xsum_small_accumulator sacc;

    MPI_Allreduce(&ssacc, &sacc, 1, acc_mpi, XSUM, MPI_COMM_WORLD);
    result(&sacc, term4[10], world_rank, "Test 4");
  }

  {
    xsum_large lacc;
    for (int i = 0; i < 10; ++i) {
      if ((i % world_size) == world_rank) {
        lacc.add(term5[i]);
      }
    }

    xsum_small_accumulator ssacc = lacc.round_to_small();
    xsum_small_accumulator sacc;

    MPI_Allreduce(&ssacc, &sacc, 1, acc_mpi, XSUM, MPI_COMM_WORLD);
    result(&sacc, term5[10], world_rank, "Test 5");
  }

  {
    xsum_large lacc;

    for (int i = 0; i < 10; ++i) {
      if ((i % world_size) == world_rank) {
        lacc.add(term6[i]);
      }
    }

    xsum_small_accumulator sacc = lacc.round_to_small();

    MPI_Allreduce(MPI_IN_PLACE, &sacc, 1, acc_mpi, XSUM, MPI_COMM_WORLD);
    result(&sacc, term6[10], world_rank, "Test 6");
  }

  // Free the created user-op
  MPI_Op_free(&XSUM);
  destroy_mpi_type(acc_mpi);

  // Create the MPI data type of the superaccumulator
  create_mpi_type<xsum_large_accumulator>(acc_mpi);

  // Create the XSUM user-op
  create_XSUM<xsum_large_accumulator>(XSUM);

  if (world_rank == 0) {
    std::cout << "\nLARGE ACCUMULATOR SUM TESTS\n";
    std::cout << "A: LARGE ACCUMULATOR, MPI_Allreduce with/out MPI_IN_PLACE\n";
  }

  {
    xsum_large_accumulator lacc;
    for (int i = 0; i < 10; ++i) {
      if ((i % world_size) == world_rank) {
        xsum_add(&lacc, term1[i]);
      }
    }

    MPI_Allreduce(MPI_IN_PLACE, &lacc, 1, acc_mpi, XSUM, MPI_COMM_WORLD);
    result(&lacc, term1[10], world_rank, "Test 1");
  }

  {
    xsum_large_accumulator llacc;

    for (int j = 0; j < 1000; ++j) {
      for (int i = 0; i < 10; ++i) {
        if ((i % world_size) == world_rank) {
          xsum_add(&llacc, term2[i]);
        }
      }
    }

    // double ss;
    xsum_large_accumulator lacc;

    MPI_Allreduce(&llacc, &lacc, 1, acc_mpi, XSUM, MPI_COMM_WORLD);
    result(&lacc, term2[10] * 1000, world_rank, "Test 2");
  }

  // Free the created user-op
  MPI_Op_free(&XSUM);

  // Free the created MPI data type
  destroy_mpi_type(acc_mpi);

  // Finalize the MPI environment.
  MPI_Finalize();
}
