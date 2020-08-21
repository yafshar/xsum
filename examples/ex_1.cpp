/* Copyright 2015, 2018 Radford M. Neal

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
   LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
   OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
   WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

//
// Copyright (c) 2020, Regents of the University of Minnesota.
// All rights reserved.
//
// Contributors:
//    Yaser Afshar
//
// Brief: This file is adapted and rewritten in C++ from the original
//        work of Radford M. Neal, 2015.
//        https://gitlab.com/radfordneal/xsum.git

// CORRECTNESS CHECKS FOR FUNCTIONS FOR EXACT SUMMATION.

#include <cmath>
#include <cstdio>
#include <iomanip>

#include "../xsum/xsum.hpp"

constexpr double pow2_16 = (1.0 / (1 << 16));
constexpr double pow2_32 = (pow2_16 * pow2_16);
constexpr double pow2_64 = (pow2_32 * pow2_32);
constexpr double pow2_128 = (pow2_64 * pow2_64);
constexpr double pow2_256 = (pow2_128 * pow2_128);
constexpr double pow2_512 = (pow2_256 * pow2_256);
constexpr double pow2_1024 = (pow2_512 * pow2_512);
constexpr double pow2_52 = (1.0 / (1 << 22) / (1 << 30));

/* Largest normal number */
constexpr double Lnormal =
    (2 * ((.5 / pow2_1024) - (.25 / pow2_1024) * pow2_52));
/* Smallest normal number */
constexpr double Snormal = (4 * pow2_1024);
/* Largest denormalized number */
constexpr double Ldenorm = (4 * pow2_1024 - 4 * pow2_1024 * pow2_52);
/* Smallest denormalized number > 0 */
constexpr double Sdenorm = (4 * pow2_1024 * pow2_52);

/* Repeat factor for second set of one term tests */
constexpr int REP1 = (1 << 23);
/* Repeat factor for second set of ten term tests */
constexpr int REP10 = (1 << 13);

/* Tests with one term. Answer should be the same as the term. */

xsum_flt one_term[] = {
    /* Some unexceptional examples of normal numbers */
    1.0,
    -1.0,
    0.1,
    -0.1,
    3.1,
    -3.1,
    2.3e10,
    -2.3e10,
    3.2e-10,
    -3.2e-10,
    123e123,
    -123e123,
    54.11e-150,
    -54.11e-150,
    /* Mantissa all 1s */
    2 * ((.5 / pow2_128) - (.25 / pow2_128) * pow2_52),
    -2 * ((.5 / pow2_128) + (.25 / pow2_128) * pow2_52),
    /* Largest normal number */
    Lnormal,
    -Lnormal,
    /* Smallest normal number */
    Snormal,
    -Snormal,
    /* Largest denormalized number */
    Ldenorm,
    -Ldenorm,
    /* Smallest denormalized number > 0 */
    Sdenorm,
    -Sdenorm,
    /* Other denormalized numbers */
    1.23e-309,
    -1.23e-309,
    4.57e-314,
    -4.57e-314,
    9.7e-322,
    -9.7e-322,
    Sdenorm / pow2_64 / 2,
    -Sdenorm / pow2_64 / 2};

constexpr int one_term_size = sizeof(one_term) / sizeof(one_term[0]);

/* Tests with two terms.  Answer should match ordinary floating point add. */

xsum_flt two_term[] = {
  /* Unexceptional adds of normal numbers */
  1.0, 2.0,
  -1.0, -2.0,
  0.1, 12.2,
  -0.1, -12.2,
  12.1, -11.3,
  -12.1, 11.3,
  11.3, -12.1,
  -11.3, 12.1,
  1.234567e14, 9.87654321,
  -1.234567e14, -9.87654321,
  1.234567e14, -9.87654321,
  -1.234567e14, 9.87654321,
  /* Smaller term should disappear */
  3.1e200, 1.7e-100,
  3.1e200, -1.7e-100,
  -3.1e200, 1.7e-100,
  -3.1e200, -1.7e-100,
  1.7e-100, 3.1e200,
  1.7e-100, -3.1e200,
  -1.7e-100, 3.1e200,
  -1.7e-100, -3.1e200,
  /* Test rounding */
  1, pow2_52,
  -1, -pow2_52,
  1, pow2_52 / 2,
  -1, -pow2_52 / 2,
  1, pow2_52 / 2 + pow2_52 / 4096,
  -1, -pow2_52 / 2 - pow2_52 / 4096,
  1, pow2_52 / 2 + pow2_52 / (1 << 30) / (1 << 10),
  -1, -pow2_52 / 2 - pow2_52 / (1 << 30) / (1 << 10),
  1, pow2_52 / 2 - pow2_52 / 4096,
  -1, -pow2_52 / 2 + pow2_52 / 4096,
  1 + pow2_52, pow2_52 / 2,
  1 + pow2_52, pow2_52 / 2 - pow2_52 *pow2_52,
  -(1 + pow2_52), -pow2_52 / 2,
  -(1 + pow2_52), -(pow2_52 / 2 - pow2_52 * pow2_52),
  /* Adds with denormalized numbers */
  Sdenorm, 7.1,
  Sdenorm, -7.1,
  -Sdenorm, -7.1,
  -Sdenorm, 7.1,
  7.1, Sdenorm,
  -7.1, Sdenorm,
  -7.1, -Sdenorm,
  7.1, -Sdenorm,
  Ldenorm, Sdenorm,
  Ldenorm, -Sdenorm,
  -Ldenorm, Sdenorm,
  -Ldenorm, -Sdenorm,
  Sdenorm, Sdenorm,
  Sdenorm, -Sdenorm,
  -Sdenorm, Sdenorm,
  -Sdenorm, -Sdenorm,
  Ldenorm, Snormal,
  Snormal, Ldenorm,
  -Ldenorm, -Snormal,
  -Snormal, -Ldenorm,
  4.57e-314, 9.7e-322,
  -4.57e-314, 9.7e-322,
  4.57e-314, -9.7e-322,
  -4.57e-314, -9.7e-322,
  4.57e-321, 9.7e-322,
  -4.57e-321, 9.7e-322,
  4.57e-321, -9.7e-322,
  -4.57e-321, -9.7e-322,
  2.0, -2.0 * (1 + pow2_52),
  /* Overflow */
  Lnormal, Lnormal,
  -Lnormal, -Lnormal,
  Lnormal, Lnormal *pow2_52 / 2,
  -Lnormal, -Lnormal *pow2_52 / 2,
  /* Infinity / NaN */
  1.0 / 0.0, 123,
  -1.0 / 0.0, 123,
  1.0 / 0.0, -1.0 / 0.0,
  0.0 / 0.0, 123,
  123, 0.0 / 0.0};

constexpr int two_term_size = sizeof(two_term) / sizeof(two_term[0]);

/* Tests with three terms.  Answers are given here as a fourth number,
   some computed/verified using Rmpfr in check.r. */

xsum_flt three_term[] = {
  Lnormal, Sdenorm, -Lnormal, Sdenorm,
  -Lnormal, Sdenorm, Lnormal, Sdenorm,
  Lnormal, -Sdenorm, -Lnormal, -Sdenorm,
  -Lnormal, -Sdenorm, Lnormal, -Sdenorm,
  Sdenorm, Snormal, -Sdenorm, Snormal,
  -Sdenorm, -Snormal, Sdenorm, -Snormal,
  12345.6, Snormal, -12345.6, Snormal,
  12345.6, -Snormal, -12345.6, -Snormal,
  12345.6, Ldenorm, -12345.6, Ldenorm,
  12345.6, -Ldenorm, -12345.6, -Ldenorm,
  2.0, -2.0 * (1 + pow2_52), pow2_52 / 8, -2 * pow2_52 + pow2_52 / 8,
  1.0, 2.0, 3.0, 6.0,
  12.0, 3.5, 2.0, 17.5,
  3423.34e12, -93.431, -3432.1e11, 3080129999999906.5,
  432457232.34, 0.3432445, -3433452433, -3000995200.3167553};

constexpr int three_term_size = sizeof(three_term) / sizeof(three_term[0]);

/* Tests with ten terms.  Answers are given here as an eleventh number,
   some computed/verified using Rmpfr in check.r. */

xsum_flt ten_term[] = {
  Lnormal, Lnormal, Lnormal, Lnormal, Lnormal, Lnormal, -Lnormal, -Lnormal, -Lnormal, -Lnormal, 1.0 / 0.0,
  -Lnormal, -Lnormal, -Lnormal, -Lnormal, -Lnormal, -Lnormal, Lnormal, Lnormal, Lnormal, Lnormal, -1.0 / 0.0,
  Lnormal, Lnormal, Lnormal, Lnormal, 0.125, 0.125, -Lnormal, -Lnormal, -Lnormal, -Lnormal, 0.25,
  2.0 * (1 + pow2_52), -2.0, -pow2_52, -pow2_52, 0, 0, 0, 0, 0, 0, 0,
  1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1111111111e0,
  -1e0, -1e1, -1e2, -1e3, -1e4, -1e5, -1e6, -1e7, -1e8, -1e9, -1111111111e0,
  1.234e88, -93.3e-23, 994.33, 1334.3, 457.34, -1.234e88, 93.3e-23, -994.33, -1334.3, -457.34, 0,
  1., -23., 456., -78910., 1112131415., -161718192021., 22232425262728., -2930313233343536., 373839404142434445., -46474849505152535455., -46103918342424313856.,
  2342423.3423, 34234.450, 945543.4, 34345.34343, 1232.343, 0.00004343, 43423.0, -342344.8343, -89544.3435, -34334.3, 2934978.4009734304,
  0.9101534, 0.9048397, 0.4036596, 0.1460245, 0.2931254, 0.9647649, 0.1125303, 0.1574193, 0.6522300, 0.7378597, 5.2826068,
  428.366070546, 707.3261930632, 103.29267289, 9040.03475821, 36.2121638, 19.307901408, 1.4810709160, 8.077159101, 1218.907244150, 778.068267017, 12341.0735011012,
  1.1e-322, 5.3443e-321, -9.343e-320, 3.33e-314, 4.41e-322, -8.8e-318, 3.1e-310, 4.1e-300, -4e-300, 7e-307, 1.0000070031003328e-301};

constexpr int ten_term_size = sizeof(ten_term) / sizeof(ten_term[0]);

int total_small_test(0);
int total_large_test(0);
int small_test_fails(0);
int large_test_fails(0);

int different(double const a, double const b) {
  return (std::isnan(a) != std::isnan(b)) ||
       (!std::isnan(a) && !std::isnan(b) && a != b);
}

void result(xsum_small_accumulator *const sacc, double const s, int const i) {
  double const r = xsum_round(sacc);
  double const r2 = xsum_round(sacc);
  ++total_small_test;

  if (different(r, r2)) {
    std::printf(" \n-- TEST %d\n", i);
    std::printf("   ANSWER: %.16le\n", s);
    std::printf("small: Different second time %.16le != %.16le\n", r, r2);
  }

  if (different(r, s)) {
    ++small_test_fails;
    std::printf(" \n-- TEST %d\n", i);
    std::printf("   ANSWER: %.16le\n", s);
    std::printf("small: Result incorrect %.16le != %.16le\n", r, s);
    std::printf("    ");
    pbinary(r);
    std::printf("    ");
    pbinary(s);
  }
}

void result(xsum_large_accumulator *const lacc, double const s, int const i) {
  double const r = xsum_round(lacc);
  double const r2 = xsum_round(lacc);
  ++total_large_test;

  if (different(r, r2)) {
    std::printf(" \n-- TEST %d\n", i);
    std::printf("   ANSWER: %.16le\n", s);
    std::printf("large: Different second time %.16le != %.16le\n", r, r2);
  }
  if (different(r, s)) {
    ++large_test_fails;
    std::printf(" \n-- TEST %d\n", i);
    std::printf("   ANSWER: %.16le\n", s);
    std::printf("large: Result incorrect %.16le != %.16le\n", r, s);
    std::printf("    ");
    pbinary(r);
    std::printf("    ");
    pbinary(s);
  }
}

int main(int argc, char **argv) {
  std::printf("\nCORRECTNESS TESTS\n");

  std::printf("\nA: ZERO TERM TEST\n");

  {
    xsum_small_accumulator sacc;
    xsum_large_accumulator lacc;
    result(&sacc, 0, 0);
    result(&lacc, 0, 0);
  }

  {
    xsum_small sacc;
    xsum_large lacc;
    result(sacc.get(), 0, 0);
    result(lacc.get(), 0, 0);
  }

  std::printf("\nB: ONE TERM TESTS\n");

  for (int i = 0; i < one_term_size; ++i) {
    double const s = one_term[i];

    xsum_small_accumulator sacc;
    xsum_add(&sacc, one_term[i]);
    result(&sacc, s, i);

    xsum_large_accumulator lacc;
    xsum_add(&lacc, *(one_term + i));
    result(&lacc, s, i);
  }

  for (int i = 0; i < one_term_size; ++i) {
    double const s = one_term[i];

    xsum_small sacc;
    sacc.add(one_term[i]);
    result(sacc.get(), s, i);

    xsum_large lacc;
    lacc.add(*(one_term + i));
    result(lacc.get(), s, i);
  }

  std::printf("\nC: ONE TERM TESTS TIMES %d\n", REP1);

  for (int i = 0; i < one_term_size; ++i) {
    double const s = one_term[i] * REP1;

    xsum_small_accumulator sacc;
    for (int j = 0; j < REP1; ++j) {
      xsum_add(&sacc, one_term[i]);
    }
    result(&sacc, s, i);

    xsum_large_accumulator lacc;
    for (int j = 0; j < REP1; ++j) {
      xsum_add(&lacc, one_term[i]);
    }
    result(&lacc, s, i);
  }

  for (int i = 0; i < one_term_size; ++i) {
    double const s = one_term[i] * REP1;

    xsum_small sacc;
    for (int j = 0; j < REP1; ++j) {
      sacc.add(one_term[i]);
    }
    result(sacc.get(), s, i);

    xsum_large lacc;
    for (int j = 0; j < REP1; ++j) {
      lacc.add(one_term[i]);
    }
    result(lacc.get(), s, i);
  }

  for (int i = 0; i < one_term_size; ++i) {
    double const s = one_term[i] * REP1;

    xsum_small_accumulator sacc1;
    xsum_small_accumulator sacc2;

    for (int j = 0; j < REP1 / 2; ++j) {
      xsum_add(&sacc1, one_term[i]);
    }
    for (int j = 0; j < REP1 / 2; ++j) {
      xsum_add(&sacc2, one_term[i]);
    }

    xsum_add(&sacc1, &sacc2);
    result(&sacc1, s, i);
  }

  for (int i = 0; i < one_term_size; ++i) {
    double const s = one_term[i] * REP1;

    xsum_small_accumulator sacc1;
    xsum_small_accumulator sacc2;
    xsum_small_accumulator sacc3;
    xsum_small_accumulator sacc4;

    for (int j = 0; j < REP1 / 4; ++j) {
      xsum_add(&sacc1, one_term[i]);
    }
    for (int j = 0; j < REP1 / 4; ++j) {
      xsum_add(&sacc2, one_term[i]);
    }
    for (int j = 0; j < REP1 / 4; ++j) {
      xsum_add(&sacc3, one_term[i]);
    }
    for (int j = 0; j < REP1 / 4; ++j) {
      xsum_add(&sacc4, one_term[i]);
    }

    xsum_add(&sacc1, &sacc2);
    xsum_add(&sacc1, &sacc3);
    xsum_add(&sacc1, &sacc4);
    result(&sacc1, s, i);
  }

  std::printf("\nD: TWO TERM TESTS\n");

  for (int i = 0; i < two_term_size; i += 2) {
    double const s = two_term[i] + two_term[i + 1];

    xsum_small_accumulator sacc;
    xsum_add(&sacc, two_term + i, 2);
    result(&sacc, s, i / 2);

    xsum_large_accumulator lacc;
    xsum_add(&lacc, two_term + i, 2);
    result(&lacc, s, i / 2);
  }

  for (int i = 0; i < two_term_size; i += 2) {
    double const s = two_term[i] + two_term[i + 1];

    xsum_small_accumulator sacc;
    xsum_small_accumulator sacc2;

    xsum_add(&sacc, *(two_term + i));
    xsum_add(&sacc2, *(two_term + i + 1));
    xsum_add(&sacc, &sacc2);
    result(&sacc, s, i / 2);
  }

  for (int i = 0; i < two_term_size; i += 2) {
    double const s = two_term[i] + two_term[i + 1];

    xsum_small_accumulator sacc2;
    xsum_small_accumulator sacc;

    xsum_add(&sacc2, *(two_term + i));
    xsum_add(&sacc, *(two_term + i + 1));
    xsum_add(&sacc, &sacc2);
    result(&sacc, s, i / 2);
  }

  for (int i = 0; i < two_term_size; i += 2) {
    double const s = two_term[i] + two_term[i + 1];

    xsum_small sacc;
    sacc.add(two_term + i, 2);
    result(sacc.get(), s, i / 2);

    xsum_large lacc;
    lacc.add(two_term + i, 2);
    result(lacc.get(), s, i / 2);
  }

  std::printf("\nE: THREE TERM TESTS\n");

  for (int i = 0; i < three_term_size; i += 4) {
    double const s = three_term[i + 3];

    xsum_small_accumulator sacc;
    xsum_add(&sacc, three_term + i, 3);
    result(&sacc, s, i / 4);

    xsum_large_accumulator lacc;
    xsum_add(&lacc, three_term + i, 3);
    result(&lacc, s, i / 4);
  }

  for (int i = 0; i < three_term_size; i += 4) {
    double const s = three_term[i + 3];

    xsum_small_accumulator sacc;
    xsum_small_accumulator sacc2;
    xsum_small_accumulator sacc3;
    xsum_add(&sacc, three_term[i]);
    xsum_add(&sacc2, three_term[i + 1]);
    xsum_add(&sacc3, three_term[i + 2]);
    xsum_add(&sacc, &sacc2);
    xsum_add(&sacc, &sacc3);
    result(&sacc, s, i / 4);
  }

  for (int i = 0; i < three_term_size; i += 4) {
    double const s = three_term[i + 3];

    xsum_small_accumulator sacc;
    xsum_small_accumulator sacc2[3];
    xsum_add(sacc2, three_term[i]);
    xsum_add(sacc2 + 1, three_term[i + 1]);
    xsum_add(sacc2 + 2, three_term[i + 2]);
    xsum_add(&sacc, sacc2, 3);
    result(&sacc, s, i / 4);
  }

  for (int i = 0; i < three_term_size; i += 4) {
    double const s = three_term[i + 3];

    xsum_small sacc;
    sacc.add(three_term + i, 3);
    result(sacc.get(), s, i / 4);

    xsum_large lacc;
    lacc.add(three_term + i, 3);
    result(lacc.get(), s, i / 4);
  }

  std::printf("\nF: TEN TERM TESTS\n");

  for (int i = 0; i < ten_term_size; i += 11) {
    double const s = ten_term[i + 10];

    xsum_small_accumulator sacc;
    xsum_add(&sacc, ten_term + i, 10);
    result(&sacc, s, i / 11);

    xsum_large_accumulator lacc;
    xsum_add(&lacc, ten_term + i, 10);
    result(&lacc, s, i / 11);
  }

  for (int i = 0; i < ten_term_size; i += 11) {
    double const s = ten_term[i + 10];

    xsum_small_accumulator sacc;
    xsum_small_accumulator sacc2;
    xsum_add(&sacc, ten_term + i, 5);
    xsum_add(&sacc2, ten_term + i + 5, 5);
    xsum_add(&sacc, &sacc2);
    result(&sacc, s, i / 11);
  }

  for (int i = 0; i < ten_term_size; i += 11) {
    double const s = ten_term[i + 10];

    xsum_small sacc;
    sacc.add(ten_term + i, 10);
    result(sacc.get(), s, i / 11);

    xsum_large lacc;
    lacc.add(ten_term + i, 10);
    result(lacc.get(), s, i / 11);
  }

  std::printf("\nG: TEN TERM TESTS TIMES %d\n", REP10);

  for (int i = 0; i < ten_term_size; i += 11) {
    double const s = ten_term[i + 10] * REP10;

    xsum_small_accumulator sacc;
    for (int j = 0; j < REP10; ++j) {
      xsum_add(&sacc, ten_term + i, 10);
    }
    result(&sacc, s, i / 11);

    xsum_large_accumulator lacc;
    for (int j = 0; j < REP10; ++j) {
      xsum_add(&lacc, ten_term + i, 10);
    }
    result(&lacc, s, i / 11);
  }

  for (int i = 0; i < ten_term_size; i += 11) {
    double const s = ten_term[i + 10] * REP10;

    xsum_large_accumulator lacc;
    for (int j = 0; j < REP10; ++j) {
      xsum_add(&lacc, ten_term + i, 10);
    }

    xsum_small_accumulator sacc = xsum_round_to_small(&lacc);
    result(&sacc, s, i / 11);
  }

  for (int i = 0; i < ten_term_size; i += 11) {
    double const s = ten_term[i + 10] * REP10;

    xsum_small sacc;
    for (int j = 0; j < REP10; ++j) {
      sacc.add(ten_term + i, 10);
    }
    result(sacc.get(), s, i / 11);

    xsum_large lacc;
    for (int j = 0; j < REP10; ++j) {
      lacc.add(ten_term + i, 10);
    }
    result(lacc.get(), s, i / 11);
  }

  for (int i = 0; i < ten_term_size; i += 11) {
    double const s = ten_term[i + 10] * REP10;

    xsum_large_accumulator lacc1;
    xsum_large_accumulator lacc2;

    for (int j = 0; j < REP10 / 2; ++j) {
      xsum_add(&lacc1, ten_term + i, 10);
    }
    for (int j = 0; j < REP10 / 2; ++j) {
      xsum_add(&lacc2, ten_term + i, 10);
    }

    xsum_add(&lacc1, &lacc2);

    result(&lacc1, s, i / 11);
  }

  if (small_test_fails || large_test_fails) {
    std::printf(
        "\nTotal number of tests = %d\n"
        "\t%d tests failed for small accumulator &"
        "\t%d tests failed for large accumulator\n\n",
        total_large_test + total_small_test, small_test_fails,
        large_test_fails);
  } else {
    std::printf("\n%d tests passed successfuly.\n",
                total_large_test + total_small_test);
  }
  std::printf("\nDONE\n\n");

  return 0;
}
