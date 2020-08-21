#
# Copyright (c) 2020, Regents of the University of Minnesota.
# All rights reserved.
#
# Contributors:
#    Yaser Afshar
#

# CORRECTNESS CHECKS FOR EXACT SUMMATION.

import numpy as np

try:
    from xsum import *
except:
    raise Exception('Failed to import `xsum` module')


pow2_16 = (1.0 / (1 << 16))
pow2_32 = (pow2_16 * pow2_16)
pow2_64 = (pow2_32 * pow2_32)
pow2_128 = (pow2_64 * pow2_64)
pow2_256 = (pow2_128 * pow2_128)
pow2_512 = (pow2_256 * pow2_256)
pow2_1024 = (pow2_512 * pow2_512)
pow2_52 = (1.0 / (1 << 22) / (1 << 30))

# Largest normal number
Lnormal = (2 * ((.5 / pow2_1024) - (.25 / pow2_1024) * pow2_52))
# Smallest normal number
Snormal = (4 * pow2_1024)
# Largest denormalized number
Ldenorm = (4 * pow2_1024 - 4 * pow2_1024 * pow2_52)
# Smallest denormalized number > 0
Sdenorm = (4 * pow2_1024 * pow2_52)
# Repeat factor for second set of one term tests
REP1 = (1 << 23)
REP12 = int(REP1/2)
REP14 = int(REP1/4)

# Repeat factor for second set of ten term tests
REP10 = (1 << 13)

one_term = (1.0,
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
            2 * ((.5 / pow2_128) - (.25 / pow2_128) * pow2_52),
            -2 * ((.5 / pow2_128) + (.25 / pow2_128) * pow2_52),
            Lnormal,
            -Lnormal,
            Snormal,
            -Snormal,
            Ldenorm,
            -Ldenorm,
            Sdenorm,
            -Sdenorm,
            1.23e-309,
            -1.23e-309,
            4.57e-314,
            -4.57e-314,
            9.7e-322,
            -9.7e-322,
            Sdenorm / pow2_64 / 2,
            -Sdenorm / pow2_64 / 2)

two_term = (
    (1.0, 2.0),
    (-1.0, -2.0),
    (0.1, 12.2),
    (-0.1, -12.2),
    (12.1, -11.3),
    (-12.1, 11.3),
    (11.3, -12.1),
    (-11.3, 12.1),
    (1.234567e14, 9.87654321),
    (-1.234567e14, -9.87654321),
    (1.234567e14, -9.87654321),
    (-1.234567e14, 9.87654321),
    (3.1e200, 1.7e-100),
    (3.1e200, -1.7e-100),
    (-3.1e200, 1.7e-100),
    (-3.1e200, -1.7e-100),
    (1.7e-100, 3.1e200),
    (1.7e-100, -3.1e200),
    (-1.7e-100, 3.1e200),
    (-1.7e-100, -3.1e200),
    (1, pow2_52),
    (-1, -pow2_52),
    (1, pow2_52 / 2),
    (-1, -pow2_52 / 2),
    (1, pow2_52 / 2 + pow2_52 / 4096),
    (-1, -pow2_52 / 2 - pow2_52 / 4096),
    (1, pow2_52 / 2 + pow2_52 / (1 << 30) / (1 << 10)),
    (-1, -pow2_52 / 2 - pow2_52 / (1 << 30) / (1 << 10)),
    (1, pow2_52 / 2 - pow2_52 / 4096),
    (-1, -pow2_52 / 2 + pow2_52 / 4096),
    (1 + pow2_52, pow2_52 / 2),
    (1 + pow2_52, pow2_52 / 2 - pow2_52 * pow2_52),
    (-(1 + pow2_52), -pow2_52 / 2),
    (-(1 + pow2_52), -(pow2_52 / 2 - pow2_52 * pow2_52)),
    (Sdenorm, 7.1),
    (Sdenorm, -7.1),
    (-Sdenorm, -7.1),
    (-Sdenorm, 7.1),
    (7.1, Sdenorm),
    (-7.1, Sdenorm),
    (-7.1, -Sdenorm),
    (7.1, -Sdenorm),
    (Ldenorm, Sdenorm),
    (Ldenorm, -Sdenorm),
    (-Ldenorm, Sdenorm),
    (-Ldenorm, -Sdenorm),
    (Sdenorm, Sdenorm),
    (Sdenorm, -Sdenorm),
    (-Sdenorm, Sdenorm),
    (-Sdenorm, -Sdenorm),
    (Ldenorm, Snormal),
    (Snormal, Ldenorm),
    (-Ldenorm, -Snormal),
    (-Snormal, -Ldenorm),
    (4.57e-314, 9.7e-322),
    (-4.57e-314, 9.7e-322),
    (4.57e-314, -9.7e-322),
    (-4.57e-314, -9.7e-322),
    (4.57e-321, 9.7e-322),
    (-4.57e-321, 9.7e-322),
    (4.57e-321, -9.7e-322),
    (-4.57e-321, -9.7e-322),
    (2.0, -2.0 * (1 + pow2_52)),
    (Lnormal, Lnormal),
    (-Lnormal, -Lnormal),
    (Lnormal, Lnormal * pow2_52 / 2),
    (-Lnormal, -Lnormal * pow2_52 / 2),
    (np.inf, 123),
    (-np.inf, 123),
    (np.inf, -np.inf),
    (np.nan, 123),
    (123, np.nan))


three_term = (
    (Lnormal, Sdenorm, -Lnormal, Sdenorm),
    (-Lnormal, Sdenorm, Lnormal, Sdenorm),
    (Lnormal, -Sdenorm, -Lnormal, -Sdenorm),
    (-Lnormal, -Sdenorm, Lnormal, -Sdenorm),
    (Sdenorm, Snormal, -Sdenorm, Snormal),
    (-Sdenorm, -Snormal, Sdenorm, -Snormal),
    (12345.6, Snormal, -12345.6, Snormal),
    (12345.6, -Snormal, -12345.6, -Snormal),
    (12345.6, Ldenorm, -12345.6, Ldenorm),
    (12345.6, -Ldenorm, -12345.6, -Ldenorm),
    (2.0, -2.0 * (1 + pow2_52), pow2_52 / 8, -2 * pow2_52 + pow2_52 / 8),
    (1.0, 2.0, 3.0, 6.0),
    (12.0, 3.5, 2.0, 17.5),
    (3423.34e12, -93.431, -3432.1e11, 3080129999999906.5),
    (432457232.34, 0.3432445, -3433452433, -3000995200.3167553))

ten_term = (
    (Lnormal, Lnormal, Lnormal, Lnormal, Lnormal, Lnormal, -
     Lnormal, -Lnormal, -Lnormal, -Lnormal, np.inf),
    (-Lnormal, -Lnormal, -Lnormal, -Lnormal, -Lnormal, -
     Lnormal, Lnormal, Lnormal, Lnormal, Lnormal, -np.inf),
    (Lnormal, Lnormal, Lnormal, Lnormal, 0.125, 0.125, -
     Lnormal, -Lnormal, -Lnormal, -Lnormal, 0.25),
    (2.0 * (1 + pow2_52), -2.0, -pow2_52, -pow2_52, 0, 0, 0, 0, 0, 0, 0),
    (1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1111111111e0),
    (-1e0, -1e1, -1e2, -1e3, -1e4, -1e5, -1e6, -1e7, -1e8, -1e9, -1111111111e0),
    (1.234e88, -93.3e-23, 994.33, 1334.3, 457.34, -
     1.234e88, 93.3e-23, -994.33, -1334.3, -457.34, 0),
    (1., -23., 456., -78910., 1112131415., -161718192021., 22232425262728., -
     2930313233343536., 373839404142434445., -46474849505152535455., -46103918342424313856.),
    (2342423.3423, 34234.450, 945543.4, 34345.34343, 1232.343, 0.00004343,
     43423.0, -342344.8343, -89544.3435, -34334.3, 2934978.4009734304),
    (0.9101534, 0.9048397, 0.4036596, 0.1460245, 0.2931254, 0.9647649,
     0.1125303, 0.1574193, 0.6522300, 0.7378597, 5.2826068),
    (428.366070546, 707.3261930632, 103.29267289, 9040.03475821, 36.2121638, 19.307901408,
     1.4810709160, 8.077159101, 1218.907244150, 778.068267017, 12341.0735011012),
    (1.1e-322, 5.3443e-321, -9.343e-320, 3.33e-314, 4.41e-322, -8.8e-318, 3.1e-310, 4.1e-300, -4e-300, 7e-307, 1.0000070031003328e-301))


total_test = 0
test_fails = 0


def different(a, b):
    return (np.isnan(a) != np.isnan(b)) or ((not np.isnan(a)) and (not np.isnan(b)) and (a != b))


def result(acc, s, i):
    global total_test
    global test_fails

    total_test += 1

    if isinstance(acc, xsum_small_accumulator) or isinstance(acc, xsum_large_accumulator):
        r = xsum_round(acc)
        r2 = xsum_round(acc)
    elif isinstance(acc, xsum_small) or isinstance(acc, xsum_large):
        r = acc.round()
        r2 = acc.round()
    else:
        print("-- TEST {}".format(i))
        test_fails += 1
        print("Wrong input!")
        return False

    if (different(r, r2)):
        test_fails += 1
        print("-- TEST {}".format(i))
        print("   ANSWER: {}".format(s))
        print("   Different second time {} != {}".format(r, r2))
        return False

    if (different(r, s)):
        test_fails += 1
        print("-- TEST {}".format(i))
        print("   ANSWER: {}".format(s))
        print("   Result incorrect {} != {}".format(r, s))
        print()
        pbinary(r)
        print()
        pbinary(s)
        return False

    return True


"""Test exact summation module components."""


def zero_term_test():
    print("A: ZERO TERM TEST")

    sacc = xsum_small_accumulator()
    lacc = xsum_large_accumulator()

    assert result(sacc, 0, 0) == True
    assert result(lacc, 0, 0) == True

    s = xsum_small()
    l = xsum_large()

    assert result(s, 0, 0) == True
    assert result(l, 0, 0) == True


def one_term_test():
    print("B: ONE TERM TESTS")

    for i, s in enumerate(one_term):
        sacc = xsum_small_accumulator()
        xsum_add(sacc, s)
        assert result(sacc, s, i) == True

        lacc = xsum_large_accumulator()
        xsum_add(lacc, s)
        assert result(lacc, s, i) == True

        sacc = xsum_small()
        sacc.add(s)
        assert result(sacc, s, i) == True

        lacc = xsum_large()
        lacc.add(s)
        assert result(lacc, s, i) == True


def one_term_test_times():
    print("C: ONE TERM TESTS TIMES {}".format(REP1))

    for i, _s in enumerate(one_term):
        s = _s * REP1
        a = np.ones(REP1) * _s

        sacc = xsum_small_accumulator()
        xsum_add(sacc, a)
        assert result(sacc, s, i) == True

        lacc = xsum_large_accumulator()
        xsum_add(lacc, a)
        assert result(lacc, s, i) == True

    for i, _s in enumerate(one_term):
        s = _s * REP1
        a = np.ones(REP1) * _s

        sacc = xsum_small()
        sacc.add(a)
        assert result(sacc, s, i) == True

        lacc = xsum_large()
        lacc.add(a)
        assert result(lacc, s, i) == True

    for i, _s in enumerate(one_term):
        s = _s * REP1
        a = np.ones(REP12) * _s

        sacc1 = xsum_small_accumulator()
        sacc2 = xsum_small_accumulator()

        xsum_add(sacc1, a)
        xsum_add(sacc2, a)
        xsum_add(sacc1, sacc2)

        assert result(sacc1, s, i) == True

    for i, _s in enumerate(one_term):
        s = _s * REP1
        b = np.ones(REP14) * _s

        sacc1 = xsum_small_accumulator()
        sacc2 = xsum_small_accumulator()
        sacc3 = xsum_small_accumulator()
        sacc4 = xsum_small_accumulator()

        xsum_add(sacc1, b)
        xsum_add(sacc2, b)
        xsum_add(sacc3, b)
        xsum_add(sacc4, b)
        xsum_add(sacc1, sacc2)
        xsum_add(sacc1, sacc3)
        xsum_add(sacc1, sacc4)

        assert result(sacc1, s, i) == True


def two_term_test():
    print("D: TWO TERM TESTS")

    for i, _s in enumerate(two_term):
        s = sum(_s)

        sacc = xsum_small_accumulator()
        xsum_add(sacc, s)
        assert result(sacc, s, int(i/2)) == True

        lacc = xsum_large_accumulator()
        xsum_add(lacc, s)
        assert result(lacc, s, int(i/2)) == True

    for i, _s in enumerate(two_term):
        s = sum(_s)

        sacc = xsum_small_accumulator()
        sacc1 = xsum_small_accumulator()
        xsum_add(sacc, _s[0])
        xsum_add(sacc1, _s[1])
        xsum_add(sacc, sacc1)
        assert result(sacc, s, int(i/2)) == True

        lacc = xsum_large_accumulator()
        lacc1 = xsum_large_accumulator()
        xsum_add(lacc, _s[0])
        xsum_add(lacc1, _s[1])
        xsum_add(lacc, lacc1)
        assert result(lacc, s, int(i/2)) == True

        lacc2 = xsum_large_accumulator()
        sacc2 = xsum_small_accumulator()
        xsum_add(lacc2, _s[0])
        xsum_add(sacc2, _s[1])
        xsum_add(lacc2, sacc2)
        assert result(lacc, s, int(i/2)) == True


def three_term_test():
    print("E: THREE TERM TESTS")

    for i, _s in enumerate(three_term):
        s = _s[3]

        sacc = xsum_small_accumulator()
        xsum_add(sacc, _s[:-1])
        assert result(sacc, s, int(i/4)) == True

        lacc = xsum_large_accumulator()
        xsum_add(lacc, _s[:-1])
        assert result(lacc, s, int(i/4)) == True

    for i, _s in enumerate(three_term):
        s = _s[3]

        sacc = xsum_small()
        sacc.add(_s[:-1])
        assert result(sacc, s, int(i/4)) == True

        lacc = xsum_large()
        lacc.add(_s[:-1])
        assert result(lacc, s, int(i/4)) == True

    for i, _s in enumerate(three_term):
        s = _s[3]

        sacc1 = xsum_small_accumulator()
        sacc2 = xsum_small_accumulator()
        sacc3 = xsum_small_accumulator()

        xsum_add(sacc1, _s[0])
        xsum_add(sacc2, _s[1])
        xsum_add(sacc3, _s[2])

        xsum_add(sacc1, sacc2)
        xsum_add(sacc1, sacc3)

        assert result(sacc1, s, int(i/4)) == True


def ten_term_test():
    print("F: TEN TERM TESTS")

    for i, _s in enumerate(ten_term):
        s = _s[10]

        sacc = xsum_small_accumulator()
        xsum_add(sacc, _s[:-1])
        assert result(sacc, s, int(i/11)) == True

        lacc = xsum_large_accumulator()
        xsum_add(lacc, _s[:-1])
        assert result(lacc, s, int(i/11)) == True

    for i, _s in enumerate(ten_term):
        s = _s[10]

        sacc1 = xsum_small_accumulator()
        sacc2 = xsum_small_accumulator()
        xsum_add(sacc1, _s[:5])
        xsum_add(sacc2, _s[5:-1])
        xsum_add(sacc1, sacc2)
        assert result(sacc1, s, int(i/11)) == True


def ten_term_test_times():
    print("G: TEN TERM TESTS TIMES {}".format(REP10))

    for i, _s in enumerate(ten_term):
        s = _s[10] * REP10
        a = np.tile(_s[:-1], REP10)

        sacc = xsum_small_accumulator()
        xsum_add(sacc, a)
        assert result(sacc, s, i) == True

        lacc = xsum_large_accumulator()
        xsum_add(lacc, a)
        assert result(lacc, s, i) == True

    for i, _s in enumerate(ten_term):
        s = _s[10] * REP10
        a = np.tile(_s[:-1], REP10)

        sacc = xsum_small()
        sacc.add(a)
        assert result(sacc, s, i) == True

        lacc = xsum_large()
        lacc.add(a)
        assert result(lacc, s, i) == True


if __name__ == '__main__':
    zero_term_test()
    one_term_test()
    one_term_test_times()
    two_term_test()
    three_term_test()
    ten_term_test()
    ten_term_test_times()
    print("\nTotal number of tests = {}".format(total_test))
    print("{:8} tests failed".format(test_fails))
    print("{:8} tests passed successfuly.".format(total_test - test_fails))
    print("\nDONE\n")
