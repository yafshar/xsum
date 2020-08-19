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
// XSUM.hpp
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
// Brief: The current implementation contains the methods described in the
//        paper "Fast Exact Summation Using Small and Large Superaccumulators",
//        , by Radford M. Neal, available at https://arxiv.org/abs/1505.05571,
//        and implemented by Radford M. Neal at
//        https://gitlab.com/radfordneal/xsum.git
//
//        It is adapted, rewritten and amended in C++ from the original work of
//        R. M. Neal by Yaser Afshar.
//        It has been rewritten with the goal of ease of use and C++ portability
//        and it also includes extra classes and functionalities especially for
//        the use in High Performance Message Passing Libraries. Where an `op`
//        handle can subsequently be used in `MPI_Reduce`,`MPI_Allreduce`,
//        `MPI_Reduce_scatter`, and `MPI_Scan` calls.
//
// Note:  To see, use or reproduce the results of the original implementation
//        reported in the paper `Fast Exact Summation Using Small and Large
//        Superaccumulators`, by Radford M. Neal, please refer to
//        https://gitlab.com/radfordneal/xsum.
//

#ifndef XSUM_HPP
#define XSUM_HPP

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

/* CONSTANTS DEFINING THE FLOATING POINT FORMAT. */

/*! C floating point type sums are done for */
using xsum_flt = double;
/*! Signed integer type for a fp value */
using xsum_int = std::int64_t;
/*! Unsigned integer type for a fp value */
using xsum_uint = std::uint64_t;
/*! Integer type for holding an exponent */
using xsum_expint = std::int_fast16_t;
/*! TYPE FOR LENGTHS OF ARRAYS.  Must be a signed integer type. */
using xsum_length = int;
/*! Integer type of small accumulator chunk */
using xsum_schunk = std::int64_t;
/*! Integer type of large accumulator chunk, must be EXACTLY 64 bits in size */
using xsum_lchunk = std::uint64_t;
/*! Signed int type of counts for large acc.*/
using xsum_lcount = std::int_least16_t;
/*! Unsigned type for holding used flags */
using xsum_used = std::uint_fast64_t;

/*! Bits in fp mantissa, excludes implict 1 */
constexpr int XSUM_MANTISSA_BITS = 52;
/*! Bits in fp exponent */
constexpr int XSUM_EXP_BITS = 11;
/*! Mask for mantissa bits */
constexpr xsum_int XSUM_MANTISSA_MASK =
    ((static_cast<xsum_int>(1) << XSUM_MANTISSA_BITS) -
     static_cast<xsum_int>(1));
/*! Mask for exponent */
constexpr int XSUM_EXP_MASK = ((1 << XSUM_EXP_BITS) - 1);
/*! Bias added to signed exponent */
constexpr int XSUM_EXP_BIAS = ((1 << (XSUM_EXP_BITS - 1)) - 1);
/*! Position of sign bit */
constexpr int XSUM_SIGN_BIT = (XSUM_MANTISSA_BITS + XSUM_EXP_BITS);
/*! Mask for sign bit */
constexpr xsum_uint XSUM_SIGN_MASK =
    (static_cast<xsum_uint>(1) << XSUM_SIGN_BIT);

/* CONSTANTS DEFINING THE SMALL ACCUMULATOR FORMAT. */

/*! Bits in chunk of the small accumulator */
constexpr int XSUM_SCHUNK_BITS = 64;
/*! # of low bits of exponent, in one chunk */
constexpr int XSUM_LOW_EXP_BITS = 5;
/*! Mask for low-order exponent bits */
constexpr int XSUM_LOW_EXP_MASK = ((1 << XSUM_LOW_EXP_BITS) - 1);
/*! # of high exponent bits for index */
constexpr int XSUM_HIGH_EXP_BITS = (XSUM_EXP_BITS - XSUM_LOW_EXP_BITS);
/*! Mask for high-order exponent bits */
constexpr int XSUM_HIGH_EXP_MASK = ((1 << XSUM_HIGH_EXP_BITS) - 1);
/*! # of chunks in small accumulator */
constexpr int XSUM_SCHUNKS = ((1 << XSUM_HIGH_EXP_BITS) + 3);
/*! Bits in low part of mantissa */
constexpr int XSUM_LOW_MANTISSA_BITS = (1 << XSUM_LOW_EXP_BITS);
/*! Bits in high part */
constexpr int XSUM_HIGH_MANTISSA_BITS =
    (XSUM_MANTISSA_BITS - XSUM_LOW_MANTISSA_BITS);
/*! Mask for low bits */
constexpr xsum_int XSUM_LOW_MANTISSA_MASK =
    ((static_cast<xsum_int>(1) << XSUM_LOW_MANTISSA_BITS) -
     static_cast<xsum_int>(1));
/*! Bits sums can carry into */
constexpr int XSUM_SMALL_CARRY_BITS =
    ((XSUM_SCHUNK_BITS - 1) - XSUM_MANTISSA_BITS);
/*! # terms can add before need prop. */
constexpr int XSUM_SMALL_CARRY_TERMS = ((1 << XSUM_SMALL_CARRY_BITS) - 1);

/* CONSTANTS DEFINING THE LARGE ACCUMULATOR FORMAT. */

/*! Bits in chunk of the large accumulator */
constexpr int XSUM_LCHUNK_BITS = 64;
/*! # of bits in count */
constexpr int XSUM_LCOUNT_BITS = (64 - XSUM_MANTISSA_BITS);
/*! # of chunks in large accumulator */
constexpr int XSUM_LCHUNKS = (1 << (XSUM_EXP_BITS + 1));

/*! DEBUG FLAG.  Set to non-zero for debug ouptut.  Ignored unless xsum.c is
 * compiled with -DDEBUG. */
constexpr int xsum_debug = 0;

/*! UNION OF FLOATING AND INTEGER TYPES. */
union fpunion {
  xsum_flt fltv;
  xsum_int intv;
  xsum_uint uintv;
};

/*! CLASSES FOR EXACT SUMMATION. */

/*!
 * \brief Small super accumulator
 *
 */
struct xsum_small_accumulator {
  /*! Chunks making up small accumulator */
  xsum_schunk chunk[XSUM_SCHUNKS] = {};
  /*! If non-zero, +Inf, -Inf, or NaN */
  xsum_int Inf = 0;
  /*! If non-zero, a NaN value with payload */
  xsum_int NaN = 0;
  /*! Number of remaining adds before carry */
  int adds_until_propagate = XSUM_SMALL_CARRY_TERMS;
};

/*!
 * \brief Large super accumulator
 *
 */
struct xsum_large_accumulator {
  xsum_large_accumulator();

  /*! Chunks making up large accumulator */
  xsum_lchunk chunk[XSUM_LCHUNKS];
  /*! Counts of # adds remaining for chunks, or -1 if not used yet or special.
   */
  xsum_lcount count[XSUM_LCHUNKS];
  /*! Bits indicate chunks in use */
  xsum_used chunks_used[XSUM_LCHUNKS / 64] = {};
  /*! Bits indicate chunk_used entries not 0 */
  xsum_used used_used = 0;
  /*! The small accumulator to condense into */
  xsum_small_accumulator sacc;
};

/*!
 * \brief Small superaccumulator class
 *
 */
class xsum_small {
 public:
  /*!
   * \brief Construct a new xsum small object
   *
   */
  xsum_small();
  explicit xsum_small(xsum_small_accumulator const &sacc);
  explicit xsum_small(xsum_small_accumulator const *sacc);

  /*!
   * \brief Replaces the xsum_small_accumulator object
   *
   */
  void reset();

  /*!
   * \brief Initilize the xsum_small_accumulator object
   *
   */
  void init();

  /*!
   * \brief Add one double value to a superaccumulator.
   *
   * \param value
   */
  void add(xsum_flt const value);
  void add(xsum_small_accumulator const &value);
  void add(xsum_small_accumulator const *value);
  void add(xsum_small const &value);

  /*!
   * \brief Add a vector of double numbers to a superaccumulator.
   *
   * Mixes calls of carry_propagate with calls of add_no_carry to add parts
   * that are small enough that no carry will result. Note that
   * xsum_add_no_carry may pre-fetch one beyond the last value it sums, so to
   * be safe, adding the last value has to be done separately at the end.
   *
   * \param vec
   * \param n
   */
  void add(xsum_flt const *vec, xsum_length const n);
  void add(std::vector<xsum_flt> const &vec);

  /*!
   * \brief Add squared norm of vector of double numbers to a superaccumulator.
   *
   * Mixes calls of carry_propagate with calls of add_sqnorm_no_carry to add
   * parts that are small enough that no carry will result. Note that
   * add_sqnorm_no_carry may pre-fetch one beyond the last value it sums, so
   * to be safe, adding the last value has to be done separately at
   *
   * \param vec vector of double numbers
   * \param n vector length
   */
  void add_sqnorm(xsum_flt const *vec, xsum_length const n);
  void add_sqnorm(std::vector<xsum_flt> const &vec);

  /*!
   * \brief Add dot product of vectors of double numbers to a superaccumulator.
   *
   * Mixes calls of carry_propagate with calls of add_dot_no_carry to add
   * parts that are small enough that no carry will result. Note that
   * add_dot_no_carry may pre-fetch one beyond the last value it sums, so to
   * be safe, adding the last value has to be done separately at the end.
   *
   * \param vec1 vector of double numbers
   * \param vec2 vector of double numbers
   * \param n vector length
   */
  void add_dot(xsum_flt const *vec1, xsum_flt const *vec2, xsum_length const n);
  void add_dot(std::vector<xsum_flt> const &vec1,
               std::vector<xsum_flt> const &vec2);

  /*!
   * \brief Return the results of rounding a superaccumulator.
   *
   * The rounding mode is to nearest, with ties to even. The superaccumulator
   *  may be modified by this operation (by carry propagation being done), but
   * the value it represents should not change.
   *
   * \return xsum_flt
   */
  xsum_flt round();

  /*!
   * \brief Display a superaccumulator.
   *
   */
  void display();

  /* Return number of chunks in use in small accumulator. */
  int chunks_used();

  /*!
   * \brief Returns a pointer to the xsum_small_accumulator object
   *
   * \return xsum_small_accumulator*
   */
  inline xsum_small_accumulator *get() const noexcept;

  /*!
   * PROPAGATE CARRIES TO NEXT CHUNK IN A SMALL ACCUMULATOR.  Needs to
   * be called often enough that accumulated carries don't overflow out
   * the top, as indicated by sacc->adds_until_propagate.  Returns the
   * index of the uppermost non-zero chunk (0 if number is zero).
   * After carry propagation, the uppermost non-zero chunk will indicate
   * the sign of the number, and will not be -1 (all 1s).  It will be in
   * the range -2^XSUM_LOW_MANTISSA_BITS to 2^XSUM_LOW_MANTISSA_BITS - 1.
   * Lower chunks will be non-negative, and in the range from 0 up to
   * 2^XSUM_LOW_MANTISSA_BITS - 1.
   *
   * Set u to the index of the uppermost non-zero (for now) chunk, or
   * return with value 0 if there is none.
   */
  int carry_propagate();

  /*
   * ADD AN INF OR NAN TO A SMALL ACCUMULATOR.  This only changes the flags,
   * not the chunks in the accumulator, which retains the sum of the finite
   * terms (which is perhaps sometimes useful to access, though no function
   * to do so is defined at present).  A NaN with larger payload (seen as a
   * 52-bit unsigned integer) takes precedence, with the sign of the NaN always
   * being positive.  This ensures that the order of summing NaN values doesn't
   * matter.
   */
  void add_inf_nan(xsum_int const ivalue);

 private:
  /*
   * ADD ONE NUMBER TO A SMALL ACCUMULATOR ASSUMING NO CARRY PROPAGATION REQ'D.
   * This function is declared "inline" for good performance it must be inlined
   * by the compiler (otherwise the procedure call overhead will result in
   * substantial inefficiency).
   */
  inline void add_no_carry(xsum_flt const value);
  inline void add_no_carry(xsum_small_accumulator const *value);

  /*
   * ADD A VECTOR TO A SMALL ACCUMULATOR, ASSUMING NO CARRY PROPAGATION NEEDED.
   * Adds n-1 numbers from vec, which must have at least n elements; n must
   * be at least 1.  This odd specificiation is designed so that in the OPT
   * version we can pre-fetch the next value to allow some time for memory
   * response before the value is used.
   */
  inline void add_no_carry(xsum_flt const *vec, xsum_length const n);

  /*
   * ADD SQUARED NORM OF VECTOR TO SMALL ACCUMULATOR, ASSUME NO CARRY NEEDED.
   * Adds n-1 squares of numbers from vec, which must have at least n elements;
   * n must be at least 1.  This odd specificiation is designed so that in the
   * OPT version we can pre-fetch the next value to allow some time for memory
   * response before the value is used.
   */
  inline void add_sqnorm_no_carry(xsum_flt const *vec, xsum_length const n);

  /*
   * ADD DOT PRODUCT OF VECTORS TO SMALL ACCUMULATOR, ASSUME NO CARRY NEEDED.
   * Adds n-1 products of numbers from vec1 and vec2, which must have at least
   * n elements; n must be at least 1.  This odd specificiation is designed so
   * that in the OPT version we can pre-fetch the next values to allow some time
   * for memory response before the value is used.
   */
  inline void add_dot_no_carry(xsum_flt const *vec1, xsum_flt const *vec2,
                               xsum_length const n);

 private:
  std::unique_ptr<xsum_small_accumulator> _sacc;
};

/*!
 * \brief Large accumulator
 *
 */
class xsum_large {
 public:
  /*!
   * \brief Construct a new xsum large object
   *
   */
  xsum_large();
  explicit xsum_large(xsum_large_accumulator const &lacc);
  explicit xsum_large(xsum_large_accumulator const *lacc);
  explicit xsum_large(xsum_small_accumulator const &sacc);
  explicit xsum_large(xsum_small_accumulator const *sacc);
  explicit xsum_large(xsum_small const &sacc);
  explicit xsum_large(xsum_small const *sacc);

  /*!
   * \brief Replaces the xsum_large_accumulator object
   *
   */
  void reset();

  /*!
   * \brief Initilize the xsum_large_accumulator object
   *
   */
  void init();

  /* ADD SINGLE NUMBER TO THE LARGE ACCUMULATOR */
  void add(xsum_flt const value);
  void add(xsum_small_accumulator const *const value);
  void add(xsum_large_accumulator *const value);

  /*
   * ADD A VECTOR OF FLOATING-POINT NUMBERS TO A LARGE ACCUMULATOR.
   */
  void add(xsum_flt const *vec, xsum_length const n);
  void add(std::vector<xsum_flt> const &vec);

  /* ADD SQUARED NORM OF VECTOR OF FLOATING-POINT NUMBERS TO LARGE ACCUMULATOR.
   */
  void add_sqnorm(xsum_flt const *vec, xsum_length const n);
  void add_sqnorm(std::vector<xsum_flt> const &vec);

  /* ADD DOT PRODUCT OF VECTORS OF FLOATING-POINT NUMBERS TO LARGE ACCUMULATOR.
   */
  void add_dot(xsum_flt const *vec1, xsum_flt const *vec2, xsum_length const n);
  void add_dot(std::vector<xsum_flt> const &vec1,
               std::vector<xsum_flt> const &vec2);

  /*
   * RETURN THE RESULT OF ROUNDING A SMALL ACCUMULATOR.  The rounding mode
   * is to nearest, with ties to even.  The small accumulator may be modified
   * by this operation (by carry propagation being done), but the value it
   * represents should not change.
   */
  xsum_flt round();

  xsum_small_accumulator *round_to_small();
  xsum_small_accumulator *round_to_small(xsum_large_accumulator *const lacc);

  /* Display a large accumulator. */
  void display();

  /* Return number of chunks in use in large accumulator. */
  int chunks_used();

  /*!
   * \brief Returns a pointer to the xsum_large_accumulator object
   *
   * \return xsum_large_accumulator*
   */
  inline xsum_large_accumulator *get() const noexcept;

 private:
  /*!
   * PROPAGATE CARRIES TO NEXT CHUNK IN A SMALL ACCUMULATOR.  Needs to
   * be called often enough that accumulated carries don't overflow out
   * the top, as indicated by sacc->adds_until_propagate.  Returns the
   * index of the uppermost non-zero chunk (0 if number is zero).
   * After carry propagation, the uppermost non-zero chunk will indicate
   * the sign of the number, and will not be -1 (all 1s).  It will be in
   * the range -2^XSUM_LOW_MANTISSA_BITS to 2^XSUM_LOW_MANTISSA_BITS - 1.
   * Lower chunks will be non-negative, and in the range from 0 up to
   * 2^XSUM_LOW_MANTISSA_BITS - 1.
   *
   * Set u to the index of the uppermost non-zero (for now) chunk, or
   * return with value 0 if there is none.
   */
  int carry_propagate();

  inline void add_no_carry(xsum_small_accumulator const *const value);

  /*
   * RETURN THE RESULT OF ROUNDING A SMALL ACCUMULATOR.  The rounding mode
   * is to nearest, with ties to even.  The small accumulator may be modified
   * by this operation (by carry propagation being done), but the value it
   * represents should not change.
   */
  xsum_flt sround();

  /* Display a small accumulator. */
  void sdisplay();

  /*
   * ADD CHUNK FROM A LARGE ACCUMULATOR TO THE SMALL ACCUMULATOR WITHIN IT.
   * The large accumulator chunk to add is indexed by ix.  This chunk will
   * be cleared to zero and its count reset after it has been added to the
   * small accumulator (except no add is done for a new chunk being
   * initialized). This procedure should not be called for the special chunks
   * correspnding to Inf or NaN, whose counts should always remain at -1.
   */
  void add_lchunk_to_small(xsum_expint const ix);

  /*
   * ADD A CHUNK TO THE LARGE ACCUMULATOR OR PROCESS NAN OR INF.  This routine
   * is called when the count for a chunk is negative after decrementing, which
   * indicates either inf/nan, or that the chunk has not been initialized, or
   * that the chunk needs to be transferred to the small accumulator.
   */
  inline void add_value_inf_nan(xsum_expint const ix, xsum_lchunk const uintv);

  /*
   * ADD AN INF OR NAN TO A SMALL ACCUMULATOR.  This only changes the flags,
   * not the chunks in the accumulator, which retains the sum of the finite
   * terms (which is perhaps sometimes useful to access, though no function
   * to do so is defined at present).  A NaN with larger payload (seen as a
   * 52-bit unsigned integer) takes precedence, with the sign of the NaN always
   * being positive.  This ensures that the order of summing NaN values doesn't
   * matter.
   */
  void add_inf_nan(xsum_int const ivalue);

 private:
  std::unique_ptr<xsum_large_accumulator> _lacc;
};

/* EXACT SUM FUNCTIONS */
static int xsum_carry_propagate(xsum_small_accumulator *const sacc);
static inline void xsum_small_add_inf_nan(xsum_small_accumulator *const sacc,
                                          xsum_int const ivalue);
static inline void xsum_add_no_carry(xsum_small_accumulator *const sacc,
                                     xsum_flt const value);
static inline void xsum_add_no_carry(xsum_small_accumulator *const sacc,
                                     xsum_small_accumulator const *const value);
static inline void xsum_add_no_carry(xsum_small_accumulator *const sacc,
                                     xsum_flt const *const vec,
                                     xsum_length const n);
static inline void xsum_add_no_carry(xsum_small_accumulator *const sacc,
                                     xsum_small_accumulator const *const vec,
                                     xsum_length const n);
static inline void xsum_add_sqnorm_no_carry(xsum_small_accumulator *const sacc,
                                            xsum_flt const *const vec,
                                            xsum_length const n);
static inline void xsum_add_dot_no_carry(xsum_small_accumulator *const sacc,
                                         xsum_flt const *const vec1,
                                         xsum_flt const *const vec2,
                                         xsum_length const n);
static inline void xsum_add_lchunk_to_small(xsum_large_accumulator *const lacc,
                                            xsum_expint const ix);
static inline void xsum_large_add_value_inf_nan(
    xsum_large_accumulator *const lacc, xsum_expint const ix,
    xsum_lchunk const uintv);

template <typename T>
void xsum_init(T *const sacc);
template <typename T>
void xsum_add(T *const acc, xsum_flt const value);
template <typename T>
void xsum_add(T *const acc, xsum_flt const *const vec, xsum_length const n);
template <typename T>
void xsum_add(T *const acc, std::vector<xsum_flt> const &vec);
void xsum_add(xsum_small_accumulator *const acc,
              xsum_small_accumulator const *const value);
void xsum_add(xsum_large_accumulator *const acc,
              xsum_large_accumulator *const value);
void xsum_add(xsum_small_accumulator *const sacc,
              xsum_small_accumulator const *const vec, xsum_length const n);
void xsum_add(xsum_large_accumulator *const acc,
              xsum_small_accumulator const *const value);
template <typename T>
void xsum_add_sqnorm(T *const acc, xsum_flt const *const vec,
                     xsum_length const n);
template <typename T>
void xsum_add_sqnorm(T *const acc, std::vector<xsum_flt> const &vec);
template <typename T>
void xsum_add_dot(T *const acc, xsum_flt const *const vec1,
                  xsum_flt const *const vec2, xsum_length const n);
template <typename T>
void xsum_add_dot(T *const acc, std::vector<xsum_flt> const &vec1,
                  std::vector<xsum_flt> const &vec2);
template <typename T>
xsum_flt xsum_round(T *const acc);
xsum_small_accumulator *xsum_round_to_small(xsum_large_accumulator *const lacc);
static void pbinary_double(double const d);

// Implementation

xsum_large_accumulator::xsum_large_accumulator() {
  std::fill(count, count + XSUM_LCHUNKS, -1);
}

/* INITIALIZE A SMALL ACCUMULATOR TO ZERO. */

xsum_small::xsum_small() : _sacc(new xsum_small_accumulator) {}

xsum_small::xsum_small(xsum_small_accumulator const &sacc)
    : _sacc(new xsum_small_accumulator) {
  std::copy(sacc.chunk, sacc.chunk + XSUM_SCHUNKS, _sacc->chunk);
  _sacc->Inf = sacc.Inf;
  _sacc->NaN = sacc.NaN;
  _sacc->adds_until_propagate = sacc.adds_until_propagate;
}

xsum_small::xsum_small(xsum_small_accumulator const *sacc)
    : _sacc(new xsum_small_accumulator) {
  if (sacc) {
    std::copy(sacc->chunk, sacc->chunk + XSUM_SCHUNKS, _sacc->chunk);
    _sacc->Inf = sacc->Inf;
    _sacc->NaN = sacc->NaN;
    _sacc->adds_until_propagate = sacc->adds_until_propagate;
  }
}

void xsum_small::reset() { _sacc.reset(new xsum_small_accumulator); }

void xsum_small::init() {
  std::fill(_sacc->chunk, _sacc->chunk + XSUM_SCHUNKS, 0);
  _sacc->Inf = 0;
  _sacc->NaN = 0;
  _sacc->adds_until_propagate = XSUM_SMALL_CARRY_TERMS;
}

void xsum_small::add(xsum_flt const value) {
  if (_sacc->adds_until_propagate == 0) {
    carry_propagate();
  }

  add_no_carry(value);

  --_sacc->adds_until_propagate;
}

void xsum_small::add(xsum_small_accumulator const &value) {
  if (_sacc->adds_until_propagate == 0) {
    carry_propagate();
  }

  add_no_carry(&value);

  --_sacc->adds_until_propagate;
}

void xsum_small::add(xsum_small_accumulator const *value) {
  if (_sacc->adds_until_propagate == 0) {
    carry_propagate();
  }

  add_no_carry(value);

  --_sacc->adds_until_propagate;
}

void xsum_small::add(xsum_small const &xvalue) {
  xsum_small_accumulator const *value = xvalue.get();

  if (_sacc->adds_until_propagate == 0) {
    carry_propagate();
  }

  add_no_carry(value);

  --_sacc->adds_until_propagate;
}

void xsum_small::add(xsum_flt const *v, xsum_length const n) {
  if (n == 0) {
    return;
  }

  xsum_small_accumulator *sacc = _sacc.get();

  xsum_flt const *vec = v;
  xsum_length c = n;
  while (c > 1) {
    if (sacc->adds_until_propagate == 0) {
      carry_propagate();
    }

    xsum_length const m =
        c - ((1 <= sacc->adds_until_propagate) ? c - 1
                                               : sacc->adds_until_propagate);

    add_no_carry(vec, m + 1);

    sacc->adds_until_propagate -= m;

    vec += m;
    c -= m;
  }

  xsum_flt const f = *vec;
  add(f);
}

void xsum_small::add(std::vector<xsum_flt> const &v) {
  xsum_length c = static_cast<xsum_length>(v.size());
  if (c == 0) {
    return;
  }

  xsum_small_accumulator *sacc = _sacc.get();

  xsum_flt const *vec = v.data();

  while (c > 1) {
    if (sacc->adds_until_propagate == 0) {
      carry_propagate();
    }

    xsum_length const m =
        c - ((1 <= sacc->adds_until_propagate) ? c - 1
                                               : sacc->adds_until_propagate);

    add_no_carry(vec, m + 1);

    sacc->adds_until_propagate -= m;

    vec += m;
    c -= m;
  }

  xsum_flt const f = *vec;
  add(f);
}

void xsum_small::add_sqnorm(xsum_flt const *v, xsum_length const n) {
  if (n == 0) {
    return;
  }

  xsum_small_accumulator *sacc = _sacc.get();

  xsum_flt const *vec = v;
  xsum_length c = n;
  while (c > 1) {
    if (sacc->adds_until_propagate == 0) {
      carry_propagate();
    }

    xsum_length const m =
        c - ((1 <= sacc->adds_until_propagate) ? c - 1
                                               : sacc->adds_until_propagate);

    add_sqnorm_no_carry(vec, m + 1);

    sacc->adds_until_propagate -= m;

    vec += m;
    c -= m;
  }

  xsum_flt const f = *vec;
  xsum_flt const g = f * f;
  add(g);
}

void xsum_small::add_sqnorm(std::vector<xsum_flt> const &v) {
  xsum_length c = static_cast<xsum_length>(v.size());
  if (c == 0) {
    return;
  }

  xsum_small_accumulator *sacc = _sacc.get();

  xsum_flt const *vec = v.data();

  while (c > 1) {
    if (sacc->adds_until_propagate == 0) {
      carry_propagate();
    }

    xsum_length const m =
        c - ((1 <= sacc->adds_until_propagate) ? c - 1
                                               : sacc->adds_until_propagate);

    add_sqnorm_no_carry(vec, m + 1);

    sacc->adds_until_propagate -= m;

    vec += m;
    c -= m;
  }

  xsum_flt const f = *vec;
  xsum_flt const g = f * f;
  add(g);
}

void xsum_small::add_dot(xsum_flt const *v1, xsum_flt const *v2,
                         xsum_length const n) {
  if (n == 0) {
    return;
  }

  xsum_small_accumulator *sacc = _sacc.get();

  xsum_flt const *vec1 = v1;
  xsum_flt const *vec2 = v2;
  xsum_length c = n;
  while (c > 1) {
    if (sacc->adds_until_propagate == 0) {
      carry_propagate();
    }

    xsum_length const m =
        c - ((1 <= sacc->adds_until_propagate) ? c - 1
                                               : sacc->adds_until_propagate);

    add_dot_no_carry(vec1, vec2, m + 1);

    vec1 += m;
    vec2 += m;

    sacc->adds_until_propagate -= m;

    c -= m;
  }

  xsum_flt const f1 = *vec1;
  xsum_flt const f2 = *vec2;
  xsum_flt const g = f1 * f2;
  add(g);
}

void xsum_small::add_dot(std::vector<xsum_flt> const &v1,
                         std::vector<xsum_flt> const &v2) {
  xsum_length c = static_cast<xsum_length>(v1.size());
  if (c == 0 || c > static_cast<xsum_length>(v2.size())) {
    return;
  }

  xsum_small_accumulator *sacc = _sacc.get();

  xsum_flt const *vec1 = v1.data();
  xsum_flt const *vec2 = v2.data();

  while (c > 1) {
    if (sacc->adds_until_propagate == 0) {
      carry_propagate();
    }

    xsum_length const m =
        c - ((1 <= sacc->adds_until_propagate) ? c - 1
                                               : sacc->adds_until_propagate);

    add_dot_no_carry(vec1, vec2, m + 1);

    vec1 += m;
    vec2 += m;

    sacc->adds_until_propagate -= m;

    c -= m;
  }

  xsum_flt const f1 = *vec1;
  xsum_flt const f2 = *vec2;
  xsum_flt const g = f1 * f2;
  add(g);
}

xsum_flt xsum_small::round() {
  if (xsum_debug) {
    std::cout << "Rounding small accumulator\n";
  }

  /* See if we have a NaN from one of the numbers being a NaN, in which
     case we return the NaN with largest payload. */

  fpunion u;

  if (_sacc->NaN != 0) {
    u.intv = _sacc->NaN;
    return u.fltv;
  }

  /* Otherwise, if any number was infinite, we return +Inf, -Inf, or a Nan
     (if both +Inf and -Inf occurred).  Note that we do NOT return NaN if
     we have both an infinite number and a sum of other numbers that
     overflows with opposite sign, since there is no real ambiguity in
     such a case. */

  if (_sacc->Inf != 0) {
    u.intv = _sacc->Inf;
    return u.fltv;
  }

  /* If none of the numbers summed were infinite or NaN, we proceed to
     propagate carries, as a preliminary to finding the magnitude of
     the sum. This also ensures that the sign of the result can be
     determined from the uppermost non-zero chunk.
     We also find the index, i, of this uppermost non-zero chunk, as
     the value returned by xsum_carry_propagate, and set ivalue to
     sacc->chunk[i]. Note that ivalue will not be 0 or -1, unless
     i is 0 (the lowest chunk), in which case it will be handled by
     the code for denormalized numbers. */

  int i = carry_propagate();

  if (xsum_debug) {
    display();
  }

  xsum_int ivalue = _sacc->chunk[i];

  /* Handle a possible denormalized number, including zero. */

  if (i <= 1) {
    /* Check for zero value, in which case we can return immediately. */

    if (ivalue == 0) {
      return 0.0;
    }

    /* Check if it is actually a denormalized number.  It always is if only
       the lowest chunk is non-zero. If the highest non-zero chunk is
       the next-to-lowest, we check the magnitude of the absolute value.
       Note that the real exponent is 1 (not 0), so we need to shift right
       by 1 here, which also means there will be no overflow from the left
       shift below (but must view absolute value as unsigned). */

    if (i == 0) {
      u.intv = ivalue >= 0 ? ivalue : -ivalue;
      u.intv >>= 1;
      if (ivalue < 0) {
        u.intv |= XSUM_SIGN_MASK;
      }
      return u.fltv;
    } else {
      /* Note: Left shift of -ve number is undefined, so do a multiply
         instead, which is probably optimized to a shift. */

      u.intv =
          ivalue * (static_cast<xsum_int>(1) << (XSUM_LOW_MANTISSA_BITS - 1)) +
          (_sacc->chunk[0] >> 1);

      if (u.intv < 0) {
        if (u.intv > -(static_cast<xsum_int>(1) << XSUM_MANTISSA_BITS)) {
          u.intv = (-u.intv) | XSUM_SIGN_MASK;
          return u.fltv;
        }
      } else {
        if (u.uintv < static_cast<xsum_uint>(1) << XSUM_MANTISSA_BITS) {
          return u.fltv;
        }
      }
      /* otherwise, it's not actually denormalized, so fall through to below */
    }
  }

  /* Find the location of the uppermost 1 bit in the absolute value of the
     upper chunk by converting it (as a signed integer) to a floating point
     value, and looking at the exponent.  Then set 'more' to the number of
     bits from the lower chunk (and maybe the next lower) that are needed
     to fill out the mantissa of the result, plus an extra bit to help decide
     on rounding.  For negative numbers, it may turn out later that we need
     another bit because negating a negative value may carry out of the top
     here, but not once more bits are shifted into the bottom later on. */

  u.fltv = static_cast<xsum_flt>(ivalue);

  int e = (u.uintv >> XSUM_MANTISSA_BITS) & XSUM_EXP_MASK;
  int more = 1 + XSUM_MANTISSA_BITS + XSUM_EXP_BIAS - e;

  if (xsum_debug) {
    std::cout << "e: " << e << ", more: " << more << ", ivalue: " << std::hex
              << std::setfill('0') << std::setw(16)
              << static_cast<long long>(ivalue) << "\n";
  }

  /* Change 'ivalue' to put in 'more' bits from lower chunks into the bottom.
     Also set 'j' to the index of the lowest chunk from which these bits came,
     and 'lower' to the remaining bits of that chunk not now in 'ivalue'.
     We make sure that 'lower' initially has at least one bit in it, which
     we can later move into 'ivalue' if it turns out that one more bit is
     needed. */

  /* multiply, since << of negative undefined */
  ivalue *= static_cast<xsum_int>(1) << more;

  if (xsum_debug) {
    std::cout << "after ivalue <<= more, ivalue: " << std::hex
              << std::setfill('0') << std::setw(16)
              << static_cast<long long>(ivalue) << "\n";
  }

  int j = i - 1;

  /* must exist, since denormalized if i==0 */
  xsum_schunk lower = _sacc->chunk[j];

  if (more >= XSUM_LOW_MANTISSA_BITS) {
    more -= XSUM_LOW_MANTISSA_BITS;
    ivalue += lower << more;
    if (xsum_debug) {
      std::cout << "after ivalue += lower << more, ivalue: " << std::hex
                << std::setfill('0') << std::setw(16)
                << static_cast<long long>(ivalue) << "\n";
    }

    --j;

    lower = ((j < 0) ? 0 : _sacc->chunk[j]);
  }

  ivalue += lower >> (XSUM_LOW_MANTISSA_BITS - more);
  lower &= (static_cast<xsum_schunk>(1) << (XSUM_LOW_MANTISSA_BITS - more)) - 1;

  if (xsum_debug) {
    std::cout << "j: " << j << ", new e: " << e
              << ", new |ivalue|: " << std::hex << std::setfill('0')
              << std::setw(16)
              << static_cast<long long>(ivalue < 0 ? -ivalue : ivalue)
              << ", lower: " << std::hex << std::setfill('0') << std::setw(16)
              << static_cast<long long>(lower) << "\n";
  }

  /* Check for a negative 'ivalue' that when negated doesn't contain a full
     mantissa's worth of bits, plus one to help rounding.  If so, move one
     more bit into 'ivalue' from 'lower' (and remove it from 'lower').
     Note that more than one additional bit will not be required because
     xsum_carry_propagate ensures the uppermost non-zero chunk is not -1. */

  if (ivalue < 0 && ((-ivalue) & (static_cast<xsum_int>(1)
                                  << (XSUM_MANTISSA_BITS + 1))) == 0) {
    int const pos = static_cast<xsum_schunk>(1)
                    << (XSUM_LOW_MANTISSA_BITS - 1 - more);
    /* note that left shift undefined if ivalue is negative */
    ivalue *= 2;
    if (lower & pos) {
      ivalue |= 1;
      lower &= ~pos;
    }
    --e;
  }

  if (xsum_debug) {
    std::cout << "j: " << j << ", new e: " << e
              << ", new |ivalue|: " << std::hex << std::setfill('0')
              << std::setw(16)
              << static_cast<long long>(ivalue < 0 ? -ivalue : ivalue)
              << ", lower: " << std::hex << std::setfill('0') << std::setw(16)
              << static_cast<long long>(lower) << "\n";
  }

  /* Set u.intv to have just the correct sign bit (rest zeros), and 'ivalue'
     to now have the absolute value of the mantissa. */

  if (ivalue >= 0) {
    u.intv = 0;
  } else {
    ivalue = -ivalue;
    u.intv = XSUM_SIGN_MASK;
  }

  if (xsum_debug) {
    if ((ivalue >> (XSUM_MANTISSA_BITS + 1)) != 1) {
      std::abort();
    }
  }

  /* Round to nearest, with ties to even. At this point, 'ivalue' has the
     absolute value of the number to be rounded, including an extra bit at
     the bottom.  Bits below that are in 'lower' and in the chunks
     indexed by 'j' and below.  Note that the bits in 'lower' and the chunks
     below add to the magnitude of the remainder if the number is positive,
     but subtract from this magnitude if the number is negative.
     This code goes to done_rounding if it finds that just discarding lower
     order bits is correct, and to round_away_from_zero if instead the
     magnitude should be increased by one in the lowest bit. */

  /* extra bit is 0 */
  if ((ivalue & 1) == 0) {
    if (xsum_debug) {
      std::cout << "round toward zero, since remainder magnitude is < 1/2\n";
    }

    goto done_rounding;
  }

  /* number is positive */
  if (u.intv == 0) {
    /* low bit 1 (odd) */
    if ((ivalue & 2) != 0) {
      if (xsum_debug) {
        std::cout
            << "round away from zero, since magnitude >= 1/2, goes to even\n";
      }

      goto round_away_from_zero;
    }

    if (lower != 0) {
      if (xsum_debug) {
        std::cout
            << "round away from zero, since magnitude > 1/2 (from 'lower')\n";
      }

      goto round_away_from_zero;
    }
  }
  /* number is negative */
  else {
    /* low bit 0 (even) */
    if ((ivalue & 2) == 0) {
      if (xsum_debug) {
        std::cout
            << "round toward zero, since magnitude <= 1/2, goes to even\n";
      }

      goto done_rounding;
    }

    if (lower != 0) {
      if (xsum_debug) {
        std::cout
            << "round toward zero, since magnitude < 1/2 (from 'lower')\n";
      }

      goto done_rounding;
    }
  }

  /* If we get here, 'lower' is zero.  We need to look at chunks lower down
     to see if any are non-zero. */

  while (j > 0) {
    --j;

    if (_sacc->chunk[j] != 0) {
      lower = 1;
      break;
    }
  }

  /* number is positive, low bit 0 (even) */
  if (u.intv == 0) {
    if (lower != 0) {
      if (xsum_debug) {
        std::cout
            << "round away from zero, since magnitude > 1/2 (low chunks)\n";
      }

      goto round_away_from_zero;
    } else {
      if (xsum_debug) {
        std::cout << "round toward zero, magnitude == 1/2 (low chunks)\n";
      }

      goto done_rounding;
    }
  }
  /* number is negative, low bit 1 (odd) */
  else {
    if (lower != 0) {
      if (xsum_debug) {
        std::cout << "round toward zero, since magnitude < 1/2 (low chunks)\n";
      }

      goto done_rounding;
    } else {
      if (xsum_debug) {
        std::cout << "round away from zero, magnitude == 1/2 (low chunks)\n";
      }

      goto round_away_from_zero;
    }
  }

round_away_from_zero:
  /* Round away from zero, then check for carry having propagated out the
     top, and shift if so. */
  ivalue += 2;
  if (ivalue & (static_cast<xsum_int>(1) << (XSUM_MANTISSA_BITS + 2))) {
    ivalue >>= 1;
    ++e;
  }

done_rounding:;

  /* Get rid of the bottom bit that was used to decide on rounding. */

  ivalue >>= 1;

  /* Adjust to the true exponent, accounting for where this chunk is. */

  e += (i << XSUM_LOW_EXP_BITS) - XSUM_EXP_BIAS - XSUM_MANTISSA_BITS;

  /* If exponent has overflowed, change to plus or minus Inf and return. */

  if (e >= XSUM_EXP_MASK) {
    u.intv |= static_cast<xsum_int>(XSUM_EXP_MASK) << XSUM_MANTISSA_BITS;

    if (xsum_debug) {
      std::cout << "Final rounded result: " << std::scientific
                << std::setprecision(17) << u.fltv << " (overflowed)\n  ";
      pbinary_double(u.fltv);
      std::cout << "\n";
    }

    return u.fltv;
  }

  /* Put exponent and mantissa into u.intv, which already has the sign,
     then return u.fltv. */

  u.intv += static_cast<xsum_int>(e) << XSUM_MANTISSA_BITS;
  u.intv += ivalue & XSUM_MANTISSA_MASK; /* mask out the implicit 1 bit */

  if (xsum_debug) {
    if ((ivalue >> XSUM_MANTISSA_BITS) != 1) {
      std::abort();
    }
    std::cout << "Final rounded result: " << std::scientific
              << std::setprecision(17) << u.fltv << "\n  ";
    pbinary_double(u.fltv);
    std::cout << "\n";
  }

  return u.fltv;
}

void xsum_small::display() {
  xsum_small_accumulator *sacc = _sacc.get();

  std::cout << "Small accumulator:" << (sacc->Inf ? "  Inf" : "")
            << (sacc->NaN ? "  NaN" : "") << "\n";

  for (int i = XSUM_SCHUNKS - 1, dots = 0; i >= 0; --i) {
    if (sacc->chunk[i] == 0) {
      if (!dots) {
        dots = 1;
        std::cout << "            ...\n";
      }
    } else {
      std::cout << std::setw(5) << i << " " << std::setw(5)
                << static_cast<int>((i << XSUM_LOW_EXP_BITS) - XSUM_EXP_BIAS -
                                    XSUM_MANTISSA_BITS)
                << " "
                << std::bitset<XSUM_SCHUNK_BITS - 32>(
                       static_cast<std::int64_t>(sacc->chunk[i] >> 32))
                << " "
                << std::bitset<32>(
                       static_cast<std::int64_t>(sacc->chunk[i] & 0xffffffff))
                << "\n";
      dots = 0;
    }
  }
  std::cout << "\n";
}

int xsum_small::chunks_used() {
  xsum_small_accumulator *sacc = _sacc.get();
  int c = 0;
  for (int i = 0; i < XSUM_SCHUNKS; ++i) {
    if (sacc->chunk[i] != 0) {
      ++c;
    }
  }
  return c;
}

inline xsum_small_accumulator *xsum_small::get() const noexcept {
  return _sacc.get();
}

int xsum_small::carry_propagate() {
  if (xsum_debug) {
    std::cout << "Carry propagating in small accumulator\n";
  }

  /* Set u to the index of the uppermost non-zero (for now) chunk, or
     return with value 0 if there is none. */

  bool found = false;

  int u = XSUM_SCHUNKS - 1;
  switch (XSUM_SCHUNKS & 0x3) {
    case 3:
      if (_sacc->chunk[u] != 0) {
        found = true;
        break;
      }
      --u;
    case 2:
      if (_sacc->chunk[u] != 0) {
        found = true;
        break;
      }
      --u;
    case 1:
      if (_sacc->chunk[u] != 0) {
        found = true;
        break;
      }
      --u;
    case 0:;
  }

  if (!found) {
    do {
      if (_sacc->chunk[u - 3] | _sacc->chunk[u - 2] | _sacc->chunk[u - 1] |
          _sacc->chunk[u]) {
        found = true;
        break;
      }
      u -= 4;
    } while (u >= 0);
  }

  if (found) {
    while (_sacc->chunk[u] == 0) {
      --u;
    }
  } else {
    if (xsum_debug) {
      std::cout << "number is zero (1)\n";
    }

    _sacc->adds_until_propagate = XSUM_SMALL_CARRY_TERMS - 1;

    /* Return index of uppermost non-zero chunk. */
    return 0;
  }

  if (xsum_debug) {
    std::cout << "u: \n" << u;
  }

  /* Carry propagate, starting at the low-order chunks.  Note that the
     loop limit of u may be increased inside the loop. */

  /* indicates that a non-zero chunk has not been found yet */
  int uix = -1;

  /* Quickly skip over unused low-order chunks.  Done here at the start
     on the theory that there are often many unused low-order chunks,
     justifying some overhead to begin, but later stretches of unused
     chunks may not be as large. */

  int i = 0;
  int const e = u - 3;
  while (i <= e) {
    if (_sacc->chunk[i] | _sacc->chunk[i + 1] | _sacc->chunk[i + 2] |
        _sacc->chunk[i + 3]) {
      break;
    }
    i += 4;
  }

  xsum_schunk c;

  do {
    bool nonzero = false;

    /* Find the next non-zero chunk, or break out of loop if there is none. */
    do {
      c = _sacc->chunk[i];

      if (c != 0) {
        nonzero = true;
        break;
      }

      ++i;

      if (i > u) {
        break;
      }

      c = _sacc->chunk[i];

      if (c != 0) {
        nonzero = true;
        break;
      }

      ++i;
    } while (i <= u);

    if (!nonzero) {
      break;
    }

    /* Propagate possible carry from this chunk to next chunk up. */
    xsum_schunk const chigh = c >> XSUM_LOW_MANTISSA_BITS;
    if (chigh == 0) {
      uix = i;
      ++i;
      /* no need to change this chunk */
      continue;
    }

    if (u == i) {
      if (chigh == -1) {
        uix = i;
        /* don't propagate -1 into the region of all zeros above */
        break;
      }

      /* we will change chunk[u+1], so we'll need to look at it */
      u = i + 1;
    }

    xsum_schunk const clow = c & XSUM_LOW_MANTISSA_MASK;
    if (clow != 0) {
      uix = i;
    }

    /* We now change chunk[i] and add to chunk[i+1]. Note that i+1 should be
       in range (no bigger than XSUM_CHUNKS-1) because the number of chunks
       is big enough to hold any sum, and we do not store redundant chunks
       with values 0 or -1 above previously non-zero chunks. */

    if (xsum_debug) {
      if (i + 1 >= XSUM_SCHUNKS) {
        std::abort();
      }
    }

    _sacc->chunk[i++] = clow;
    _sacc->chunk[i] += chigh;
  } while (i <= u);

  if (xsum_debug) {
    std::cout << "  uix: " << uix << "  new u: " << u << "\n";
  }

  /* Check again for the number being zero, since carry propagation might
   have created zero from something that initially looked non-zero. */

  if (uix < 0) {
    if (xsum_debug) {
      std::cout << "number is zero (2)\n";
    }

    _sacc->adds_until_propagate = XSUM_SMALL_CARRY_TERMS - 1;

    /* Return index of uppermost non-zero chunk. */
    return 0;
  }

  /* While the uppermost chunk is negative, with value -1, combine it with
     the chunk below (if there is one) to produce the same number but with
     one fewer non-zero chunks. */

  while (_sacc->chunk[uix] == -1 && uix > 0) {
    _sacc->chunk[uix] = 0;
    --uix;
    // @yafshar solution
    // _sacc->chunk[uix] +=
    // static_cast<xsum_schunk>(static_cast<xsum_lchunk>(-1) <<
    // XSUM_LOW_MANTISSA_BITS); Neal solution A shift of a negative number is
    // undefined according to the standard, so do a multiply - it's all
    // presumably constant-folded by the compiler.
    _sacc->chunk[uix] +=
        static_cast<xsum_schunk>(-1) *
        (static_cast<xsum_schunk>(1) << XSUM_LOW_MANTISSA_BITS);
  }

  /* We can now add one less than the total allowed terms before the
     next carry propagate. */

  _sacc->adds_until_propagate = XSUM_SMALL_CARRY_TERMS - 1;

  /* Return index of uppermost non-zero chunk. */
  return uix;
}

void xsum_small::add_inf_nan(xsum_int const ivalue) {
  xsum_int const mantissa = ivalue & XSUM_MANTISSA_MASK;

  /* Inf */
  if (mantissa == 0) {
    if (_sacc->Inf == 0) {
      /* no previous Inf */
      _sacc->Inf = ivalue;
    } else if (_sacc->Inf != ivalue) {
      fpunion u;

      /* previous Inf was opposite sign */
      u.intv = ivalue;

      /* result will be a NaN */
      u.fltv = u.fltv - u.fltv;

      _sacc->Inf = u.intv;
    }
  }
  /* NaN */
  else {
    /* Choose the NaN with the bigger payload and clear its sign.  Using <=
       ensures that we will choose the first NaN over the previous zero. */
    if ((_sacc->NaN & XSUM_MANTISSA_MASK) <= mantissa) {
      _sacc->NaN = ivalue & ~XSUM_SIGN_MASK;
    }
  }
}

inline void xsum_small::add_no_carry(xsum_flt const value) {
  if (xsum_debug) {
    std::cout << "ADD +" << std::setprecision(17) << static_cast<double>(value)
              << "\n     ";
    pbinary_double(static_cast<double>(value));
    std::cout << "\n";
  }

  fpunion u;

  /* Extract exponent and mantissa. */
  u.fltv = value;

  xsum_int const ivalue = u.intv;

  xsum_int mantissa = ivalue & XSUM_MANTISSA_MASK;
  xsum_expint exp = (ivalue >> XSUM_MANTISSA_BITS) & XSUM_EXP_MASK;

  /* Categorize number as normal, denormalized, or Inf/NaN according to
     the value of the exponent field. */

  /* normalized OR in implicit 1 bit at top of mantissa */
  if (exp != 0 && exp != XSUM_EXP_MASK) {
    mantissa |= static_cast<xsum_int>(1) << XSUM_MANTISSA_BITS;
  }
  /* zero or denormalized */
  else if (exp == 0) {
    /* If it's a zero (positive or negative), we do nothing. */
    if (mantissa == 0) {
      return;
    }

    /* Denormalized mantissa has no implicit 1, but exponent is 1 not 0. */
    exp = 1;
  }
  /* Inf or NaN */
  else {
    /* Just update flags in accumulator structure. */
    add_inf_nan(ivalue);
    return;
  }

  /* Separate high part of exponent, used as index of chunk, and low
     part of exponent, giving position within chunk. */

  xsum_expint const low_exp = exp & XSUM_LOW_EXP_MASK;
  xsum_expint const high_exp = exp >> XSUM_LOW_EXP_BITS;

  if (xsum_debug) {
    std::cout << "  high exp: " << std::bitset<XSUM_HIGH_EXP_BITS>(high_exp)
              << "  low exp: " << std::bitset<XSUM_LOW_EXP_BITS>(low_exp)
              << "\n";
  }

  xsum_schunk *const chunk_ptr = _sacc->chunk + high_exp;
  xsum_schunk const chunk0 = chunk_ptr[0];
  xsum_schunk const chunk1 = chunk_ptr[1];

  /* Separate mantissa into two parts, after shifting, and add to (or
     subtract from) this chunk and the next higher chunk (which always
     exists since there are three extra ones at the top). */

  xsum_int const low_mantissa =
      (static_cast<xsum_uint>(mantissa) << low_exp) & XSUM_LOW_MANTISSA_MASK;
  xsum_int const high_mantissa = mantissa >> (XSUM_LOW_MANTISSA_BITS - low_exp);

  /* Add or subtract to or from the two affected chunks. */

  if (ivalue < 0) {
    chunk_ptr[0] = chunk0 - low_mantissa;
    chunk_ptr[1] = chunk1 - high_mantissa;

    if (xsum_debug) {
      std::cout << " -high man: "
                << std::bitset<XSUM_MANTISSA_BITS>(-high_mantissa)
                << "\n  -low man: "
                << std::bitset<XSUM_LOW_MANTISSA_BITS>(-low_mantissa) << "\n";
    }
  } else {
    chunk_ptr[0] = chunk0 + low_mantissa;
    chunk_ptr[1] = chunk1 + high_mantissa;

    if (xsum_debug) {
      std::cout << "  high man: "
                << std::bitset<XSUM_MANTISSA_BITS>(high_mantissa)
                << "\n   low man: "
                << std::bitset<XSUM_LOW_MANTISSA_BITS>(low_mantissa) << "\n";
    }
  }
}

inline void xsum_small::add_no_carry(xsum_small_accumulator const *value) {
  if (value->Inf != 0) {
    if (_sacc->Inf == 0) {
      _sacc->Inf = value->Inf;
    } else if (_sacc->Inf != value->Inf) {
      fpunion u;
      u.intv = value->Inf;

      /* result will be a NaN */
      u.fltv = u.fltv - u.fltv;

      _sacc->Inf = u.intv;
    }
    return;
  }

  if (value->NaN != 0) {
    if ((_sacc->NaN & XSUM_MANTISSA_MASK) < (value->NaN & XSUM_MANTISSA_MASK)) {
      _sacc->NaN = value->NaN;
    }
    return;
  }

  xsum_schunk *sc = _sacc->chunk;
  xsum_schunk const *vc = value->chunk;
  for (int i = 0; i < XSUM_SCHUNKS; ++i) {
    sc[i] += vc[i];
  }
}

inline void xsum_small::add_no_carry(xsum_flt const *vec, xsum_length const n) {
  for (xsum_length i = 0; i < n - 1; ++i) {
    xsum_flt const f = vec[i];
    add_no_carry(f);
  }
}

inline void xsum_small::add_sqnorm_no_carry(xsum_flt const *vec,
                                            xsum_length const n) {
  for (xsum_length i = 0; i < n - 1; ++i) {
    xsum_flt const f = vec[i];
    xsum_flt const g = f * f;
    add_no_carry(g);
  }
}

inline void xsum_small::add_dot_no_carry(xsum_flt const *vec1,
                                         xsum_flt const *vec2,
                                         xsum_length const n) {
  for (xsum_length i = 0; i < n - 1; ++i) {
    xsum_flt const f = vec1[i];
    xsum_flt const g = vec2[i];
    xsum_flt const h = f * g;
    add_no_carry(h);
  }
}

/* LARGE ACCUMULATOR */

xsum_large::xsum_large() : _lacc(new xsum_large_accumulator) {}

xsum_large::xsum_large(xsum_large_accumulator const &lacc)
    : _lacc(new xsum_large_accumulator) {
  std::copy(lacc.chunk, lacc.chunk + XSUM_LCHUNKS, _lacc->chunk);
  std::copy(lacc.count, lacc.count + XSUM_LCHUNKS, _lacc->count);
  std::copy(lacc.chunks_used, lacc.chunks_used + XSUM_LCHUNKS / 64,
            _lacc->chunks_used);
  _lacc->used_used = lacc.used_used;
  std::copy(lacc.sacc.chunk, lacc.sacc.chunk + XSUM_SCHUNKS, _lacc->sacc.chunk);
  _lacc->sacc.Inf = lacc.sacc.Inf;
  _lacc->sacc.NaN = lacc.sacc.NaN;
  _lacc->sacc.adds_until_propagate = lacc.sacc.adds_until_propagate;
}

xsum_large::xsum_large(xsum_large_accumulator const *lacc)
    : _lacc(new xsum_large_accumulator) {
  if (lacc) {
    std::copy(lacc->chunk, lacc->chunk + XSUM_LCHUNKS, _lacc->chunk);
    std::copy(lacc->count, lacc->count + XSUM_LCHUNKS, _lacc->count);
    std::copy(lacc->chunks_used, lacc->chunks_used + XSUM_LCHUNKS / 64,
              _lacc->chunks_used);
    _lacc->used_used = lacc->used_used;
    std::copy(lacc->sacc.chunk, lacc->sacc.chunk + XSUM_SCHUNKS,
              _lacc->sacc.chunk);
    _lacc->sacc.Inf = lacc->sacc.Inf;
    _lacc->sacc.NaN = lacc->sacc.NaN;
    _lacc->sacc.adds_until_propagate = lacc->sacc.adds_until_propagate;
  }
}

xsum_large::xsum_large(xsum_small_accumulator const &sacc)
    : _lacc(new xsum_large_accumulator) {
  std::copy(sacc.chunk, sacc.chunk + XSUM_SCHUNKS, _lacc->sacc.chunk);
  _lacc->sacc.Inf = sacc.Inf;
  _lacc->sacc.NaN = sacc.NaN;
  _lacc->sacc.adds_until_propagate = sacc.adds_until_propagate;
}

xsum_large::xsum_large(xsum_small_accumulator const *sacc)
    : _lacc(new xsum_large_accumulator) {
  std::copy(sacc->chunk, sacc->chunk + XSUM_SCHUNKS, _lacc->sacc.chunk);
  _lacc->sacc.Inf = sacc->Inf;
  _lacc->sacc.NaN = sacc->NaN;
  _lacc->sacc.adds_until_propagate = sacc->adds_until_propagate;
}

xsum_large::xsum_large(xsum_small const &sacc) : xsum_large(sacc.get()) {}

xsum_large::xsum_large(xsum_small const *sacc) : xsum_large(sacc->get()) {}

void xsum_large::reset() { _lacc.reset(new xsum_large_accumulator); }

void xsum_large::init() {
  std::fill(_lacc->count, _lacc->count + XSUM_LCHUNKS, -1);
  std::fill(_lacc->chunks_used, _lacc->chunks_used + XSUM_LCHUNKS / 64, 0);
  _lacc->used_used = 0;
  std::fill(_lacc->sacc.chunk, _lacc->sacc.chunk + XSUM_SCHUNKS, 0);
  _lacc->sacc.Inf = 0;
  _lacc->sacc.NaN = 0;
  _lacc->sacc.adds_until_propagate = XSUM_SMALL_CARRY_TERMS;
}

void xsum_large::add(xsum_flt const value) {
  if (xsum_debug) {
    std::cout << "LARGE ADD SINGLE NUMBER\n";
  }

  /* Version not manually optimized - maybe the compiler can do better. */
  fpunion u;

  /* Fetch the next number, and convert to integer form in u.uintv. */
  u.fltv = value;

  /* Isolate the upper sign+exponent bits that index the chunk. */
  xsum_expint const ix = u.uintv >> XSUM_MANTISSA_BITS;

  /* Find the count for this chunk, and subtract one. */
  xsum_lcount const count = _lacc->count[ix] - 1;

  if (count < 0) {
    /* If the decremented count is negative, it's either a special
       Inf/NaN chunk (in which case count will stay at -1), or one that
       needs to be transferred to the small accumulator, or one that
       has never been used before and needs to be initialized. */
    add_value_inf_nan(ix, u.uintv);
  } else {
    /* Store the decremented count of additions allowed before transfer,
       and add this value to the chunk. */
    _lacc->count[ix] = count;
    _lacc->chunk[ix] += u.uintv;
  }
}

void xsum_large::add(xsum_small_accumulator const *const value) {
  if (xsum_debug) {
    std::cout << "LARGE ADD SMALL ACCUMULATOR VALUE\n";
  }

  if (_lacc->sacc.adds_until_propagate == 0) {
    carry_propagate();
  }

  add_no_carry(value);

  --_lacc->sacc.adds_until_propagate;
}

void xsum_large::add(xsum_large_accumulator *const value) {
  xsum_small_accumulator *sacc = round_to_small(value);
  add(sacc);
}

void xsum_large::add(xsum_flt const *vec, xsum_length const n) {
  if (n == 0) {
    return;
  }

  if (xsum_debug) {
    std::cout << "LARGE ADD OF " << n << " VALUES\n";
  }

  /* Version that's been manually optimized:  Loop unrolled, pre-fetch
     attempted, branches eliminated, ... */
  fpunion u1;
  fpunion u2;

  int count1;
  int count2;

  xsum_expint ix1;
  xsum_expint ix2;

  xsum_flt const *v = vec;

  /* Unrolled loop processing two values each time around.  The loop is
     done as two nested loops, arranged so that the inner one will have
     no branches except for the one looping back.  This is achieved by
     a trick for combining three tests for negativity into one.  The
     last one or two values are not done here, so that the pre-fetching
     will not go past the end of the array (which would probably be OK,
     but is technically not allowed). */

  /* leave out last one or two, terminate when negative, for trick */
  xsum_length m = n - 3;
  while (m >= 0) {
    /* Loop processing two values at a time until we're done, or until
       one (or both) of the values result in a chunk needing to be processed.
       Updates are done here for both of these chunks, even though it is not
       yet known whether these updates ought to have been done.  We hope
       this allows for better memory pre-fetch and instruction scheduling. */
    do {
      u1.fltv = *v++;
      u2.fltv = *v++;

      ix1 = u1.uintv >> XSUM_MANTISSA_BITS;
      count1 = _lacc->count[ix1] - 1;
      _lacc->count[ix1] = count1;
      _lacc->chunk[ix1] += u1.uintv;

      ix2 = u2.uintv >> XSUM_MANTISSA_BITS;
      count2 = _lacc->count[ix2] - 1;
      _lacc->count[ix2] = count2;
      _lacc->chunk[ix2] += u2.uintv;

      m -= 2;
    } while ((static_cast<xsum_length>(count1) |
              static_cast<xsum_length>(count2) | m) >= 0);
    /* ... equivalent to while (count1 >= 0 && count2 >= 0 && m >= 0) */

    /* See if we were actually supposed to update these chunks.  If not,
           back out the changes and then process the chunks as they ought to
           have been processed. */

    if (count1 < 0 || count2 < 0) {
      _lacc->count[ix2] = count2 + 1;
      _lacc->chunk[ix2] -= u2.uintv;

      if (count1 < 0) {
        _lacc->count[ix1] = count1 + 1;
        _lacc->chunk[ix1] -= u1.uintv;

        add_value_inf_nan(ix1, u1.uintv);

        count2 = _lacc->count[ix2] - 1;
      }

      if (count2 < 0) {
        add_value_inf_nan(ix2, u2.uintv);
      } else {
        _lacc->count[ix2] = count2;
        _lacc->chunk[ix2] += u2.uintv;
      }
    }
  }

  /* Process the last one or two values, without pre-fetching. */

  m += 3;
  for (;;) {
    u1.fltv = *v++;
    ix1 = u1.uintv >> XSUM_MANTISSA_BITS;
    count1 = _lacc->count[ix1] - 1;

    if (count1 < 0) {
      add_value_inf_nan(ix1, u1.uintv);
    } else {
      _lacc->count[ix1] = count1;
      _lacc->chunk[ix1] += u1.uintv;
    }

    --m;
    if (m == 0) {
      break;
    }
  }
}

void xsum_large::add(std::vector<xsum_flt> const &vec) {
  xsum_length const n = static_cast<xsum_length>(vec.size());
  if (n == 0) {
    return;
  }

  if (xsum_debug) {
    std::cout << "LARGE ADD OF " << n << " VALUES\n";
  }

  /* Version that's been manually optimized:  Loop unrolled, pre-fetch
     attempted, branches eliminated, ... */
  fpunion u1;
  fpunion u2;

  int count1;
  int count2;

  xsum_expint ix1;
  xsum_expint ix2;

  xsum_flt const *v = vec.data();

  /* Unrolled loop processing two values each time around.  The loop is
     done as two nested loops, arranged so that the inner one will have
     no branches except for the one looping back.  This is achieved by
     a trick for combining three tests for negativity into one.  The
     last one or two values are not done here, so that the pre-fetching
     will not go past the end of the array (which would probably be OK,
     but is technically not allowed). */

  /* leave out last one or two, terminate when negative, for trick */
  xsum_length m = n - 3;
  while (m >= 0) {
    /* Loop processing two values at a time until we're done, or until
       one (or both) of the values result in a chunk needing to be processed.
       Updates are done here for both of these chunks, even though it is not
       yet known whether these updates ought to have been done.  We hope
       this allows for better memory pre-fetch and instruction scheduling. */
    do {
      u1.fltv = *v++;
      u2.fltv = *v++;

      ix1 = u1.uintv >> XSUM_MANTISSA_BITS;
      count1 = _lacc->count[ix1] - 1;
      _lacc->count[ix1] = count1;
      _lacc->chunk[ix1] += u1.uintv;

      ix2 = u2.uintv >> XSUM_MANTISSA_BITS;
      count2 = _lacc->count[ix2] - 1;
      _lacc->count[ix2] = count2;
      _lacc->chunk[ix2] += u2.uintv;

      m -= 2;
    } while ((static_cast<xsum_length>(count1) |
              static_cast<xsum_length>(count2) | m) >= 0);
    /* ... equivalent to while (count1 >= 0 && count2 >= 0 && m >= 0) */

    /* See if we were actually supposed to update these chunks.  If not,
           back out the changes and then process the chunks as they ought to
           have been processed. */

    if (count1 < 0 || count2 < 0) {
      _lacc->count[ix2] = count2 + 1;
      _lacc->chunk[ix2] -= u2.uintv;

      if (count1 < 0) {
        _lacc->count[ix1] = count1 + 1;
        _lacc->chunk[ix1] -= u1.uintv;

        add_value_inf_nan(ix1, u1.uintv);

        count2 = _lacc->count[ix2] - 1;
      }

      if (count2 < 0) {
        add_value_inf_nan(ix2, u2.uintv);
      } else {
        _lacc->count[ix2] = count2;
        _lacc->chunk[ix2] += u2.uintv;
      }
    }
  }

  /* Process the last one or two values, without pre-fetching. */

  m += 3;
  for (;;) {
    u1.fltv = *v++;
    ix1 = u1.uintv >> XSUM_MANTISSA_BITS;
    count1 = _lacc->count[ix1] - 1;

    if (count1 < 0) {
      add_value_inf_nan(ix1, u1.uintv);
    } else {
      _lacc->count[ix1] = count1;
      _lacc->chunk[ix1] += u1.uintv;
    }

    --m;
    if (m == 0) {
      break;
    }
  }
}

void xsum_large::add_sqnorm(xsum_flt const *vec, xsum_length const n) {
  if (n == 0) {
    return;
  }

  if (xsum_debug) {
    std::cout << "LARGE ADD_SQNORM OF " << n << " VALUES\n";
  }

  fpunion u1;
  fpunion u2;

  int count1;
  int count2;

  xsum_expint ix1;
  xsum_expint ix2;

  xsum_flt const *v = vec;

  /* Unrolled loop processing two squares each time around.  The loop is
     done as two nested loops, arranged so that the inner one will have
     no branches except for the one looping back.  This is achieved by
     a trick for combining three tests for negativity into one.  The
     last one or two squares are not done here, so that the pre-fetching
     will not go past the end of the array (which would probably be OK,
     but is technically not allowed). */

  /* leave out last one or two, terminate when negative, for trick */
  xsum_length m = n - 3;

  while (m >= 0) {
    /* Loop processing two squares at a time until we're done, or until
       one (or both) of them result in a chunk needing to be processed.
       Updates are done here for both of these chunks, even though it is not
       yet known whether these updates ought to have been done.  We hope
       this allows for better memory pre-fetch and instruction scheduling. */
    do {
      u1.fltv = *v * *v;
      ++v;
      u2.fltv = *v * *v;
      ++v;

      ix1 = u1.uintv >> XSUM_MANTISSA_BITS;
      count1 = _lacc->count[ix1] - 1;
      _lacc->count[ix1] = count1;
      _lacc->chunk[ix1] += u1.uintv;

      ix2 = u2.uintv >> XSUM_MANTISSA_BITS;
      count2 = _lacc->count[ix2] - 1;
      _lacc->count[ix2] = count2;
      _lacc->chunk[ix2] += u2.uintv;

      m -= 2;
    } while ((static_cast<xsum_length>(count1) |
              static_cast<xsum_length>(count2) | m) >= 0);
    /* ... equivalent to while (count1 >= 0 && count2 >= 0 && m >= 0) */

    /* See if we were actually supposed to update these chunks.  If not,
       back out the changes and then process the chunks as they ought to
       have been processed. */

    if (count1 < 0 || count2 < 0) {
      _lacc->count[ix2] = count2 + 1;
      _lacc->chunk[ix2] -= u2.uintv;

      if (count1 < 0) {
        _lacc->count[ix1] = count1 + 1;
        _lacc->chunk[ix1] -= u1.uintv;

        add_value_inf_nan(ix1, u1.uintv);

        count2 = _lacc->count[ix2] - 1;
      }

      if (count2 < 0) {
        add_value_inf_nan(ix2, u2.uintv);
      } else {
        _lacc->count[ix2] = count2;
        _lacc->chunk[ix2] += u2.uintv;
      }
    }
  }

  /* Process the last one or two squares, without pre-fetching. */

  m += 3;
  for (;;) {
    u1.fltv = *v * *v;
    ++v;

    ix1 = u1.uintv >> XSUM_MANTISSA_BITS;

    count1 = _lacc->count[ix1] - 1;
    if (count1 < 0) {
      add_value_inf_nan(ix1, u1.uintv);
    } else {
      _lacc->count[ix1] = count1;
      _lacc->chunk[ix1] += u1.uintv;
    }

    --m;
    if (m == 0) {
      break;
    }
  }
}

void xsum_large::add_sqnorm(std::vector<xsum_flt> const &vec) {
  xsum_length const n = static_cast<xsum_length>(vec.size());
  if (n == 0) {
    return;
  }

  if (xsum_debug) {
    std::cout << "LARGE ADD_SQNORM OF " << n << " VALUES\n";
  }

  fpunion u1;
  fpunion u2;

  int count1;
  int count2;

  xsum_expint ix1;
  xsum_expint ix2;

  xsum_flt const *v = vec.data();

  /* Unrolled loop processing two squares each time around.  The loop is
     done as two nested loops, arranged so that the inner one will have
     no branches except for the one looping back.  This is achieved by
     a trick for combining three tests for negativity into one.  The
     last one or two squares are not done here, so that the pre-fetching
     will not go past the end of the array (which would probably be OK,
     but is technically not allowed). */

  /* leave out last one or two, terminate when negative, for trick */
  xsum_length m = n - 3;

  while (m >= 0) {
    /* Loop processing two squares at a time until we're done, or until
       one (or both) of them result in a chunk needing to be processed.
       Updates are done here for both of these chunks, even though it is not
       yet known whether these updates ought to have been done.  We hope
       this allows for better memory pre-fetch and instruction scheduling. */
    do {
      u1.fltv = *v * *v;
      ++v;
      u2.fltv = *v * *v;
      ++v;

      ix1 = u1.uintv >> XSUM_MANTISSA_BITS;
      count1 = _lacc->count[ix1] - 1;
      _lacc->count[ix1] = count1;
      _lacc->chunk[ix1] += u1.uintv;

      ix2 = u2.uintv >> XSUM_MANTISSA_BITS;
      count2 = _lacc->count[ix2] - 1;
      _lacc->count[ix2] = count2;
      _lacc->chunk[ix2] += u2.uintv;

      m -= 2;
    } while ((static_cast<xsum_length>(count1) |
              static_cast<xsum_length>(count2) | m) >= 0);
    /* ... equivalent to while (count1 >= 0 && count2 >= 0 && m >= 0) */

    /* See if we were actually supposed to update these chunks.  If not,
       back out the changes and then process the chunks as they ought to
       have been processed. */

    if (count1 < 0 || count2 < 0) {
      _lacc->count[ix2] = count2 + 1;
      _lacc->chunk[ix2] -= u2.uintv;

      if (count1 < 0) {
        _lacc->count[ix1] = count1 + 1;
        _lacc->chunk[ix1] -= u1.uintv;

        add_value_inf_nan(ix1, u1.uintv);

        count2 = _lacc->count[ix2] - 1;
      }

      if (count2 < 0) {
        add_value_inf_nan(ix2, u2.uintv);
      } else {
        _lacc->count[ix2] = count2;
        _lacc->chunk[ix2] += u2.uintv;
      }
    }
  }

  /* Process the last one or two squares, without pre-fetching. */

  m += 3;
  for (;;) {
    u1.fltv = *v * *v;
    ++v;

    ix1 = u1.uintv >> XSUM_MANTISSA_BITS;

    count1 = _lacc->count[ix1] - 1;
    if (count1 < 0) {
      add_value_inf_nan(ix1, u1.uintv);
    } else {
      _lacc->count[ix1] = count1;
      _lacc->chunk[ix1] += u1.uintv;
    }

    --m;
    if (m == 0) {
      break;
    }
  }
}

void xsum_large::add_dot(xsum_flt const *vec1, xsum_flt const *vec2,
                         xsum_length const n) {
  if (n == 0) {
    return;
  }

  if (xsum_debug) {
    std::cout << "LARGE ADD_DOT OF " << n << " VALUES\n";
  }

  /* Version that's been manually optimized:  Loop unrolled, pre-fetch
     attempted, branches eliminated, ... */

  fpunion u1;
  fpunion u2;

  int count1;
  int count2;

  xsum_expint ix1;
  xsum_expint ix2;

  xsum_flt const *v1 = vec1;
  xsum_flt const *v2 = vec2;

  /* Unrolled loop processing two products each time around.  The loop is
     done as two nested loops, arranged so that the inner one will have
     no branches except for the one looping back.  This is achieved by
     a trick for combining three tests for negativity into one.  The
     last one or two products are not done here, so that the pre-fetching
     will not go past the end of the array (which would probably be OK,
     but is technically not allowed). */

  /* leave out last one or two, terminate when negative, for trick */
  xsum_length m = n - 3;

  while (m >= 0) {
    /* Loop processing two products at a time until we're done, or until
       one (or both) of them result in a chunk needing to be processed.
       Updates are done here for both of these chunks, even though it is not
       yet known whether these updates ought to have been done.  We hope
       this allows for better memory pre-fetch and instruction scheduling. */
    do {
      u1.fltv = *v1 * *v2;
      ++v1;
      ++v2;

      u2.fltv = *v1 * *v2;
      ++v1;
      ++v2;

      ix1 = u1.uintv >> XSUM_MANTISSA_BITS;
      count1 = _lacc->count[ix1] - 1;
      _lacc->count[ix1] = count1;
      _lacc->chunk[ix1] += u1.uintv;

      ix2 = u2.uintv >> XSUM_MANTISSA_BITS;
      count2 = _lacc->count[ix2] - 1;
      _lacc->count[ix2] = count2;
      _lacc->chunk[ix2] += u2.uintv;

      m -= 2;
    } while ((static_cast<xsum_length>(count1) |
              static_cast<xsum_length>(count2) | m) >= 0);
    /* ... equivalent to while (count1 >= 0 && count2 >= 0 && m >= 0) */

    /* See if we were actually supposed to update these chunks.  If not,
       back out the changes and then process the chunks as they ought to
       have been processed. */

    if (count1 < 0 || count2 < 0) {
      _lacc->count[ix2] = count2 + 1;
      _lacc->chunk[ix2] -= u2.uintv;

      if (count1 < 0) {
        _lacc->count[ix1] = count1 + 1;
        _lacc->chunk[ix1] -= u1.uintv;

        add_value_inf_nan(ix1, u1.uintv);

        count2 = _lacc->count[ix2] - 1;
      }

      if (count2 < 0) {
        add_value_inf_nan(ix2, u2.uintv);
      } else {
        _lacc->count[ix2] = count2;
        _lacc->chunk[ix2] += u2.uintv;
      }
    }
  }

  /* Process the last one or two products, without pre-fetching. */

  m += 3;
  for (;;) {
    u1.fltv = *v1 * *v2;
    ++v1;
    ++v2;

    ix1 = u1.uintv >> XSUM_MANTISSA_BITS;

    count1 = _lacc->count[ix1] - 1;
    if (count1 < 0) {
      add_value_inf_nan(ix1, u1.uintv);
    } else {
      _lacc->count[ix1] = count1;
      _lacc->chunk[ix1] += u1.uintv;
    }

    --m;
    if (m == 0) {
      break;
    }
  }
}

void xsum_large::add_dot(std::vector<xsum_flt> const &vec1,
                         std::vector<xsum_flt> const &vec2) {
  xsum_length const n = static_cast<xsum_length>(vec1.size());
  if (n == 0 || n > static_cast<xsum_length>(vec2.size())) {
    return;
  }

  if (xsum_debug) {
    std::cout << "LARGE ADD_DOT OF " << n << " VALUES\n";
  }

  /* Version that's been manually optimized:  Loop unrolled, pre-fetch
     attempted, branches eliminated, ... */

  fpunion u1;
  fpunion u2;

  int count1;
  int count2;

  xsum_expint ix1;
  xsum_expint ix2;

  xsum_flt const *v1 = vec1.data();
  xsum_flt const *v2 = vec2.data();

  /* Unrolled loop processing two products each time around.  The loop is
     done as two nested loops, arranged so that the inner one will have
     no branches except for the one looping back.  This is achieved by
     a trick for combining three tests for negativity into one.  The
     last one or two products are not done here, so that the pre-fetching
     will not go past the end of the array (which would probably be OK,
     but is technically not allowed). */

  /* leave out last one or two, terminate when negative, for trick */
  xsum_length m = n - 3;

  while (m >= 0) {
    /* Loop processing two products at a time until we're done, or until
       one (or both) of them result in a chunk needing to be processed.
       Updates are done here for both of these chunks, even though it is not
       yet known whether these updates ought to have been done.  We hope
       this allows for better memory pre-fetch and instruction scheduling. */
    do {
      u1.fltv = *v1 * *v2;
      ++v1;
      ++v2;

      u2.fltv = *v1 * *v2;
      ++v1;
      ++v2;

      ix1 = u1.uintv >> XSUM_MANTISSA_BITS;
      count1 = _lacc->count[ix1] - 1;
      _lacc->count[ix1] = count1;
      _lacc->chunk[ix1] += u1.uintv;

      ix2 = u2.uintv >> XSUM_MANTISSA_BITS;
      count2 = _lacc->count[ix2] - 1;
      _lacc->count[ix2] = count2;
      _lacc->chunk[ix2] += u2.uintv;

      m -= 2;
    } while ((static_cast<xsum_length>(count1) |
              static_cast<xsum_length>(count2) | m) >= 0);
    /* ... equivalent to while (count1 >= 0 && count2 >= 0 && m >= 0) */

    /* See if we were actually supposed to update these chunks.  If not,
       back out the changes and then process the chunks as they ought to
       have been processed. */

    if (count1 < 0 || count2 < 0) {
      _lacc->count[ix2] = count2 + 1;
      _lacc->chunk[ix2] -= u2.uintv;

      if (count1 < 0) {
        _lacc->count[ix1] = count1 + 1;
        _lacc->chunk[ix1] -= u1.uintv;

        add_value_inf_nan(ix1, u1.uintv);

        count2 = _lacc->count[ix2] - 1;
      }

      if (count2 < 0) {
        add_value_inf_nan(ix2, u2.uintv);
      } else {
        _lacc->count[ix2] = count2;
        _lacc->chunk[ix2] += u2.uintv;
      }
    }
  }

  /* Process the last one or two products, without pre-fetching. */

  m += 3;
  for (;;) {
    u1.fltv = *v1 * *v2;
    ++v1;
    ++v2;

    ix1 = u1.uintv >> XSUM_MANTISSA_BITS;

    count1 = _lacc->count[ix1] - 1;
    if (count1 < 0) {
      add_value_inf_nan(ix1, u1.uintv);
    } else {
      _lacc->count[ix1] = count1;
      _lacc->chunk[ix1] += u1.uintv;
    }

    --m;
    if (m == 0) {
      break;
    }
  }
}

xsum_flt xsum_large::round() {
  if (xsum_debug) {
    std::cout << "Rounding large accumulator\n";
  }

  xsum_used *p = _lacc->chunks_used;
  xsum_used *e = p + XSUM_LCHUNKS / 64;

  /* Very quickly skip some unused low-order blocks of chunks
     by looking at the used_used flags. */

  xsum_used uu = _lacc->used_used;
  if ((uu & 0xffffffff) == 0) {
    uu >>= 32;
    p += 32;
  }

  if ((uu & 0xffff) == 0) {
    uu >>= 16;
    p += 16;
  }

  if ((uu & 0xff) == 0) {
    p += 8;
  }

  /* Loop over remaining blocks of chunks. */
  xsum_used u;
  int ix;
  do {
    /* Loop to quickly find the next non-zero block of used flags, or finish
       up if we've added all the used blocks to the small accumulator. */

    for (;;) {
      u = *p;
      if (u != 0) {
        break;
      }

      ++p;
      if (p == e) {
        return sround();
      }

      u = *p;
      if (u != 0) {
        break;
      }

      ++p;
      if (p == e) {
        return sround();
      }

      u = *p;
      if (u != 0) {
        break;
      }

      ++p;
      if (p == e) {
        return sround();
      }

      u = *p;
      if (u != 0) {
        break;
      }

      ++p;
      if (p == e) {
        return sround();
      }
    }

    /* Find and process the chunks in this block that are used.  We skip
       forward based on the chunks_used flags until we're within eight
       bits of a chunk that is in use. */

    ix = (p - _lacc->chunks_used) << 6;
    if ((u & 0xffffffff) == 0) {
      u >>= 32;
      ix += 32;
    }

    if ((u & 0xffff) == 0) {
      u >>= 16;
      ix += 16;
    }

    if ((u & 0xff) == 0) {
      u >>= 8;
      ix += 8;
    }

    do {
      if (_lacc->count[ix] >= 0) {
        add_lchunk_to_small(ix);
      }

      ++ix;
      u >>= 1;
    } while (u != 0);

    ++p;
  } while (p != e);

  /* Finish now that all blocks have been added to the small accumulator
     by calling the small accumulator rounding function. */
  return sround();
}

xsum_small_accumulator *xsum_large::round_to_small() {
  xsum_used const *p = _lacc->chunks_used;
  xsum_used const *e = p + XSUM_LCHUNKS / 64;

  /* Very quickly skip some unused low-order blocks of chunks
     by looking at the used_used flags. */
  xsum_used uu = _lacc->used_used;

  if ((uu & 0xffffffff) == 0) {
    uu >>= 32;
    p += 32;
  }
  if ((uu & 0xffff) == 0) {
    uu >>= 16;
    p += 16;
  }
  if ((uu & 0xff) == 0) {
    p += 8;
  }

  /* Loop over remaining blocks of chunks. */
  xsum_used u;
  int ix;
  do {
    /* Loop to quickly find the next non-zero block of used flags, or finish
       up if we've added all the used blocks to the small accumulator. */

    for (;;) {
      u = *p;
      if (u != 0) {
        break;
      }

      ++p;
      if (p == e) {
        return &_lacc->sacc;
      }

      u = *p;
      if (u != 0) {
        break;
      }

      ++p;
      if (p == e) {
        return &_lacc->sacc;
      }

      u = *p;
      if (u != 0) {
        break;
      }

      ++p;
      if (p == e) {
        return &_lacc->sacc;
      }

      u = *p;
      if (u != 0) {
        break;
      }

      ++p;
      if (p == e) {
        return &_lacc->sacc;
      }
    }

    /* Find and process the chunks in this block that are used.  We skip
       forward based on the chunks_used flags until we're within eight
       bits of a chunk that is in use. */

    ix = (p - _lacc->chunks_used) << 6;
    if ((u & 0xffffffff) == 0) {
      u >>= 32;
      ix += 32;
    }

    if ((u & 0xffff) == 0) {
      u >>= 16;
      ix += 16;
    }

    if ((u & 0xff) == 0) {
      u >>= 8;
      ix += 8;
    }

    do {
      if (_lacc->count[ix] >= 0) {
        add_lchunk_to_small(ix);
      }

      ++ix;
      u >>= 1;
    } while (u != 0);

    ++p;
  } while (p != e);

  /* Finish now that all blocks have been added to the small accumulator
     by calling the small accumulator rounding function. */
  return &_lacc->sacc;
}

xsum_small_accumulator *xsum_large::round_to_small(
    xsum_large_accumulator *const lacc) {
  xsum_used const *p = lacc->chunks_used;
  xsum_used const *e = p + XSUM_LCHUNKS / 64;

  /* Very quickly skip some unused low-order blocks of chunks
     by looking at the used_used flags. */
  xsum_used uu = lacc->used_used;

  if ((uu & 0xffffffff) == 0) {
    uu >>= 32;
    p += 32;
  }
  if ((uu & 0xffff) == 0) {
    uu >>= 16;
    p += 16;
  }
  if ((uu & 0xff) == 0) {
    p += 8;
  }

  /* Loop over remaining blocks of chunks. */
  xsum_used u;
  int ix;
  do {
    /* Loop to quickly find the next non-zero block of used flags, or finish
       up if we've added all the used blocks to the small accumulator. */

    for (;;) {
      u = *p;
      if (u != 0) {
        break;
      }

      ++p;
      if (p == e) {
        return &lacc->sacc;
      }

      u = *p;
      if (u != 0) {
        break;
      }

      ++p;
      if (p == e) {
        return &lacc->sacc;
      }

      u = *p;
      if (u != 0) {
        break;
      }

      ++p;
      if (p == e) {
        return &lacc->sacc;
      }

      u = *p;
      if (u != 0) {
        break;
      }

      ++p;
      if (p == e) {
        return &lacc->sacc;
      }
    }

    /* Find and process the chunks in this block that are used.  We skip
       forward based on the chunks_used flags until we're within eight
       bits of a chunk that is in use. */

    ix = (p - lacc->chunks_used) << 6;
    if ((u & 0xffffffff) == 0) {
      u >>= 32;
      ix += 32;
    }

    if ((u & 0xffff) == 0) {
      u >>= 16;
      ix += 16;
    }

    if ((u & 0xff) == 0) {
      u >>= 8;
      ix += 8;
    }

    do {
      if (lacc->count[ix] >= 0) {
        add_lchunk_to_small(ix);
      }

      ++ix;
      u >>= 1;
    } while (u != 0);

    ++p;
  } while (p != e);

  /* Finish now that all blocks have been added to the small accumulator
     by calling the small accumulator rounding function. */
  return &lacc->sacc;
}

void xsum_large::display() {
  std::cout << "Large accumulator:\n";

  int dots = 0;
  for (int i = XSUM_LCHUNKS - 1; i >= 0; --i) {
    if (_lacc->count[i] < 0) {
      if (!dots) {
        std::cout << "            ...\n";
      }
      dots = 1;
    } else {
      std::cout << (i & 0x800 ? '-' : '+') << std::setw(4) << (i & 0x7ff) << " "
                << std::setw(5) << _lacc->count[i] << " "
                << std::bitset<XSUM_LCHUNK_BITS - 32>(
                       static_cast<std::int64_t>(_lacc->chunk[i]) >> 32)
                << " "
                << std::bitset<32>(static_cast<std::int64_t>(_lacc->chunk[i]) &
                                   0xffffffff)
                << "\n";
      dots = 0;
    }
  }

  std::cout << "\nWithin the large accumulator:  ";

  sdisplay();
}

void xsum_large::sdisplay() {
  std::cout << "Small accumulator:" << (_lacc->sacc.Inf ? "  Inf" : "")
            << (_lacc->sacc.NaN ? "  NaN" : "") << "\n";

  for (int i = XSUM_SCHUNKS - 1, dots = 0; i >= 0; --i) {
    if (_lacc->sacc.chunk[i] == 0) {
      if (!dots) {
        dots = 1;
        std::cout << "            ...\n";
      }
    } else {
      std::cout << std::setw(5) << i << " " << std::setw(5)
                << static_cast<int>((i << XSUM_LOW_EXP_BITS) - XSUM_EXP_BIAS -
                                    XSUM_MANTISSA_BITS)
                << " "
                << std::bitset<XSUM_SCHUNK_BITS - 32>(
                       static_cast<std::int64_t>(_lacc->sacc.chunk[i] >> 32))
                << " "
                << std::bitset<32>(static_cast<std::int64_t>(
                       _lacc->sacc.chunk[i] & 0xffffffff))
                << "\n";
      dots = 0;
    }
  }
  std::cout << "\n";
}

int xsum_large::chunks_used() {
  int c = 0;
  for (int i = 0; i < XSUM_LCHUNKS; ++i) {
    if (_lacc->count[i] >= 0) {
      ++c;
    }
  }
  return c;
}

int xsum_large::carry_propagate() {
  if (xsum_debug) {
    std::cout << "Carry propagating in small accumulator\n";
  }

  /* Set u to the index of the uppermost non-zero (for now) chunk, or
     return with value 0 if there is none. */

  bool found = false;

  int u = XSUM_SCHUNKS - 1;
  switch (XSUM_SCHUNKS & 0x3) {
    case 3:
      if (_lacc->sacc.chunk[u] != 0) {
        found = true;
        break;
      }
      --u;
    case 2:
      if (_lacc->sacc.chunk[u] != 0) {
        found = true;
        break;
      }
      --u;
    case 1:
      if (_lacc->sacc.chunk[u] != 0) {
        found = true;
        break;
      }
      --u;
    case 0:;
  }

  if (!found) {
    do {
      if (_lacc->sacc.chunk[u - 3] | _lacc->sacc.chunk[u - 2] |
          _lacc->sacc.chunk[u - 1] | _lacc->sacc.chunk[u]) {
        found = true;
        break;
      }
      u -= 4;
    } while (u >= 0);
  }

  if (found) {
    while (_lacc->sacc.chunk[u] == 0) {
      --u;
    }
  } else {
    if (xsum_debug) {
      std::cout << "number is zero (1)\n";
    }

    _lacc->sacc.adds_until_propagate = XSUM_SMALL_CARRY_TERMS - 1;

    /* Return index of uppermost non-zero chunk. */
    return 0;
  }

  if (xsum_debug) {
    std::cout << "u: \n" << u;
  }

  /* Carry propagate, starting at the low-order chunks.  Note that the
     loop limit of u may be increased inside the loop. */

  /* indicates that a non-zero chunk has not been found yet */
  int uix = -1;

  /* Quickly skip over unused low-order chunks.  Done here at the start
     on the theory that there are often many unused low-order chunks,
     justifying some overhead to begin, but later stretches of unused
     chunks may not be as large. */

  int i = 0;
  int const e = u - 3;
  while (i <= e) {
    if (_lacc->sacc.chunk[i] | _lacc->sacc.chunk[i + 1] |
        _lacc->sacc.chunk[i + 2] | _lacc->sacc.chunk[i + 3]) {
      break;
    }
    i += 4;
  }

  xsum_schunk c;

  do {
    bool nonzero = false;

    /* Find the next non-zero chunk, or break out of loop if there is none. */
    do {
      c = _lacc->sacc.chunk[i];

      if (c != 0) {
        nonzero = true;
        break;
      }

      ++i;

      if (i > u) {
        break;
      }

      c = _lacc->sacc.chunk[i];

      if (c != 0) {
        nonzero = true;
        break;
      }

      ++i;
    } while (i <= u);

    if (!nonzero) {
      break;
    }

    /* Propagate possible carry from this chunk to next chunk up. */
    xsum_schunk const chigh = c >> XSUM_LOW_MANTISSA_BITS;
    if (chigh == 0) {
      uix = i;
      ++i;
      /* no need to change this chunk */
      continue;
    }

    if (u == i) {
      if (chigh == -1) {
        uix = i;
        /* don't propagate -1 into the region of all zeros above */
        break;
      }

      /* we will change chunk[u+1], so we'll need to look at it */
      u = i + 1;
    }

    xsum_schunk const clow = c & XSUM_LOW_MANTISSA_MASK;

    if (clow != 0) {
      uix = i;
    }

    /* We now change chunk[i] and add to chunk[i+1]. Note that i+1 should be
       in range (no bigger than XSUM_CHUNKS-1) because the number of chunks
       is big enough to hold any sum, and we do not store redundant chunks
       with values 0 or -1 above previously non-zero chunks. */

    if (xsum_debug) {
      if (i + 1 >= XSUM_SCHUNKS) {
        std::abort();
      }
    }

    _lacc->sacc.chunk[i++] = clow;
    _lacc->sacc.chunk[i] += chigh;
  } while (i <= u);

  if (xsum_debug) {
    std::cout << "  uix: " << uix << "  new u: " << u << "\n";
  }

  /* Check again for the number being zero, since carry propagation might
   have created zero from something that initially looked non-zero. */

  if (uix < 0) {
    if (xsum_debug) {
      std::cout << "number is zero (2)\n";
    }

    _lacc->sacc.adds_until_propagate = XSUM_SMALL_CARRY_TERMS - 1;

    /* Return index of uppermost non-zero chunk. */
    return 0;
  }

  /* While the uppermost chunk is negative, with value -1, combine it with
     the chunk below (if there is one) to produce the same number but with
     one fewer non-zero chunks. */

  while (_lacc->sacc.chunk[uix] == -1 && uix > 0) {
    _lacc->sacc.chunk[uix] = 0;
    --uix;
    // @yafshar solution
    // _lacc->sacc.chunk[uix] +=
    // static_cast<xsum_schunk>(static_cast<xsum_lchunk>(-1) <<
    // XSUM_LOW_MANTISSA_BITS); Neal solution A shift of a negative number is
    // undefined according to the standard, so do a multiply - it's all
    // presumably constant-folded by the compiler.
    _lacc->sacc.chunk[uix] +=
        static_cast<xsum_schunk>(-1) *
        (static_cast<xsum_schunk>(1) << XSUM_LOW_MANTISSA_BITS);
  }

  /* We can now add one less than the total allowed terms before the
     next carry propagate. */

  _lacc->sacc.adds_until_propagate = XSUM_SMALL_CARRY_TERMS - 1;

  /* Return index of uppermost non-zero chunk. */
  return uix;
}

inline void xsum_large::add_no_carry(
    xsum_small_accumulator const *const value) {
  if (value->Inf != 0) {
    if (_lacc->sacc.Inf == 0) {
      _lacc->sacc.Inf = value->Inf;
    } else if (_lacc->sacc.Inf != value->Inf) {
      fpunion u;
      u.intv = value->Inf;

      /* result will be a NaN */
      u.fltv = u.fltv - u.fltv;

      _lacc->sacc.Inf = u.intv;
    }
    return;
  }

  if (value->NaN != 0) {
    if ((_lacc->sacc.NaN & XSUM_MANTISSA_MASK) <
        (value->NaN & XSUM_MANTISSA_MASK)) {
      _lacc->sacc.NaN = value->NaN;
    }
    return;
  }

  xsum_schunk *sc = _lacc->sacc.chunk;
  xsum_schunk const *vc = value->chunk;
  for (int i = 0; i < XSUM_SCHUNKS; ++i) {
    sc[i] += vc[i];
  }
}

xsum_flt xsum_large::sround() {
  if (xsum_debug) {
    std::cout << "Rounding small accumulator\n";
  }

  /* See if we have a NaN from one of the numbers being a NaN, in which
     case we return the NaN with largest payload. */

  fpunion u;

  if (_lacc->sacc.NaN != 0) {
    u.intv = _lacc->sacc.NaN;
    return u.fltv;
  }

  /* Otherwise, if any number was infinite, we return +Inf, -Inf, or a Nan
     (if both +Inf and -Inf occurred).  Note that we do NOT return NaN if
     we have both an infinite number and a sum of other numbers that
     overflows with opposite sign, since there is no real ambiguity in
     such a case. */

  if (_lacc->sacc.Inf != 0) {
    u.intv = _lacc->sacc.Inf;
    return u.fltv;
  }

  /* If none of the numbers summed were infinite or NaN, we proceed to
     propagate carries, as a preliminary to finding the magnitude of
     the sum.  This also ensures that the sign of the result can be
     determined from the uppermost non-zero chunk.
     We also find the index, i, of this uppermost non-zero chunk, as
     the value returned by xsum_carry_propagate, and set ivalue to
     sacc->chunk[i].  Note that ivalue will not be 0 or -1, unless
     i is 0 (the lowest chunk), in which case it will be handled by
     the code for denormalized numbers. */

  int i = carry_propagate();

  if (xsum_debug) {
    display();
  }

  xsum_int ivalue = _lacc->sacc.chunk[i];

  /* Handle a possible denormalized number, including zero. */

  if (i <= 1) {
    /* Check for zero value, in which case we can return immediately. */

    if (ivalue == 0) {
      return 0.0;
    }

    /* Check if it is actually a denormalized number.  It always is if only
       the lowest chunk is non-zero. If the highest non-zero chunk is
       the next-to-lowest, we check the magnitude of the absolute value.
       Note that the real exponent is 1 (not 0), so we need to shift right
       by 1 here, which also means there will be no overflow from the left
       shift below (but must view absolute value as unsigned). */

    if (i == 0) {
      u.intv = ivalue >= 0 ? ivalue : -ivalue;
      u.intv >>= 1;
      if (ivalue < 0) {
        u.intv |= XSUM_SIGN_MASK;
      }
      return u.fltv;
    } else {
      /* Note: Left shift of -ve number is undefined, so do a multiply
         instead, which is probably optimized to a shift. */

      u.intv =
          ivalue * (static_cast<xsum_int>(1) << (XSUM_LOW_MANTISSA_BITS - 1)) +
          (_lacc->sacc.chunk[0] >> 1);

      if (u.intv < 0) {
        if (u.intv > -(static_cast<xsum_int>(1) << XSUM_MANTISSA_BITS)) {
          u.intv = (-u.intv) | XSUM_SIGN_MASK;
          return u.fltv;
        }
      } else {
        if (u.uintv < static_cast<xsum_uint>(1) << XSUM_MANTISSA_BITS) {
          return u.fltv;
        }
      }
      /* otherwise, it's not actually denormalized, so fall through to below */
    }
  }

  /* Find the location of the uppermost 1 bit in the absolute value of the
   upper chunk by converting it (as a signed integer) to a floating point
   value, and looking at the exponent.  Then set 'more' to the number of
   bits from the lower chunk (and maybe the next lower) that are needed
   to fill out the mantissa of the result, plus an extra bit to help decide
   on rounding.  For negative numbers, it may turn out later that we need
   another bit because negating a negative value may carry out of the top
   here, but not once more bits are shifted into the bottom later on. */

  u.fltv = static_cast<xsum_flt>(ivalue);

  int e = (u.uintv >> XSUM_MANTISSA_BITS) & XSUM_EXP_MASK;
  int more = 1 + XSUM_MANTISSA_BITS + XSUM_EXP_BIAS - e;

  if (xsum_debug) {
    std::cout << "e: " << e << ", more: " << more << ", ivalue: " << std::hex
              << std::setfill('0') << std::setw(16)
              << static_cast<long long>(ivalue) << "\n";
  }

  /* Change 'ivalue' to put in 'more' bits from lower chunks into the bottom.
     Also set 'j' to the index of the lowest chunk from which these bits came,
     and 'lower' to the remaining bits of that chunk not now in 'ivalue'.
     We make sure that 'lower' initially has at least one bit in it, which
     we can later move into 'ivalue' if it turns out that one more bit is
     needed. */

  /* multiply, since << of negative undefined */
  ivalue *= static_cast<xsum_int>(1) << more;

  if (xsum_debug) {
    std::cout << "after ivalue <<= more, ivalue: " << std::hex
              << std::setfill('0') << std::setw(16)
              << static_cast<long long>(ivalue) << "\n";
  }

  int j = i - 1;

  /* must exist, since denormalized if i==0 */
  xsum_schunk lower = _lacc->sacc.chunk[j];

  if (more >= XSUM_LOW_MANTISSA_BITS) {
    more -= XSUM_LOW_MANTISSA_BITS;
    ivalue += lower << more;
    if (xsum_debug) {
      std::cout << "after ivalue += lower << more, ivalue: " << std::hex
                << std::setfill('0') << std::setw(16)
                << static_cast<long long>(ivalue) << "\n";
    }

    --j;

    lower = ((j < 0) ? 0 : _lacc->sacc.chunk[j]);
  }

  ivalue += lower >> (XSUM_LOW_MANTISSA_BITS - more);
  lower &= (static_cast<xsum_schunk>(1) << (XSUM_LOW_MANTISSA_BITS - more)) - 1;

  if (xsum_debug) {
    std::cout << "j: " << j << ", new e: " << e
              << ", new |ivalue|: " << std::hex << std::setfill('0')
              << std::setw(16)
              << static_cast<long long>(ivalue < 0 ? -ivalue : ivalue)
              << ", lower: " << std::hex << std::setfill('0') << std::setw(16)
              << static_cast<long long>(lower) << "\n";
  }

  /* Check for a negative 'ivalue' that when negated doesn't contain a full
     mantissa's worth of bits, plus one to help rounding.  If so, move one
     more bit into 'ivalue' from 'lower' (and remove it from 'lower').
     Note that more than one additional bit will not be required because
     xsum_carry_propagate ensures the uppermost non-zero chunk is not -1. */

  if (ivalue < 0 && ((-ivalue) & (static_cast<xsum_int>(1)
                                  << (XSUM_MANTISSA_BITS + 1))) == 0) {
    int const pos = static_cast<xsum_schunk>(1)
                    << (XSUM_LOW_MANTISSA_BITS - 1 - more);
    /* note that left shift undefined if ivalue is negative */
    ivalue *= 2;
    if (lower & pos) {
      ivalue |= 1;
      lower &= ~pos;
    }
    --e;
  }

  if (xsum_debug) {
    std::cout << "j: " << j << ", new e: " << e
              << ", new |ivalue|: " << std::hex << std::setfill('0')
              << std::setw(16)
              << static_cast<long long>(ivalue < 0 ? -ivalue : ivalue)
              << ", lower: " << std::hex << std::setfill('0') << std::setw(16)
              << static_cast<long long>(lower) << "\n";
  }

  /* Set u.intv to have just the correct sign bit (rest zeros), and 'ivalue'
     to now have the absolute value of the mantissa. */

  if (ivalue >= 0) {
    u.intv = 0;
  } else {
    ivalue = -ivalue;
    u.intv = XSUM_SIGN_MASK;
  }

  if (xsum_debug) {
    if ((ivalue >> (XSUM_MANTISSA_BITS + 1)) != 1) {
      std::abort();
    }
  }

  /* Round to nearest, with ties to even. At this point, 'ivalue' has the
     absolute value of the number to be rounded, including an extra bit at
     the bottom.  Bits below that are in 'lower' and in the chunks
     indexed by 'j' and below.  Note that the bits in 'lower' and the chunks
     below add to the magnitude of the remainder if the number is positive,
     but subtract from this magnitude if the number is negative.
     This code goes to done_rounding if it finds that just discarding lower
     order bits is correct, and to round_away_from_zero if instead the
     magnitude should be increased by one in the lowest bit. */

  /* extra bit is 0 */
  if ((ivalue & 1) == 0) {
    if (xsum_debug) {
      std::cout << "round toward zero, since remainder magnitude is < 1/2\n";
    }

    goto done_rounding;
  }

  /* number is positive */
  if (u.intv == 0) {
    /* low bit 1 (odd) */
    if ((ivalue & 2) != 0) {
      if (xsum_debug) {
        std::cout
            << "round away from zero, since magnitude >= 1/2, goes to even\n";
      }

      goto round_away_from_zero;
    }

    if (lower != 0) {
      if (xsum_debug) {
        std::cout
            << "round away from zero, since magnitude > 1/2 (from 'lower')\n";
      }

      goto round_away_from_zero;
    }
  }
  /* number is negative */
  else {
    /* low bit 0 (even) */
    if ((ivalue & 2) == 0) {
      if (xsum_debug) {
        std::cout
            << "round toward zero, since magnitude <= 1/2, goes to even\n";
      }

      goto done_rounding;
    }

    if (lower != 0) {
      if (xsum_debug) {
        std::cout
            << "round toward zero, since magnitude < 1/2 (from 'lower')\n";
      }

      goto done_rounding;
    }
  }

  /* If we get here, 'lower' is zero.  We need to look at chunks lower down
     to see if any are non-zero. */

  while (j > 0) {
    --j;

    if (_lacc->sacc.chunk[j] != 0) {
      lower = 1;
      break;
    }
  }

  /* number is positive, low bit 0 (even) */
  if (u.intv == 0) {
    if (lower != 0) {
      if (xsum_debug) {
        std::cout
            << "round away from zero, since magnitude > 1/2 (low chunks)\n";
      }

      goto round_away_from_zero;
    } else {
      if (xsum_debug) {
        std::cout << "round toward zero, magnitude == 1/2 (low chunks)\n";
      }

      goto done_rounding;
    }
  } else /* number is negative, low bit 1 (odd) */
  {
    if (lower != 0) {
      if (xsum_debug) {
        std::cout << "round toward zero, since magnitude < 1/2 (low chunks)\n";
      }

      goto done_rounding;
    } else {
      if (xsum_debug) {
        std::cout << "round away from zero, magnitude == 1/2 (low chunks)\n";
      }

      goto round_away_from_zero;
    }
  }

round_away_from_zero:
  /* Round away from zero, then check for carry having propagated out the
     top, and shift if so. */
  ivalue += 2;
  if (ivalue & (static_cast<xsum_int>(1) << (XSUM_MANTISSA_BITS + 2))) {
    ivalue >>= 1;
    ++e;
  }

done_rounding:;

  /* Get rid of the bottom bit that was used to decide on rounding. */

  ivalue >>= 1;

  /* Adjust to the true exponent, accounting for where this chunk is. */

  e += (i << XSUM_LOW_EXP_BITS) - XSUM_EXP_BIAS - XSUM_MANTISSA_BITS;

  /* If exponent has overflowed, change to plus or minus Inf and return. */

  if (e >= XSUM_EXP_MASK) {
    u.intv |= static_cast<xsum_int>(XSUM_EXP_MASK) << XSUM_MANTISSA_BITS;
    if (xsum_debug) {
      std::cout << "Final rounded result: " << std::scientific
                << std::setprecision(17) << u.fltv << " (overflowed)\n  ";
      pbinary_double(u.fltv);
      std::cout << "\n";
    }
    return u.fltv;
  }

  /* Put exponent and mantissa into u.intv, which already has the sign,
     then return u.fltv. */

  u.intv += static_cast<xsum_int>(e) << XSUM_MANTISSA_BITS;
  u.intv += ivalue & XSUM_MANTISSA_MASK; /* mask out the implicit 1 bit */

  if (xsum_debug) {
    if ((ivalue >> XSUM_MANTISSA_BITS) != 1) {
      std::abort();
    }
    std::cout << "Final rounded result: " << std::scientific
              << std::setprecision(17) << u.fltv << "\n  ";
    pbinary_double(u.fltv);
    std::cout << "\n";
  }

  return u.fltv;
}

void xsum_large::add_lchunk_to_small(xsum_expint const ix) {
  xsum_expint const count = _lacc->count[ix];

  /* Add to the small accumulator only if the count is not -1, which
     indicates a chunk that contains nothing yet. */
  if (count >= 0) {
    /* Propagate carries in the small accumulator if necessary. */

    if (_lacc->sacc.adds_until_propagate == 0) {
      carry_propagate();
    }

    /* Get the chunk we will add.  Note that this chunk is the integer sum
       of entire 64-bit floating-point representations, with sign, exponent,
       and mantissa, but we want only the sum of the mantissas. */

    xsum_lchunk chunk = _lacc->chunk[ix];

    if (xsum_debug) {
      std::cout << "Adding chunk " << static_cast<int>(ix)
                << " to small accumulator (count " << static_cast<int>(count)
                << ", chunk " << std::hex << std::setfill('0') << std::setw(16)
                << static_cast<long long>(chunk) << ")\n";
    }

    /* If we added the maximum number of values to 'chunk', the sum of
       the sign and exponent parts (all the same, equal to the index) will
       have overflowed out the top, leaving only the sum of the mantissas.
       If the count of how many more terms we could have summed is greater
       than zero, we therefore add this count times the index (shifted to
       the position of the sign and exponent) to get the unwanted bits to
       overflow out the top. */
    if (count > 0) {
      chunk += static_cast<xsum_lchunk>(count * ix) << XSUM_MANTISSA_BITS;
    }

    /* Find the exponent for this chunk from the low bits of the index,
       and split it into low and high parts, for accessing the small
       accumulator.  Noting that for denormalized numbers where the
       exponent part is zero, the actual exponent is 1 (before subtracting
       the bias), not zero. */

    xsum_expint const exp = ix & XSUM_EXP_MASK;
    xsum_expint const low_exp = ((exp == 0) ? 1 : (exp & XSUM_LOW_EXP_MASK));
    xsum_expint const high_exp = ((exp == 0) ? 0 : (exp >> XSUM_LOW_EXP_BITS));

    /* Split the mantissa into three parts, for three consecutive chunks in
       the small accumulator.  Except for denormalized numbers, add in the sum
       of all the implicit 1 bits that are above the actual mantissa bits. */

    xsum_uint const low_chunk = (chunk << low_exp) & XSUM_LOW_MANTISSA_MASK;
    xsum_uint mid_chunk = chunk >> (XSUM_LOW_MANTISSA_BITS - low_exp);

    /* normalized */
    if (exp != 0) {
      mid_chunk += static_cast<xsum_lchunk>((1 << XSUM_LCOUNT_BITS) - count)
                   << (XSUM_MANTISSA_BITS - XSUM_LOW_MANTISSA_BITS + low_exp);
    }

    xsum_uint const high_chunk = mid_chunk >> XSUM_LOW_MANTISSA_BITS;
    mid_chunk &= XSUM_LOW_MANTISSA_MASK;

    if (xsum_debug) {
      std::cout << "chunk div: low " << std::bitset<64>(low_chunk) << "\n"
                << "           mid " << std::bitset<64>(mid_chunk) << "\n"
                << "           high " << std::bitset<64>(high_chunk) << "\n";

      /* Add or subtract the three parts of the mantissa from three small
         accumulator chunks, according to the sign that is part of the index. */
      std::cout << "Small chunks " << static_cast<int>(high_exp) << ", "
                << static_cast<int>(high_exp) + 1 << ", "
                << static_cast<int>(high_exp) + 2
                << " before add or subtract:\n"
                << std::bitset<64>(_lacc->sacc.chunk[high_exp]) << "\n"
                << std::bitset<64>(_lacc->sacc.chunk[high_exp + 1]) << "\n"
                << std::bitset<64>(_lacc->sacc.chunk[high_exp + 2]) << "\n";
    }

    if (ix & (1 << XSUM_EXP_BITS)) {
      _lacc->sacc.chunk[high_exp] -= low_chunk;
      _lacc->sacc.chunk[high_exp + 1] -= mid_chunk;
      _lacc->sacc.chunk[high_exp + 2] -= high_chunk;
    } else {
      _lacc->sacc.chunk[high_exp] += low_chunk;
      _lacc->sacc.chunk[high_exp + 1] += mid_chunk;
      _lacc->sacc.chunk[high_exp + 2] += high_chunk;
    }

    if (xsum_debug) {
      std::cout << "Small chunks " << static_cast<int>(high_exp) << ", "
                << static_cast<int>(high_exp) + 1 << ", "
                << static_cast<int>(high_exp) + 2 << " after add or subtract:\n"
                << std::bitset<64>(_lacc->sacc.chunk[high_exp]) << "\n"
                << std::bitset<64>(_lacc->sacc.chunk[high_exp + 1]) << "\n"
                << std::bitset<64>(_lacc->sacc.chunk[high_exp + 2]) << "\n";
    }

    /* The above additions/subtractions reduce by one the number we can
       do before we need to do carry propagation again. */
    _lacc->sacc.adds_until_propagate -= 1;
  }

  /* We now clear the chunk to zero, and set the count to the number
     of adds we can do before the mantissa would overflow.  We also
     set the bit in chunks_used to indicate that this chunk is in use
     (if that is enabled). */

  _lacc->chunk[ix] = 0;
  _lacc->count[ix] = 1 << XSUM_LCOUNT_BITS;
  _lacc->chunks_used[ix >> 6] |= static_cast<xsum_used>(1) << (ix & 0x3f);
  _lacc->used_used |= static_cast<xsum_used>(1) << (ix >> 6);
}

inline void xsum_large::add_value_inf_nan(xsum_expint const ix,
                                          xsum_lchunk const uintv) {
  if ((ix & XSUM_EXP_MASK) == XSUM_EXP_MASK) {
    add_inf_nan(uintv);
  } else {
    add_lchunk_to_small(ix);
    _lacc->count[ix] -= 1;
    _lacc->chunk[ix] += uintv;
  }
}

void xsum_large::add_inf_nan(xsum_int const ivalue) {
  xsum_int const mantissa = ivalue & XSUM_MANTISSA_MASK;

  /* Inf */
  if (mantissa == 0) {
    if (_lacc->sacc.Inf == 0) {
      /* no previous Inf */
      _lacc->sacc.Inf = ivalue;
    } else if (_lacc->sacc.Inf != ivalue) {
      fpunion u;

      /* previous Inf was opposite sign */
      u.intv = ivalue;

      /* result will be a NaN */
      u.fltv = u.fltv - u.fltv;

      _lacc->sacc.Inf = u.intv;
    }
  }
  /* NaN */
  else {
    /* Choose the NaN with the bigger payload and clear its sign.  Using <=
       ensures that we will choose the first NaN over the previous zero. */
    if ((_lacc->sacc.NaN & XSUM_MANTISSA_MASK) <= mantissa) {
      _lacc->sacc.NaN = ivalue & ~XSUM_SIGN_MASK;
    }
  }
}

inline xsum_large_accumulator *xsum_large::get() const noexcept {
  return _lacc.get();
}

// AUXILLARY

int xsum_carry_propagate(xsum_small_accumulator *const sacc) {
  /* Set u to the index of the uppermost non-zero (for now) chunk, or
     return with value 0 if there is none. */

  bool found = false;
  int u = XSUM_SCHUNKS - 1;
  switch (XSUM_SCHUNKS & 0x3) {
    case 3: {
      if (sacc->chunk[u] != 0) {
        found = true;
        break;
      }
      --u;
    }
    case 2: {
      if (sacc->chunk[u] != 0) {
        found = true;
        break;
      }
      --u;
    }
    case 1: {
      if (sacc->chunk[u] != 0) {
        found = true;
        break;
      }
      --u;
    }
    case 0:;
  }
  do {
    if (sacc->chunk[u - 3] | sacc->chunk[u - 2] | sacc->chunk[u - 1] |
        sacc->chunk[u]) {
      found = true;
      break;
    }
    u -= 4;
  } while (u >= 0);

  if (found) {
    while (sacc->chunk[u] == 0) {
      --u;
    }
  } else {
    sacc->adds_until_propagate = XSUM_SMALL_CARRY_TERMS - 1;
    /* Return index of uppermost non-zero chunk. */
    return 0;
  }

  // xsum_schunk c, clow, chigh;
  // int i, u, uix;

  /* Carry propagate, starting at the low-order chunks.  Note that the
   loop limit of u may be increased inside the loop. */

  /* indicates that a non-zero chunk has not been found yet */
  int uix = -1;

  /* Quickly skip over unused low-order chunks.  Done here at the start
     on the theory that there are often many unused low-order chunks,
     justifying some overhead to begin, but later stretches of unused
     chunks may not be as large. */
  int i = 0;
  int const e = u - 3;
  while (i <= e) {
    if (sacc->chunk[i] | sacc->chunk[i + 1] | sacc->chunk[i + 2] |
        sacc->chunk[i + 3]) {
      break;
    }
    i += 4;
  }

  xsum_schunk c;

  do {
    /* Find the next non-zero chunk, or break out of loop if there is none. */
    bool nonzero = false;

    do {
      c = sacc->chunk[i];
      if (c != 0) {
        nonzero = true;
        break;
      }
      ++i;

      if (i > u) {
        break;
      }

      c = sacc->chunk[i];
      if (c != 0) {
        nonzero = true;
        break;
      }
      ++i;
    } while (i <= u);

    if (!nonzero) {
      break;
    }

    /* Propagate possible carry from this chunk to next chunk up. */
    xsum_schunk const chigh = c >> XSUM_LOW_MANTISSA_BITS;

    if (chigh == 0) {
      uix = i;
      ++i;
      /* no need to change this chunk */
      continue;
    }

    if (u == i) {
      if (chigh == -1) {
        uix = i;
        /* don't propagate -1 into the region of all zeros above */
        break;
      }

      /* we will change chunk[u+1], so we'll need to look at it */
      u = i + 1;
    }

    xsum_schunk const clow = c & XSUM_LOW_MANTISSA_MASK;
    if (clow != 0) {
      uix = i;
    }

    /* We now change chunk[i] and add to chunk[i+1]. Note that i+1 should be
       in range (no bigger than XSUM_CHUNKS-1) because the number of chunks
       is big enough to hold any sum, and we do not store redundant chunks
       with values 0 or -1 above previously non-zero chunks. */

    if (xsum_debug && i + 1 >= XSUM_SCHUNKS) {
      std::abort();
    }
    sacc->chunk[i++] = clow;
    sacc->chunk[i] += chigh;
  } while (i <= u);

  /* Check again for the number being zero, since carry propagation might
     have created zero from something that initially looked non-zero. */
  if (uix < 0) {
    sacc->adds_until_propagate = XSUM_SMALL_CARRY_TERMS - 1;
    /* Return index of uppermost non-zero chunk. */
    return 0;
  }

  /* While the uppermost chunk is negative, with value -1, combine it with
     the chunk below (if there is one) to produce the same number but with
     one fewer non-zero chunks. */

  while (sacc->chunk[uix] == -1 && uix > 0) {
    /* A shift of a negative number is undefined according to the standard, so
       do a multiply - it's all presumably constant-folded by the compiler. */
    sacc->chunk[uix--] = 0;
    sacc->chunk[uix] += static_cast<xsum_schunk>(-1) *
                        (static_cast<xsum_schunk>(1) << XSUM_LOW_MANTISSA_BITS);
  }

  /* We can now add one less than the total allowed terms before the
   next carry propagate. */
  sacc->adds_until_propagate = XSUM_SMALL_CARRY_TERMS - 1;

  /* Return index of uppermost non-zero chunk. */
  return uix;
}

inline void xsum_small_add_inf_nan(xsum_small_accumulator *const sacc,
                                   xsum_int const ivalue) {
  xsum_int const mantissa = ivalue & XSUM_MANTISSA_MASK;

  /* Inf */
  if (mantissa == 0) {
    /* no previous Inf */
    if (sacc->Inf == 0) {
      sacc->Inf = ivalue;
    }
    /* previous Inf was opposite sign */
    else if (sacc->Inf != ivalue) {
      fpunion u;
      u.intv = ivalue;

      /* result will be a NaN */
      u.fltv = u.fltv - u.fltv;

      sacc->Inf = u.intv;
    }
  }
  /* NaN */
  else {
    /* Choose the NaN with the bigger payload and clear its sign. Using <=
       ensures that we will choose the first NaN over the previous zero. */
    if ((sacc->NaN & XSUM_MANTISSA_MASK) <= mantissa) {
      sacc->NaN = ivalue & ~XSUM_SIGN_MASK;
    }
  }
}

inline void xsum_add_no_carry(xsum_small_accumulator *const sacc,
                              xsum_flt const value) {
  /* Extract exponent and mantissa. */
  fpunion u;
  u.fltv = value;

  xsum_int const ivalue = u.intv;
  xsum_int mantissa = ivalue & XSUM_MANTISSA_MASK;
  xsum_expint exp = (ivalue >> XSUM_MANTISSA_BITS) & XSUM_EXP_MASK;

  /* Categorize number as normal, denormalized, or Inf/NaN according to
     the value of the exponent field. */

  /* normalized */
  /* OR in implicit 1 bit at top of mantissa */
  if (exp != 0 && exp != XSUM_EXP_MASK) {
    mantissa |= static_cast<xsum_int>(1) << XSUM_MANTISSA_BITS;
  }
  /* zero or denormalized */
  /* If it's a zero (positive or negative), we do nothing. */
  else if (exp == 0) {
    if (mantissa == 0) {
      return;
    }
    /* Denormalized mantissa has no implicit 1, but exponent is 1 not 0. */
    exp = 1;
  }
  /* Inf or NaN */
  /* Just update flags in accumulator structure. */
  else {
    xsum_small_add_inf_nan(sacc, ivalue);
    return;
  }

  /* Separate high part of exponent, used as index of chunk, and low
     part of exponent, giving position within chunk. */

  xsum_expint const low_exp = exp & XSUM_LOW_EXP_MASK;
  xsum_expint const high_exp = exp >> XSUM_LOW_EXP_BITS;

  xsum_schunk *chunk_ptr = sacc->chunk + high_exp;

  xsum_schunk const chunk0 = chunk_ptr[0];
  xsum_schunk const chunk1 = chunk_ptr[1];

  /* Separate mantissa into two parts, after shifting, and add to (or
     subtract from) this chunk and the next higher chunk (which always
     exists since there are three extra ones at the top). */
  xsum_int const low_mantissa =
      (static_cast<xsum_uint>(mantissa) << low_exp) & XSUM_LOW_MANTISSA_MASK;
  xsum_int const high_mantissa = mantissa >> (XSUM_LOW_MANTISSA_BITS - low_exp);

  /* Add or subtract to or from the two affected chunks. */
  if (ivalue < 0) {
    chunk_ptr[0] = chunk0 - low_mantissa;
    chunk_ptr[1] = chunk1 - high_mantissa;
  } else {
    chunk_ptr[0] = chunk0 + low_mantissa;
    chunk_ptr[1] = chunk1 + high_mantissa;
  }
}

inline void xsum_add_no_carry(xsum_small_accumulator *const sacc,
                              xsum_small_accumulator const *const value) {
  if (value->Inf != 0) {
    /* no previous Inf */
    if (sacc->Inf == 0) {
      sacc->Inf = value->Inf;
    }
    /* previous Inf had opposite sign */
    else if (sacc->Inf != value->Inf) {
      fpunion u;
      u.intv = value->Inf;

      /* result will be a NaN */
      u.fltv = u.fltv - u.fltv;

      sacc->Inf = u.intv;
    }
    return;
  }
  if (value->NaN != 0 || sacc->NaN != 0) {
    if (value->NaN != 0) {
      if ((sacc->NaN & XSUM_MANTISSA_MASK) <
          (value->NaN & XSUM_MANTISSA_MASK)) {
        sacc->NaN = value->NaN;
      }
    }
    return;
  }
  xsum_schunk *sacc_chunk = sacc->chunk;
  xsum_schunk const *const value_chunk = value->chunk;
  for (int i = 0; i < XSUM_SCHUNKS; ++i) {
    sacc_chunk[i] += value_chunk[i];
  }
}

inline void xsum_add_no_carry(xsum_small_accumulator *const sacc,
                              xsum_flt const *const vec, xsum_length const n) {
  xsum_flt const *const f = vec;
  for (int i = 0; i < n - 1; ++i) {
    xsum_flt const g = f[i];
    xsum_add_no_carry(sacc, g);
  }
}

inline void xsum_add_no_carry(xsum_small_accumulator *const sacc,
                              xsum_small_accumulator const *const vec,
                              xsum_length const n) {
  xsum_small_accumulator const *const f = vec;
  for (int i = 0; i < n - 1; ++i) {
    xsum_add_no_carry(sacc, &f[i]);
  }
}

inline void xsum_add_sqnorm_no_carry(xsum_small_accumulator *const sacc,
                                     xsum_flt const *const vec,
                                     xsum_length const n) {
  xsum_flt const *v = vec;
  for (xsum_length i = 0; i < n - 1; ++i, ++v) {
    xsum_flt const f = *v;
    xsum_flt const g = f * f;
    xsum_add_no_carry(sacc, g);
  }
}

inline void xsum_add_dot_no_carry(xsum_small_accumulator *const sacc,
                                  xsum_flt const *const vec1,
                                  xsum_flt const *const vec2,
                                  xsum_length const n) {
  xsum_flt const *v1 = vec1;
  xsum_flt const *v2 = vec2;
  for (xsum_length i = 0; i < n - 1; ++i, ++v1, ++v2) {
    xsum_flt const f1 = *v1;
    xsum_flt const f2 = *v2;
    xsum_flt const g = f1 * f2;
    xsum_add_no_carry(sacc, g);
  }
}

template <>
void xsum_init<xsum_small_accumulator>(xsum_small_accumulator *const sacc) {
  std::fill(sacc->chunk, sacc->chunk + XSUM_SCHUNKS, 0);
  sacc->Inf = 0;
  sacc->NaN = 0;
  sacc->adds_until_propagate = XSUM_SMALL_CARRY_TERMS;
}

template <>
void xsum_init<xsum_large_accumulator>(xsum_large_accumulator *const lacc) {
  std::fill(lacc->count, lacc->count + XSUM_LCHUNKS, -1);
  std::fill(lacc->chunks_used, lacc->chunks_used + XSUM_LCHUNKS / 64, 0);
  lacc->used_used = 0;
  std::fill(lacc->sacc.chunk, lacc->sacc.chunk + XSUM_SCHUNKS, 0);
  lacc->sacc.Inf = 0;
  lacc->sacc.NaN = 0;
  lacc->sacc.adds_until_propagate = XSUM_SMALL_CARRY_TERMS;
}

template <>
void xsum_add<xsum_small_accumulator>(xsum_small_accumulator *const sacc,
                                      xsum_flt const value) {
  if (sacc->adds_until_propagate == 0) {
    xsum_carry_propagate(sacc);
  }

  xsum_add_no_carry(sacc, value);

  --sacc->adds_until_propagate;
}

template <>
void xsum_add<xsum_large_accumulator>(xsum_large_accumulator *const lacc,
                                      xsum_flt const value) {
  fpunion u;
  u.fltv = value;
  xsum_expint const ix = u.uintv >> XSUM_MANTISSA_BITS;
  int const count = lacc->count[ix] - 1;
  if (count < 0) {
    xsum_large_add_value_inf_nan(lacc, ix, u.uintv);
  } else {
    lacc->count[ix] = count;
    lacc->chunk[ix] += u.uintv;
  }
}

template <>
void xsum_add<xsum_small_accumulator>(xsum_small_accumulator *const sacc,
                                      xsum_flt const *const vec,
                                      xsum_length const c) {
  if (c == 0) {
    return;
  }
  xsum_flt const *v = vec;
  xsum_length n = c;
  while (n > 1) {
    if (sacc->adds_until_propagate == 0) {
      xsum_carry_propagate(sacc);
    }
    xsum_length const m = (n - 1 <= sacc->adds_until_propagate)
                              ? n - 1
                              : sacc->adds_until_propagate;
    xsum_add_no_carry(sacc, v, m + 1);
    sacc->adds_until_propagate -= m;
    v += m;
    n -= m;
  }
  xsum_add(sacc, *v);
}

template <>
void xsum_add<xsum_large_accumulator>(xsum_large_accumulator *const lacc,
                                      xsum_flt const *const vec,
                                      xsum_length const n) {
  if (n == 0) {
    return;
  }

  xsum_flt const *v = vec;

  /* Version that's been manually optimized:  Loop unrolled, pre-fetch
     attempted, branches eliminated, ... */

  /* Unrolled loop processing two values each time around.  The loop is
     done as two nested loops, arranged so that the inner one will have
     no branches except for the one looping back.  This is achieved by
     a trick for combining three tests for negativity into one.  The
     last one or two values are not done here, so that the pre-fetching
     will not go past the end of the array (which would probably be OK,
     but is technically not allowed). */

  fpunion u1;
  fpunion u2;

  int count1;
  int count2;

  xsum_expint ix1;
  xsum_expint ix2;

  /* leave out last one or two, terminate when negative, for trick */
  xsum_length m = n - 3;

  while (m >= 0) {
    /* Loop processing two values at a time until we're done, or until
       one (or both) of the values result in a chunk needing to be processed.
       Updates are done here for both of these chunks, even though it is not
       yet known whether these updates ought to have been done.  We hope
       this allows for better memory pre-fetch and instruction scheduling. */

    for (;;) {
      u1.fltv = *v++;
      u2.fltv = *v++;

      ix1 = u1.uintv >> XSUM_MANTISSA_BITS;
      count1 = lacc->count[ix1] - 1;
      lacc->count[ix1] = count1;
      lacc->chunk[ix1] += u1.uintv;

      ix2 = u2.uintv >> XSUM_MANTISSA_BITS;
      count2 = lacc->count[ix2] - 1;
      lacc->count[ix2] = count2;
      lacc->chunk[ix2] += u2.uintv;

      m -= 2;

      /* ... equivalent to while (count1 >= 0 && count2 >= 0 && m >= 0) */
      if ((static_cast<xsum_length>(count1) | static_cast<xsum_length>(count2) |
           m) < 0) {
        break;
      }
    }

    /* See if we were actually supposed to update these chunks.  If not,
       back out the changes and then process the chunks as they ought to
       have been processed. */
    if (count1 < 0 || count2 < 0) {
      lacc->count[ix2] = count2 + 1;
      lacc->chunk[ix2] -= u2.uintv;

      if (count1 < 0) {
        lacc->count[ix1] = count1 + 1;
        lacc->chunk[ix1] -= u1.uintv;
        xsum_large_add_value_inf_nan(lacc, ix1, u1.uintv);
        count2 = lacc->count[ix2] - 1;
      }

      if (count2 < 0) {
        xsum_large_add_value_inf_nan(lacc, ix2, u2.uintv);
      } else {
        lacc->count[ix2] = count2;
        lacc->chunk[ix2] += u2.uintv;
      }
    }
  }

  /* Process the last one or two values, without pre-fetching. */
  m += 3;
  for (;;) {
    u1.fltv = *v++;
    ix1 = u1.uintv >> XSUM_MANTISSA_BITS;
    count1 = lacc->count[ix1] - 1;

    if (count1 < 0) {
      xsum_large_add_value_inf_nan(lacc, ix1, u1.uintv);
    } else {
      lacc->count[ix1] = count1;
      lacc->chunk[ix1] += u1.uintv;
    }

    --m;
    if (m == 0) {
      break;
    }
  }
}

template <>
void xsum_add<xsum_small_accumulator>(xsum_small_accumulator *const sacc,
                                      std::vector<xsum_flt> const &vec) {
  xsum_length n = static_cast<xsum_length>(vec.size());
  if (n == 0) {
    return;
  }

  xsum_flt const *v = vec.data();

  while (n > 1) {
    if (sacc->adds_until_propagate == 0) {
      xsum_carry_propagate(sacc);
    }
    xsum_length const m = (n - 1 <= sacc->adds_until_propagate)
                              ? n - 1
                              : sacc->adds_until_propagate;
    xsum_add_no_carry(sacc, v, m + 1);
    sacc->adds_until_propagate -= m;
    v += m;
    n -= m;
  }
  xsum_add(sacc, *v);
}

template <>
void xsum_add<xsum_large_accumulator>(xsum_large_accumulator *const lacc,
                                      std::vector<xsum_flt> const &vec) {
  xsum_length const n = static_cast<xsum_length>(vec.size());
  if (n == 0) {
    return;
  }

  xsum_flt const *v = vec.data();

  /* Version that's been manually optimized:  Loop unrolled, pre-fetch
     attempted, branches eliminated, ... */

  /* Unrolled loop processing two values each time around.  The loop is
     done as two nested loops, arranged so that the inner one will have
     no branches except for the one looping back.  This is achieved by
     a trick for combining three tests for negativity into one.  The
     last one or two values are not done here, so that the pre-fetching
     will not go past the end of the array (which would probably be OK,
     but is technically not allowed). */

  fpunion u1;
  fpunion u2;

  int count1;
  int count2;

  xsum_expint ix1;
  xsum_expint ix2;

  /* leave out last one or two, terminate when negative, for trick */
  xsum_length m = n - 3;

  while (m >= 0) {
    /* Loop processing two values at a time until we're done, or until
       one (or both) of the values result in a chunk needing to be processed.
       Updates are done here for both of these chunks, even though it is not
       yet known whether these updates ought to have been done.  We hope
       this allows for better memory pre-fetch and instruction scheduling. */

    for (;;) {
      u1.fltv = *v++;
      u2.fltv = *v++;

      ix1 = u1.uintv >> XSUM_MANTISSA_BITS;
      count1 = lacc->count[ix1] - 1;
      lacc->count[ix1] = count1;
      lacc->chunk[ix1] += u1.uintv;

      ix2 = u2.uintv >> XSUM_MANTISSA_BITS;
      count2 = lacc->count[ix2] - 1;
      lacc->count[ix2] = count2;
      lacc->chunk[ix2] += u2.uintv;

      m -= 2;

      /* ... equivalent to while (count1 >= 0 && count2 >= 0 && m >= 0) */
      if ((static_cast<xsum_length>(count1) | static_cast<xsum_length>(count2) |
           m) < 0) {
        break;
      }
    }

    /* See if we were actually supposed to update these chunks.  If not,
       back out the changes and then process the chunks as they ought to
       have been processed. */
    if (count1 < 0 || count2 < 0) {
      lacc->count[ix2] = count2 + 1;
      lacc->chunk[ix2] -= u2.uintv;

      if (count1 < 0) {
        lacc->count[ix1] = count1 + 1;
        lacc->chunk[ix1] -= u1.uintv;
        xsum_large_add_value_inf_nan(lacc, ix1, u1.uintv);
        count2 = lacc->count[ix2] - 1;
      }

      if (count2 < 0) {
        xsum_large_add_value_inf_nan(lacc, ix2, u2.uintv);
      } else {
        lacc->count[ix2] = count2;
        lacc->chunk[ix2] += u2.uintv;
      }
    }
  }

  /* Process the last one or two values, without pre-fetching. */
  m += 3;
  for (;;) {
    u1.fltv = *v++;
    ix1 = u1.uintv >> XSUM_MANTISSA_BITS;
    count1 = lacc->count[ix1] - 1;

    if (count1 < 0) {
      xsum_large_add_value_inf_nan(lacc, ix1, u1.uintv);
    } else {
      lacc->count[ix1] = count1;
      lacc->chunk[ix1] += u1.uintv;
    }

    --m;
    if (m == 0) {
      break;
    }
  }
}

void xsum_add(xsum_small_accumulator *const sacc,
              xsum_small_accumulator const *const value) {
  if (sacc->adds_until_propagate == 0) {
    xsum_carry_propagate(sacc);
  }

  xsum_add_no_carry(sacc, value);

  --sacc->adds_until_propagate;
}

void xsum_add(xsum_large_accumulator *const lacc,
              xsum_small_accumulator const *const value) {
  if (lacc->sacc.adds_until_propagate == 0) {
    xsum_carry_propagate(&lacc->sacc);
  }

  xsum_add_no_carry(&lacc->sacc, value);

  --lacc->sacc.adds_until_propagate;
}

void xsum_add(xsum_large_accumulator *const lacc,
              xsum_large_accumulator *const value) {
  xsum_small_accumulator *sacc = xsum_round_to_small(value);
  xsum_add(lacc, sacc);
}

void xsum_add(xsum_small_accumulator *const sacc,
              xsum_small_accumulator const *const vec, xsum_length const c) {
  if (c == 0) {
    return;
  }
  xsum_small_accumulator const *v = vec;
  xsum_length n = c;
  while (n > 1) {
    if (sacc->adds_until_propagate == 0) {
      xsum_carry_propagate(sacc);
    }
    xsum_length const m = (n - 1 <= sacc->adds_until_propagate)
                              ? n - 1
                              : sacc->adds_until_propagate;
    xsum_add_no_carry(sacc, v, m + 1);
    sacc->adds_until_propagate -= m;
    v += m;
    n -= m;
  }
  xsum_add(sacc, v);
}

template <>
void xsum_add_sqnorm<xsum_small_accumulator>(xsum_small_accumulator *const sacc,
                                             xsum_flt const *const vec,
                                             xsum_length const c) {
  if (c == 0) {
    return;
  }

  xsum_flt const *v = vec;
  xsum_length n = c;
  while (n > 1) {
    if (sacc->adds_until_propagate == 0) {
      xsum_carry_propagate(sacc);
    }

    xsum_length const m = n - 1 <= sacc->adds_until_propagate
                              ? n - 1
                              : sacc->adds_until_propagate;
    xsum_add_sqnorm_no_carry(sacc, v, m + 1);
    sacc->adds_until_propagate -= m;
    v += m;
    n -= m;
  }

  xsum_add(sacc, *v * *v);
}

template <>
void xsum_add_sqnorm<xsum_large_accumulator>(xsum_large_accumulator *const lacc,
                                             xsum_flt const *const vec,
                                             xsum_length const n) {
  if (n == 0) {
    return;
  }

  xsum_flt const *v = vec;

  /* Unrolled loop processing two squares each time around.  The loop is
     done as two nested loops, arranged so that the inner one will have
     no branches except for the one looping back.  This is achieved by
     a trick for combining three tests for negativity into one.  The
     last one or two squares are not done here, so that the pre-fetching
     will not go past the end of the array (which would probably be OK,
     but is technically not allowed). */

  fpunion u1;
  fpunion u2;

  int count1;
  int count2;

  xsum_expint ix1;
  xsum_expint ix2;

  /* leave out last one or two, terminate when negative, for trick */
  xsum_length m = n - 3;
  while (m >= 0) {
    /* Loop processing two squares at a time until we're done, or until
       one (or both) of them result in a chunk needing to be processed.
       Updates are done here for both of these chunks, even though it is not
       yet known whether these updates ought to have been done.  We hope
       this allows for better memory pre-fetch and instruction scheduling. */

    for (;;) {
      u1.fltv = *v * *v;
      ++v;
      u2.fltv = *v * *v;
      ++v;

      ix1 = u1.uintv >> XSUM_MANTISSA_BITS;
      count1 = lacc->count[ix1] - 1;
      lacc->count[ix1] = count1;
      lacc->chunk[ix1] += u1.uintv;

      ix2 = u2.uintv >> XSUM_MANTISSA_BITS;
      count2 = lacc->count[ix2] - 1;
      lacc->count[ix2] = count2;
      lacc->chunk[ix2] += u2.uintv;

      m -= 2;

      /* ... equivalent to while (count1 >= 0 && count2 >= 0 && m >= 0) */
      /* ... while (((xsum_length)count1 | (xsum_length)count2 | m) >= 0); */
      if ((static_cast<xsum_length>(count1) | static_cast<xsum_length>(count2) |
           m) < 0) {
        break;
      }
    }

    /* See if we were actually supposed to update these chunks.  If not,
       back out the changes and then process the chunks as they ought to
       have been processed. */
    if (count1 < 0 || count2 < 0) {
      lacc->count[ix2] = count2 + 1;
      lacc->chunk[ix2] -= u2.uintv;

      if (count1 < 0) {
        lacc->count[ix1] = count1 + 1;
        lacc->chunk[ix1] -= u1.uintv;
        xsum_large_add_value_inf_nan(lacc, ix1, u1.uintv);
        count2 = lacc->count[ix2] - 1;
      }

      if (count2 < 0) {
        xsum_large_add_value_inf_nan(lacc, ix2, u2.uintv);
      } else {
        lacc->count[ix2] = count2;
        lacc->chunk[ix2] += u2.uintv;
      }
    }
  }

  /* Process the last one or two squares, without pre-fetching. */
  m += 3;
  for (;;) {
    u1.fltv = *v * *v;
    ++v;

    ix1 = u1.uintv >> XSUM_MANTISSA_BITS;
    count1 = lacc->count[ix1] - 1;
    if (count1 < 0) {
      xsum_large_add_value_inf_nan(lacc, ix1, u1.uintv);
    } else {
      lacc->count[ix1] = count1;
      lacc->chunk[ix1] += u1.uintv;
    }

    --m;
    if (m == 0) {
      break;
    }
  }
}

template <>
void xsum_add_sqnorm<xsum_small_accumulator>(xsum_small_accumulator *const sacc,
                                             std::vector<xsum_flt> const &vec) {
  xsum_length n = static_cast<xsum_length>(vec.size());
  if (n == 0) {
    return;
  }

  xsum_flt const *v = vec.data();

  while (n > 1) {
    if (sacc->adds_until_propagate == 0) {
      xsum_carry_propagate(sacc);
    }

    xsum_length const m = n - 1 <= sacc->adds_until_propagate
                              ? n - 1
                              : sacc->adds_until_propagate;
    xsum_add_sqnorm_no_carry(sacc, v, m + 1);
    sacc->adds_until_propagate -= m;
    v += m;
    n -= m;
  }

  xsum_add(sacc, *v * *v);
}

template <>
void xsum_add_sqnorm<xsum_large_accumulator>(xsum_large_accumulator *const lacc,
                                             std::vector<xsum_flt> const &vec) {
  xsum_length const n = static_cast<xsum_length>(vec.size());
  if (n == 0) {
    return;
  }

  xsum_flt const *v = vec.data();

  /* Unrolled loop processing two squares each time around.  The loop is
     done as two nested loops, arranged so that the inner one will have
     no branches except for the one looping back.  This is achieved by
     a trick for combining three tests for negativity into one.  The
     last one or two squares are not done here, so that the pre-fetching
     will not go past the end of the array (which would probably be OK,
     but is technically not allowed). */

  fpunion u1;
  fpunion u2;

  int count1;
  int count2;

  xsum_expint ix1;
  xsum_expint ix2;

  /* leave out last one or two, terminate when negative, for trick */
  xsum_length m = n - 3;
  while (m >= 0) {
    /* Loop processing two squares at a time until we're done, or until
       one (or both) of them result in a chunk needing to be processed.
       Updates are done here for both of these chunks, even though it is not
       yet known whether these updates ought to have been done.  We hope
       this allows for better memory pre-fetch and instruction scheduling. */

    for (;;) {
      u1.fltv = *v * *v;
      ++v;
      u2.fltv = *v * *v;
      ++v;

      ix1 = u1.uintv >> XSUM_MANTISSA_BITS;
      count1 = lacc->count[ix1] - 1;
      lacc->count[ix1] = count1;
      lacc->chunk[ix1] += u1.uintv;

      ix2 = u2.uintv >> XSUM_MANTISSA_BITS;
      count2 = lacc->count[ix2] - 1;
      lacc->count[ix2] = count2;
      lacc->chunk[ix2] += u2.uintv;

      m -= 2;

      /* ... equivalent to while (count1 >= 0 && count2 >= 0 && m >= 0) */
      /* ... while (((xsum_length)count1 | (xsum_length)count2 | m) >= 0); */
      if ((static_cast<xsum_length>(count1) | static_cast<xsum_length>(count2) |
           m) < 0) {
        break;
      }
    }

    /* See if we were actually supposed to update these chunks.  If not,
       back out the changes and then process the chunks as they ought to
       have been processed. */
    if (count1 < 0 || count2 < 0) {
      lacc->count[ix2] = count2 + 1;
      lacc->chunk[ix2] -= u2.uintv;

      if (count1 < 0) {
        lacc->count[ix1] = count1 + 1;
        lacc->chunk[ix1] -= u1.uintv;
        xsum_large_add_value_inf_nan(lacc, ix1, u1.uintv);
        count2 = lacc->count[ix2] - 1;
      }

      if (count2 < 0) {
        xsum_large_add_value_inf_nan(lacc, ix2, u2.uintv);
      } else {
        lacc->count[ix2] = count2;
        lacc->chunk[ix2] += u2.uintv;
      }
    }
  }

  /* Process the last one or two squares, without pre-fetching. */
  m += 3;
  for (;;) {
    u1.fltv = *v * *v;
    ++v;

    ix1 = u1.uintv >> XSUM_MANTISSA_BITS;
    count1 = lacc->count[ix1] - 1;
    if (count1 < 0) {
      xsum_large_add_value_inf_nan(lacc, ix1, u1.uintv);
    } else {
      lacc->count[ix1] = count1;
      lacc->chunk[ix1] += u1.uintv;
    }

    --m;
    if (m == 0) {
      break;
    }
  }
}

template <>
void xsum_add_dot<xsum_small_accumulator>(xsum_small_accumulator *const sacc,
                                          xsum_flt const *const vec1,
                                          xsum_flt const *const vec2,
                                          xsum_length const c) {
  if (c == 0) {
    return;
  }

  xsum_flt const *v1 = vec1;
  xsum_flt const *v2 = vec2;
  xsum_length n = c;

  while (n > 1) {
    if (sacc->adds_until_propagate == 0) {
      xsum_carry_propagate(sacc);
    }
    xsum_length const m = n - 1 <= sacc->adds_until_propagate
                              ? n - 1
                              : sacc->adds_until_propagate;
    xsum_add_dot_no_carry(sacc, v1, v2, m + 1);
    sacc->adds_until_propagate -= m;
    v1 += m;
    v2 += m;
    n -= m;
  }

  xsum_add(sacc, *v1 * *v2);
}

template <>
void xsum_add_dot<xsum_large_accumulator>(xsum_large_accumulator *const lacc,
                                          xsum_flt const *const vec1,
                                          xsum_flt const *const vec2,
                                          xsum_length const n) {
  if (n == 0) {
    return;
  }

  xsum_flt const *v1 = vec1;
  xsum_flt const *v2 = vec2;

  /* Version that's been manually optimized:  Loop unrolled, pre-fetch
     attempted, branches eliminated, ... */

  // union fpunion u1, u2;
  // int count1, count2;
  // xsum_expint ix1, ix2;
  // const xsum_flt *v1, *v2;
  // xsum_flt f1, f2;
  // xsum_length m;

  fpunion u1;
  fpunion u2;

  int count1;
  int count2;

  xsum_expint ix1;
  xsum_expint ix2;

  /* Unrolled loop processing two products each time around.  The loop is
     done as two nested loops, arranged so that the inner one will have
     no branches except for the one looping back.  This is achieved by
     a trick for combining three tests for negativity into one.  The
     last one or two products are not done here, so that the pre-fetching
     will not go past the end of the array (which would probably be OK,
     but is technically not allowed). */

  /* leave out last one or two, terminate when negative, for trick */
  xsum_length m = n - 3;
  while (m >= 0) {
    /* Loop processing two products at a time until we're done, or until
       one (or both) of them result in a chunk needing to be processed.
       Updates are done here for both of these chunks, even though it is not
       yet known whether these updates ought to have been done.  We hope
       this allows for better memory pre-fetch and instruction scheduling. */

    for (;;) {
      u1.fltv = *v1 * *v2;
      ++v1;
      ++v2;
      u2.fltv = *v1 * *v2;
      ++v1;
      ++v2;

      ix1 = u1.uintv >> XSUM_MANTISSA_BITS;
      count1 = lacc->count[ix1] - 1;
      lacc->count[ix1] = count1;
      lacc->chunk[ix1] += u1.uintv;

      ix2 = u2.uintv >> XSUM_MANTISSA_BITS;
      count2 = lacc->count[ix2] - 1;
      lacc->count[ix2] = count2;
      lacc->chunk[ix2] += u2.uintv;

      m -= 2;

      /* ... equivalent to while (count1 >= 0 && count2 >= 0 && m >= 0) */
      /* ... while (((xsum_length)count1 | (xsum_length)count2 | m) >= 0); */
      if ((static_cast<xsum_length>(count1) | static_cast<xsum_length>(count2) |
           m) < 0) {
        break;
      }
    }

    /* See if we were actually supposed to update these chunks.  If not,
       back out the changes and then process the chunks as they ought to
       have been processed. */

    if (count1 < 0 || count2 < 0) {
      lacc->count[ix2] = count2 + 1;
      lacc->chunk[ix2] -= u2.uintv;

      if (count1 < 0) {
        lacc->count[ix1] = count1 + 1;
        lacc->chunk[ix1] -= u1.uintv;
        xsum_large_add_value_inf_nan(lacc, ix1, u1.uintv);
        count2 = lacc->count[ix2] - 1;
      }

      if (count2 < 0) {
        xsum_large_add_value_inf_nan(lacc, ix2, u2.uintv);
      } else {
        lacc->count[ix2] = count2;
        lacc->chunk[ix2] += u2.uintv;
      }
    }
  }

  /* Process the last one or two products, without pre-fetching. */
  m += 3;
  for (;;) {
    u1.fltv = *v1 * *v2;
    ++v1;
    ++v2;

    ix1 = u1.uintv >> XSUM_MANTISSA_BITS;
    count1 = lacc->count[ix1] - 1;
    if (count1 < 0) {
      xsum_large_add_value_inf_nan(lacc, ix1, u1.uintv);
    } else {
      lacc->count[ix1] = count1;
      lacc->chunk[ix1] += u1.uintv;
    }

    --m;
    if (m == 0) {
      break;
    }
  }
}

template <>
void xsum_add_dot<xsum_small_accumulator>(xsum_small_accumulator *const sacc,
                                          std::vector<xsum_flt> const &vec1,
                                          std::vector<xsum_flt> const &vec2) {
  xsum_length n = static_cast<xsum_length>(vec1.size());
  if (n == 0 || n > static_cast<xsum_length>(vec2.size())) {
    return;
  }

  xsum_flt const *v1 = vec1.data();
  xsum_flt const *v2 = vec2.data();

  while (n > 1) {
    if (sacc->adds_until_propagate == 0) {
      xsum_carry_propagate(sacc);
    }
    xsum_length const m = n - 1 <= sacc->adds_until_propagate
                              ? n - 1
                              : sacc->adds_until_propagate;
    xsum_add_dot_no_carry(sacc, v1, v2, m + 1);
    sacc->adds_until_propagate -= m;
    v1 += m;
    v2 += m;
    n -= m;
  }

  xsum_add(sacc, *v1 * *v2);
}

template <>
void xsum_add_dot<xsum_large_accumulator>(xsum_large_accumulator *const lacc,
                                          std::vector<xsum_flt> const &vec1,
                                          std::vector<xsum_flt> const &vec2) {
  xsum_length const n = static_cast<xsum_length>(vec1.size());
  if (n == 0 || n > static_cast<xsum_length>(vec2.size())) {
    return;
  }

  xsum_flt const *v1 = vec1.data();
  xsum_flt const *v2 = vec2.data();

  /* Version that's been manually optimized:  Loop unrolled, pre-fetch
     attempted, branches eliminated, ... */

  // union fpunion u1, u2;
  // int count1, count2;
  // xsum_expint ix1, ix2;
  // const xsum_flt *v1, *v2;
  // xsum_flt f1, f2;
  // xsum_length m;

  fpunion u1;
  fpunion u2;

  int count1;
  int count2;

  xsum_expint ix1;
  xsum_expint ix2;

  /* Unrolled loop processing two products each time around.  The loop is
     done as two nested loops, arranged so that the inner one will have
     no branches except for the one looping back.  This is achieved by
     a trick for combining three tests for negativity into one.  The
     last one or two products are not done here, so that the pre-fetching
     will not go past the end of the array (which would probably be OK,
     but is technically not allowed). */

  /* leave out last one or two, terminate when negative, for trick */
  xsum_length m = n - 3;
  while (m >= 0) {
    /* Loop processing two products at a time until we're done, or until
       one (or both) of them result in a chunk needing to be processed.
       Updates are done here for both of these chunks, even though it is not
       yet known whether these updates ought to have been done.  We hope
       this allows for better memory pre-fetch and instruction scheduling. */

    for (;;) {
      u1.fltv = *v1 * *v2;
      ++v1;
      ++v2;
      u2.fltv = *v1 * *v2;
      ++v1;
      ++v2;

      ix1 = u1.uintv >> XSUM_MANTISSA_BITS;
      count1 = lacc->count[ix1] - 1;
      lacc->count[ix1] = count1;
      lacc->chunk[ix1] += u1.uintv;

      ix2 = u2.uintv >> XSUM_MANTISSA_BITS;
      count2 = lacc->count[ix2] - 1;
      lacc->count[ix2] = count2;
      lacc->chunk[ix2] += u2.uintv;

      m -= 2;

      /* ... equivalent to while (count1 >= 0 && count2 >= 0 && m >= 0) */
      /* ... while (((xsum_length)count1 | (xsum_length)count2 | m) >= 0); */
      if ((static_cast<xsum_length>(count1) | static_cast<xsum_length>(count2) |
           m) < 0) {
        break;
      }
    }

    /* See if we were actually supposed to update these chunks.  If not,
       back out the changes and then process the chunks as they ought to
       have been processed. */

    if (count1 < 0 || count2 < 0) {
      lacc->count[ix2] = count2 + 1;
      lacc->chunk[ix2] -= u2.uintv;

      if (count1 < 0) {
        lacc->count[ix1] = count1 + 1;
        lacc->chunk[ix1] -= u1.uintv;
        xsum_large_add_value_inf_nan(lacc, ix1, u1.uintv);
        count2 = lacc->count[ix2] - 1;
      }

      if (count2 < 0) {
        xsum_large_add_value_inf_nan(lacc, ix2, u2.uintv);
      } else {
        lacc->count[ix2] = count2;
        lacc->chunk[ix2] += u2.uintv;
      }
    }
  }

  /* Process the last one or two products, without pre-fetching. */
  m += 3;
  for (;;) {
    u1.fltv = *v1 * *v2;
    ++v1;
    ++v2;

    ix1 = u1.uintv >> XSUM_MANTISSA_BITS;
    count1 = lacc->count[ix1] - 1;
    if (count1 < 0) {
      xsum_large_add_value_inf_nan(lacc, ix1, u1.uintv);
    } else {
      lacc->count[ix1] = count1;
      lacc->chunk[ix1] += u1.uintv;
    }

    --m;
    if (m == 0) {
      break;
    }
  }
}

template <>
xsum_flt xsum_round<xsum_small_accumulator>(
    xsum_small_accumulator *const sacc) {
  fpunion u;

  /* See if we have a NaN from one of the numbers being a NaN, in which
     case we return the NaN with largest payload. */
  if (sacc->NaN != 0) {
    u.intv = sacc->NaN;
    return u.fltv;
  }

  /* Otherwise, if any number was infinite, we return +Inf, -Inf, or a Nan
     (if both +Inf and -Inf occurred).  Note that we do NOT return NaN if
     we have both an infinite number and a sum of other numbers that
     overflows with opposite sign, since there is no real ambiguity in
     such a case. */
  if (sacc->Inf != 0) {
    u.intv = sacc->Inf;
    return u.fltv;
  }

  /* If none of the numbers summed were infinite or NaN, we proceed to
     propagate carries, as a preliminary to finding the magnitude of
     the sum.  This also ensures that the sign of the result can be
     determined from the uppermost non-zero chunk.

     We also find the index, i, of this uppermost non-zero chunk, as
     the value returned by xsum_carry_propagate, and set ivalue to
     sacc->chunk[i].  Note that ivalue will not be 0 or -1, unless
     i is 0 (the lowest chunk), in which case it will be handled by
     the code for denormalized numbers. */

  int i = xsum_carry_propagate(sacc);

  xsum_int ivalue = sacc->chunk[i];

  /* Handle a possible denormalized number, including zero. */
  if (i <= 1) {
    /* Check for zero value, in which case we can return immediately. */
    if (ivalue == 0) {
      return 0.0;
    }

    /* Check if it is actually a denormalized number.  It always is if only
       the lowest chunk is non-zero.  If the highest non-zero chunk is the
       next-to-lowest, we check the magnitude of the absolute value.
       Note that the real exponent is 1 (not 0), so we need to shift right
       by 1 here, which also means there will be no overflow from the left
       shift below (but must view absolute value as unsigned). */
    if (i == 0) {
      u.intv = ivalue >= 0 ? ivalue : -ivalue;
      u.intv >>= 1;
      if (ivalue < 0) {
        u.intv |= XSUM_SIGN_MASK;
      }
      return u.fltv;
    }
    /* Note: Left shift of -ve number is undefined, so do a multiply instead,
       which is probably optimized to a shift. */
    else {
      u.intv =
          ivalue * (static_cast<xsum_int>(1) << (XSUM_LOW_MANTISSA_BITS - 1)) +
          (sacc->chunk[0] >> 1);
      if (u.intv < 0) {
        if (u.intv > -(static_cast<xsum_int>(1) << XSUM_MANTISSA_BITS)) {
          u.intv = (-u.intv) | XSUM_SIGN_MASK;
          return u.fltv;
        }
      } else {
        if (u.uintv < static_cast<xsum_int>(1) << XSUM_MANTISSA_BITS) {
          return u.fltv;
        }
      }
      /* otherwise, it's not actually denormalized, so fall through to below */
    }
  }

  /* Find the location of the uppermost 1 bit in the absolute value of the
     upper chunk by converting it (as a signed integer) to a floating point
     value, and looking at the exponent.  Then set 'more' to the number of
     bits from the lower chunk (and maybe the next lower) that are needed
     to fill out the mantissa of the result, plus an extra bit to help decide
     on rounding.  For negative numbers, it may turn out later that we need
     another bit because negating a negative value may carry out of the top
     here, but not once more bits are shifted into the bottom later on. */

  u.fltv = static_cast<xsum_flt>(ivalue);
  int e = (u.uintv >> XSUM_MANTISSA_BITS) & XSUM_EXP_MASK;
  int more = 1 + XSUM_MANTISSA_BITS + XSUM_EXP_BIAS - e;

  /* Change 'ivalue' to put in 'more' bits from lower chunks into the bottom.
     Also set 'j' to the index of the lowest chunk from which these bits came,
     and 'lower' to the remaining bits of that chunk not now in 'ivalue'.
     We make sure that 'lower' initially has at least one bit in it, which
     we can later move into 'ivalue' if it turns out that one more bit is
     needed. */

  /* multiply, since << of negative undefined */
  ivalue *= static_cast<xsum_int>(1) << more;
  int j = i - 1;
  /* must exist, since denormalized if i==0 */
  xsum_schunk lower = sacc->chunk[j];
  if (more >= XSUM_LOW_MANTISSA_BITS) {
    more -= XSUM_LOW_MANTISSA_BITS;
    ivalue += lower << more;
    --j;
    lower = j < 0 ? 0 : sacc->chunk[j];
  }

  ivalue += lower >> (XSUM_LOW_MANTISSA_BITS - more);
  lower &= (static_cast<xsum_schunk>(1) << (XSUM_LOW_MANTISSA_BITS - more)) - 1;

  /* Check for a negative 'ivalue' that when negated doesn't contain a full
     mantissa's worth of bits, plus one to help rounding.  If so, move one
     more bit into 'ivalue' from 'lower' (and remove it from 'lower').
     Note that more than one additional bit will not be required because
     xsum_carry_propagate ensures the uppermost non-zero chunk is not -1. */

  if (ivalue < 0 && ((-ivalue) & (static_cast<xsum_int>(1)
                                  << (XSUM_MANTISSA_BITS + 1))) == 0) {
    int const pos = static_cast<xsum_schunk>(1)
                    << (XSUM_LOW_MANTISSA_BITS - 1 - more);
    /* note that left shift undefined if ivalue is negative */
    ivalue *= 2;
    if (lower & pos) {
      ivalue |= 1;
      lower &= ~pos;
    }
    --e;
  }

  /* Set u.intv to have just the correct sign bit (rest zeros), and 'ivalue'
     to now have the absolute value of the mantissa. */

  if (ivalue >= 0) {
    u.intv = 0;
  } else {
    ivalue = -ivalue;
    u.intv = XSUM_SIGN_MASK;
  }

  if (xsum_debug && (ivalue >> (XSUM_MANTISSA_BITS + 1)) != 1) {
    std::abort();
  }

  /* Round to nearest, with ties to even. At this point, 'ivalue' has the
     absolute value of the number to be rounded, including an extra bit at
     the bottom.  Bits below that are in 'lower' and in the chunks
     indexed by 'j' and below.  Note that the bits in 'lower' and the chunks
     below add to the magnitude of the remainder if the number is positive,
     but subtract from this magnitude if the number is negative.

     This code goes to done_rounding if it finds that just discarding lower
     order bits is correct, and to round_away_from_zero if instead the
     magnitude should be increased by one in the lowest bit. */

  /* extra bit is 0 */
  if ((ivalue & 1) == 0) {
    goto done_rounding;
  }

  /* number is positive */
  if (u.intv == 0) {
    /* low bit 1 (odd) */
    if ((ivalue & 2) != 0) {
      goto round_away_from_zero;
    }
    if (lower != 0) {
      goto round_away_from_zero;
    }
  }
  /* number is negative */
  else {
    /* low bit 0 (even) */
    if ((ivalue & 2) == 0) {
      goto done_rounding;
    }
    if (lower != 0) {
      goto done_rounding;
    }
  }

  /* If we get here, 'lower' is zero.  We need to look at chunks lower down
     to see if any are non-zero. */
  while (j > 0) {
    --j;
    if (sacc->chunk[j] != 0) {
      lower = 1;
      break;
    }
  }

  /* number is positive, low bit 0 (even) */
  if (u.intv == 0) {
    if (lower != 0) {
      goto round_away_from_zero;
    } else {
      goto done_rounding;
    }
  }
  /* number is negative, low bit 1 (odd) */
  else {
    if (lower != 0) {
      goto done_rounding;
    } else {
      goto round_away_from_zero;
    }
  }

round_away_from_zero:

  /* Round away from zero, then check for carry having propagated out the
     top, and shift if so. */
  ivalue += 2;
  if (ivalue & (static_cast<xsum_int>(1) << (XSUM_MANTISSA_BITS + 2))) {
    ivalue >>= 1;
    ++e;
  }

done_rounding:;

  /* Get rid of the bottom bit that was used to decide on rounding. */
  ivalue >>= 1;

  /* Adjust to the true exponent, accounting for where this chunk is. */
  e += (i << XSUM_LOW_EXP_BITS) - XSUM_EXP_BIAS - XSUM_MANTISSA_BITS;

  /* If exponent has overflowed, change to plus or minus Inf and return. */
  if (e >= XSUM_EXP_MASK) {
    u.intv |= static_cast<xsum_int>(XSUM_EXP_MASK) << XSUM_MANTISSA_BITS;
    return u.fltv;
  }

  /* Put exponent and mantissa into u.intv, which already has the sign,
     then return u.fltv. */
  u.intv += static_cast<xsum_int>(e) << XSUM_MANTISSA_BITS;

  /* mask out the implicit 1 bit */
  u.intv += ivalue & XSUM_MANTISSA_MASK;

  if (xsum_debug && (ivalue >> XSUM_MANTISSA_BITS) != 1) {
    std::abort();
  }

  return u.fltv;
}

xsum_small_accumulator *xsum_round_to_small(
    xsum_large_accumulator *const lacc) {
  xsum_used *p = lacc->chunks_used;
  xsum_used *e = p + XSUM_LCHUNKS / 64;

  /* Very quickly skip some unused low-order blocks of chunks by looking
     at the used_used flags. */

  xsum_used uu = lacc->used_used;

  if ((uu & 0xffffffff) == 0) {
    uu >>= 32;
    p += 32;
  }
  if ((uu & 0xffff) == 0) {
    uu >>= 16;
    p += 16;
  }
  if ((uu & 0xff) == 0) {
    p += 8;
  }

  xsum_used u;

  /* Loop over remaining blocks of chunks. */
  do {
    /* Loop to quickly find the next non-zero block of used flags, or finish
       up if we've added all the used blocks to the small accumulator. */

    for (;;) {
      u = *p;
      if (u != 0) {
        break;
      }

      ++p;
      if (p == e) {
        return &lacc->sacc;
      }

      u = *p;
      if (u != 0) {
        break;
      }

      ++p;
      if (p == e) {
        return &lacc->sacc;
      }

      u = *p;
      if (u != 0) {
        break;
      }

      ++p;
      if (p == e) {
        return &lacc->sacc;
      }

      u = *p;
      if (u != 0) {
        break;
      }

      ++p;
      if (p == e) {
        return &lacc->sacc;
      }
    }

    /* Find and process the chunks in this block that are used.  We skip
       forward based on the chunks_used flags until we're within eight
        bits of a chunk that is in use. */

    int ix = (p - lacc->chunks_used) << 6;

    if ((u & 0xffffffff) == 0) {
      u >>= 32;
      ix += 32;
    }
    if ((u & 0xffff) == 0) {
      u >>= 16;
      ix += 16;
    }
    if ((u & 0xff) == 0) {
      u >>= 8;
      ix += 8;
    }

    do {
      if (lacc->count[ix] >= 0) {
        xsum_add_lchunk_to_small(lacc, ix);
      }
      ++ix;
      u >>= 1;
    } while (u != 0);

    ++p;

  } while (p != e);

  /* Finish now that all blocks have been added to the small accumulator
     by calling the small accumulator rounding function. */

  return &lacc->sacc;
}

template <>
xsum_flt xsum_round<xsum_large_accumulator>(
    xsum_large_accumulator *const lacc) {
  return xsum_round<xsum_small_accumulator>(xsum_round_to_small(lacc));
}

inline void xsum_add_lchunk_to_small(xsum_large_accumulator *const lacc,
                                     xsum_expint const ix) {
  // xsum_expint exp, low_exp, high_exp;
  // xsum_uint low_chunk, mid_chunk, high_chunk;
  // xsum_lchunk chunk;

  xsum_expint const count = lacc->count[ix];

  /* Add to the small accumulator only if the count is not -1, which
     indicates a chunk that contains nothing yet. */
  if (count >= 0) {
    /* Propagate carries in the small accumulator if necessary. */
    if (lacc->sacc.adds_until_propagate == 0) {
      xsum_carry_propagate(&lacc->sacc);
    }

    /* Get the chunk we will add.  Note that this chunk is the integer sum
       of entire 64-bit floating-point representations, with sign, exponent,
       and mantissa, but we want only the sum of the mantissas. */
    xsum_lchunk chunk = lacc->chunk[ix];

    /* If we added the maximum number of values to 'chunk', the sum of
       the sign and exponent parts (all the same, equal to the index) will
       have overflowed out the top, leaving only the sum of the mantissas.
       If the count of how many more terms we could have summed is greater
       than zero, we therefore add this count times the index (shifted to
       the position of the sign and exponent) to get the unwanted bits to
       overflow out the top. */
    if (count > 0) {
      chunk += static_cast<xsum_lchunk>(count * ix) << XSUM_MANTISSA_BITS;
    }

    /* Find the exponent for this chunk from the low bits of the index,
       and split it into low and high parts, for accessing the small
       accumulator.  Noting that for denormalized numbers where the
       exponent part is zero, the actual exponent is 1 (before subtracting
       the bias), not zero. */

    xsum_expint low_exp;
    xsum_expint high_exp;

    xsum_expint const exp = ix & XSUM_EXP_MASK;
    if (exp == 0) {
      low_exp = 1;
      high_exp = 0;
    } else {
      low_exp = exp & XSUM_LOW_EXP_MASK;
      high_exp = exp >> XSUM_LOW_EXP_BITS;
    }

    /* Split the mantissa into three parts, for three consecutive chunks in
       the small accumulator.  Except for denormalized numbers, add in the sum
       of all the implicit 1 bits that are above the actual mantissa bits. */
    xsum_uint const low_chunk = (chunk << low_exp) & XSUM_LOW_MANTISSA_MASK;
    xsum_uint mid_chunk = chunk >> (XSUM_LOW_MANTISSA_BITS - low_exp);

    /* normalized */
    if (exp != 0) {
      mid_chunk += static_cast<xsum_lchunk>((1 << XSUM_LCOUNT_BITS) - count)
                   << (XSUM_MANTISSA_BITS - XSUM_LOW_MANTISSA_BITS + low_exp);
    }

    xsum_uint const high_chunk = mid_chunk >> XSUM_LOW_MANTISSA_BITS;
    mid_chunk &= XSUM_LOW_MANTISSA_MASK;

    /* Add or subtract the three parts of the mantissa from three small
       accumulator chunks, according to the sign that is part of the index. */

    if (ix & (1 << XSUM_EXP_BITS)) {
      lacc->sacc.chunk[high_exp] -= low_chunk;
      lacc->sacc.chunk[high_exp + 1] -= mid_chunk;
      lacc->sacc.chunk[high_exp + 2] -= high_chunk;
    } else {
      lacc->sacc.chunk[high_exp] += low_chunk;
      lacc->sacc.chunk[high_exp + 1] += mid_chunk;
      lacc->sacc.chunk[high_exp + 2] += high_chunk;
    }

    /* The above additions/subtractions reduce by one the number we can
       do before we need to do carry propagation again. */
    --lacc->sacc.adds_until_propagate;
  }

  /* We now clear the chunk to zero, and set the count to the number
     of adds we can do before the mantissa would overflow.  We also
     set the bit in chunks_used to indicate that this chunk is in use
     (if that is enabled). */

  lacc->chunk[ix] = 0;
  lacc->count[ix] = 1 << XSUM_LCOUNT_BITS;
  lacc->chunks_used[ix >> 6] |= static_cast<xsum_used>(1) << (ix & 0x3f);
  lacc->used_used |= static_cast<xsum_used>(1) << (ix >> 6);
}

inline void xsum_large_add_value_inf_nan(xsum_large_accumulator *const lacc,
                                         xsum_expint const ix,
                                         xsum_lchunk const uintv) {
  if ((ix & XSUM_EXP_MASK) == XSUM_EXP_MASK) {
    xsum_small_add_inf_nan(&lacc->sacc, uintv);
  } else {
    xsum_add_lchunk_to_small(lacc, ix);
    --lacc->count[ix];
    lacc->chunk[ix] += uintv;
  }
}

/* PRINT DOUBLE-PRECISION FLOATING POINT VALUE IN BINARY. */
void pbinary_double(double const d) {
  union {
    double f;
    std::int64_t i;
  } u;

  u.f = d;

  std::int64_t const exp = (u.i >> 52) & 0x7ff;

  std::cout << (u.i < 0 ? "- " : "+ ") << std::bitset<11>(exp);
  if (exp == 0) {
    std::cout << " (denorm) ";
  } else if (exp == 0x7ff) {
    std::cout << " (InfNaN) ";
  } else {
    std::cout << " (+" << std::setfill('0') << std::setw(6)
              << static_cast<int>(exp - 1023) << ") ";
  }
  std::cout << std::bitset<52>(u.i & 0xfffffffffffffL);
}

#endif  // XSUM_HPP
