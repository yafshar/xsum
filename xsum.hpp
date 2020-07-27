#ifndef EXACTSUM_HPP
#define EXACTSUM_HPP

#include <cstdlib>
#include <cstdint>

#include <memory>

/* CONSTANTS DEFINING THE FLOATING POINT FORMAT. */

/* C floating point type sums are done for */
using xsum_flt = double;
/* Signed integer type for a fp value */
using xsum_int = std::int64_t;
/* Unsigned integer type for a fp value */
using xsum_uint = std::uint64_t;
/* Integer type for holding an exponent */
using xsum_expint = std::int_fast16_t;
/* TYPE FOR LENGTHS OF ARRAYS.  Must be a signed integer type. */
using xsum_length = int;
/* Integer type of small accumulator chunk */
using xsum_schunk = std::int64_t;
/* Integer type of large accumulator chunk, must be EXACTLY 64 bits in size */
using xsum_lchunk = std::uint64_t;
/* Signed int type of counts for large acc.*/
using xsum_lcount = std::int_least16_t;
/* Unsigned type for holding used flags */
using xsum_used = std::uint_fast64_t;

/* Bits in fp mantissa, excludes implict 1 */
constexpr int XSUM_MANTISSA_BITS = 52;
/* Bits in fp exponent */
constexpr int XSUM_EXP_BITS = 11;
/* Mask for mantissa bits */
constexpr xsum_int XSUM_MANTISSA_MASK = ((static_cast<xsum_int>(1) << XSUM_MANTISSA_BITS) - static_cast<xsum_int>(1));
/* Mask for exponent */
constexpr int XSUM_EXP_MASK = ((1 << XSUM_EXP_BITS) - 1);
/* Bias added to signed exponent */
constexpr int XSUM_EXP_BIAS = ((1 << (XSUM_EXP_BITS - 1)) - 1);
/* Position of sign bit */
constexpr int XSUM_SIGN_BIT = (XSUM_MANTISSA_BITS + XSUM_EXP_BITS);
/* Mask for sign bit */
constexpr xsum_uint XSUM_SIGN_MASK = (static_cast<xsum_uint>(1) << XSUM_SIGN_BIT);

/* CONSTANTS DEFINING THE SMALL ACCUMULATOR FORMAT. */

/* Bits in chunk of the small accumulator */
constexpr int XSUM_SCHUNK_BITS = 64;
/* # of low bits of exponent, in one chunk */
constexpr int XSUM_LOW_EXP_BITS = 5;
/* Mask for low-order exponent bits */
constexpr int XSUM_LOW_EXP_MASK = ((1 << XSUM_LOW_EXP_BITS) - 1);
/* # of high exponent bits for index */
constexpr int XSUM_HIGH_EXP_BITS = (XSUM_EXP_BITS - XSUM_LOW_EXP_BITS);
/* Mask for high-order exponent bits */
constexpr int XSUM_HIGH_EXP_MASK = ((1 << XSUM_HIGH_EXP_BITS) - 1);
/* # of chunks in small accumulator */
constexpr int XSUM_SCHUNKS = ((1 << XSUM_HIGH_EXP_BITS) + 3);
/* Bits in low part of mantissa */
constexpr int XSUM_LOW_MANTISSA_BITS = (1 << XSUM_LOW_EXP_BITS);
/* Bits in high part */
constexpr int XSUM_HIGH_MANTISSA_BITS = (XSUM_MANTISSA_BITS - XSUM_LOW_MANTISSA_BITS);
/* Mask for low bits */
constexpr xsum_int XSUM_LOW_MANTISSA_MASK = ((static_cast<xsum_int>(1) << XSUM_LOW_MANTISSA_BITS) - static_cast<xsum_int>(1));
/* Bits sums can carry into */
constexpr int XSUM_SMALL_CARRY_BITS = ((XSUM_SCHUNK_BITS - 1) - XSUM_MANTISSA_BITS);
/* # terms can add before need prop. */
constexpr int XSUM_SMALL_CARRY_TERMS = ((1 << XSUM_SMALL_CARRY_BITS) - 1);

/* CONSTANTS DEFINING THE LARGE ACCUMULATOR FORMAT. */
/* Bits in chunk of the large accumulator */
constexpr int XSUM_LCHUNK_BITS = 64;
/* # of bits in count */
constexpr int XSUM_LCOUNT_BITS = (64 - XSUM_MANTISSA_BITS);
/* # of chunks in large accumulator */
constexpr int XSUM_LCHUNKS = (1 << (XSUM_EXP_BITS + 1));

/* CLASSES FOR EXACT SUMMATION. */

struct xsum_small_accumulator
{
    /* Chunks making up small accumulator */
    xsum_schunk chunk[XSUM_SCHUNKS] = {};
    /* If non-zero, +Inf, -Inf, or NaN */
    xsum_int Inf = 0;
    /* If non-zero, a NaN value with payload */
    xsum_int NaN = 0;
    /* Number of remaining adds before carry */
    int adds_until_propagate = XSUM_SMALL_CARRY_TERMS;
};

class xsum_small
{
public:
    xsum_small();
    ~xsum_small();
    explicit xsum_small(xsum_small &&other);
    xsum_small &operator=(xsum_small &&other);
    void swap(xsum_small &other);
    void reset();

    /*
     * ADD ONE DOUBLE TO A SMALL ACCUMULATOR.  This is equivalent to, but
     * somewhat faster than, calling xsum_small_addv with a vector of one
     * value (which in fact will call this function).
     */
    void add(xsum_flt const value);

    /*
     * ADD A VECTOR OF FLOATING-POINT NUMBERS TO A SMALL ACCUMULATOR.  Mixes
     * calls of xsum_carry_propagate with calls of xsum_addv_no_carry to add
     * parts that are small enough that no carry will result.  Note that
     * xsum_addv_no_carry may pre-fetch one beyond the last value it sums,
     * so to be safe, adding the last value has to be done separately at
     * the end.
     */
    void add(xsum_flt const *__restrict__ vec, xsum_length const n);

    /*
     * ADD SQUARED NORM OF VECTOR OF FLOATING-POINT NUMBERS TO SMALL ACCUMULATOR.
     * Mixes calls of xsum_carry_propagate with calls of xsum_add_sqnorm_no_carry
     * to add parts that are small enough that no carry will result.  Note that
     * xsum_add_sqnorm_no_carry may pre-fetch one beyond the last value it sums,
     * so to be safe, adding the last value has to be done separately at
     * the end.
     */
    void add_sqnorm(xsum_flt const *__restrict__ vec, xsum_length const n);

    /*
     * ADD DOT PRODUCT OF VECTORS FLOATING-POINT NUMBERS TO SMALL ACCUMULATOR.
     * Mixes calls of xsum_carry_propagate with calls of xsum_add_dot_no_carry
     * to add parts that are small enough that no carry will result.  Note that
     * xsum_add_dot_no_carry may pre-fetch one beyond the last value it sums,
     * so to be safe, adding the last value has to be done separately at
     * the end.
     */
    void add_dot(xsum_flt const *__restrict__ vec1, xsum_flt const *__restrict__ vec2, xsum_length const n);

    /*
     * RETURN THE RESULT OF ROUNDING A SMALL ACCUMULATOR.  The rounding mode
     * is to nearest, with ties to even.  The small accumulator may be modified
     * by this operation (by carry propagation being done), but the value it
     * represents should not change.
     */
    xsum_flt round();

    /* Display a small accumulator. */
    void display();

    /* Return number of chunks in use in small accumulator. */
    int chunks_used();

    inline int n_remaining_adds();

    inline xsum_small_accumulator *get();

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

    /*
     * ADD ONE NUMBER TO A SMALL ACCUMULATOR ASSUMING NO CARRY PROPAGATION REQ'D.
     * This function is declared "inline" for good performance it must be inlined
     * by the compiler (otherwise the procedure call overhead will result in
     * substantial inefficiency).
     */
    inline void add_no_carry(xsum_flt const value);

    /*
     * ADD A VECTOR TO A SMALL ACCUMULATOR, ASSUMING NO CARRY PROPAGATION NEEDED.
     * Adds n-1 numbers from vec, which must have at least n elements; n must
     * be at least 1.  This odd specificiation is designed so that in the OPT
     * version we can pre-fetch the next value to allow some time for memory
     * response before the value is used.
     */
    inline void add_no_carry(xsum_flt const *__restrict__ vec, xsum_length const n);

    /*
     * ADD SQUARED NORM OF VECTOR TO SMALL ACCUMULATOR, ASSUME NO CARRY NEEDED.
     * Adds n-1 squares of numbers from vec, which must have at least n elements;
     * n must be at least 1.  This odd specificiation is designed so that in the
     * OPT version we can pre-fetch the next value to allow some time for memory
     * response before the value is used.
     */
    inline void add_sqnorm_no_carry(xsum_flt const *__restrict__ vec, xsum_length const n);

    /*
     * ADD DOT PRODUCT OF VECTORS TO SMALL ACCUMULATOR, ASSUME NO CARRY NEEDED.
     * Adds n-1 products of numbers from vec1 and vec2, which must have at least
     * n elements; n must be at least 1.  This odd specificiation is designed so
     * that in the OPT version we can pre-fetch the next values to allow some time
     * for memory response before the value is used.
     */
    inline void add_dot_no_carry(xsum_flt const *__restrict__ vec1, xsum_flt const *__restrict__ vec2,
                                 xsum_length const n);

private:
    xsum_small(xsum_small const &) = delete;
    xsum_small &operator=(xsum_small const &) = delete;

private:
    std::unique_ptr<xsum_small_accumulator> _sacc;
};

struct xsum_large_accumulator
{
    xsum_large_accumulator();

    /* Chunks making up large accumulator */
    xsum_lchunk chunk[XSUM_LCHUNKS] = {};
    /* Counts of # adds remaining for chunks, or -1 if not used yet or special. */
    xsum_lcount count[XSUM_LCHUNKS];
    /* Bits indicate chunks in use */
    xsum_used chunks_used[XSUM_LCHUNKS / 64] = {};
    /* Bits indicate chunk_used entries not 0 */
    xsum_used used_used = 0;
    /* The small accumulator to condense into */
    xsum_small sacc;
};

class xsum_large
{
public:
    xsum_large();
    ~xsum_large();
    explicit xsum_large(xsum_large &&other);
    xsum_large &operator=(xsum_large &&other);
    void swap(xsum_large &other);
    void reset();

    /* ADD SINGLE NUMBER TO THE LARGE ACCUMULATOR */
    void add(xsum_flt const value);

    /*
     * ADD A VECTOR OF FLOATING-POINT NUMBERS TO A LARGE ACCUMULATOR.
     */
    void add(xsum_flt const *__restrict__ vec, xsum_length const n);

    /* ADD SQUARED NORM OF VECTOR OF FLOATING-POINT NUMBERS TO LARGE ACCUMULATOR. */
    void add_sqnorm(xsum_flt const *__restrict__ vec, xsum_length const n);

    /* ADD DOT PRODUCT OF VECTORS OF FLOATING-POINT NUMBERS TO LARGE ACCUMULATOR. */
    void add_dot(xsum_flt const *__restrict__ vec1, xsum_flt const *__restrict__ vec2, xsum_length const n);

    /*
     * RETURN RESULT OF ROUNDING A LARGE ACCUMULATOR.  Rounding mode is to nearest,
     * with ties to even.
     * This is done by adding all the chunks in the large accumulator to the
     * small accumulator, and then calling its rounding procedure.
     */
    xsum_flt round();

    /* Display a large accumulator. */
    void display();

    /* Return number of chunks in use in large accumulator. */
    int chunks_used();

    /*
     * ADD CHUNK FROM A LARGE ACCUMULATOR TO THE SMALL ACCUMULATOR WITHIN IT.
     * The large accumulator chunk to add is indexed by ix.  This chunk will
     * be cleared to zero and its count reset after it has been added to the
     * small accumulator (except no add is done for a new chunk being initialized).
     * This procedure should not be called for the special chunks correspnding to
     * Inf or NaN, whose counts should always remain at -1.
     */
    void add_lchunk_to_small(xsum_expint const ix);

    /*
     * ADD A CHUNK TO THE LARGE ACCUMULATOR OR PROCESS NAN OR INF.  This routine
     * is called when the count for a chunk is negative after decrementing, which
     * indicates either inf/nan, or that the chunk has not been initialized, or
     * that the chunk needs to be transferred to the small accumulator.
     */
    inline void add_value_inf_nan(xsum_expint const ix, xsum_lchunk const uintv);

    inline xsum_large_accumulator *get();

private:
    xsum_large(xsum_large const &) = delete;
    xsum_large &operator=(xsum_large const &) = delete;

private:
    std::unique_ptr<xsum_large_accumulator> _lacc;
};


/* DEBUG FLAG.  Set to non-zero for debug ouptut.  Ignored unless xsum.c is compiled with -DDEBUG. */

constexpr int xsum_debug = 0;

/* IMPLEMENTATION OPTIONS.  Can be set to either 0 or 1, whichever seems to be fastest. */

/*   operations done with simple FP arithmetic?   */
constexpr int OPT_SIMPLE_SQNORM = 1;
constexpr int OPT_SIMPLE_DOT = 1;
constexpr int OPT_KAHAN_SUM = 0;

/* UNION OF FLOATING AND INTEGER TYPES. */
union fpunion
{
    xsum_flt fltv;
    xsum_int intv;
    xsum_uint uintv;
};

static void pbinary_int64(std::int64_t const v, int const n);
static void pbinary_double(double const d);

/* FUNCTIONS FOR DOUBLE AND OTHER INEXACT SUMMATION. */

xsum_flt xsum_sum_double(xsum_flt const *__restrict__ vec, xsum_length const n);
xsum_flt xsum_sum_double_not_ordered(xsum_flt const *__restrict__ vec, xsum_length const n);
xsum_flt xsum_sum_float128(xsum_flt const *__restrict__ vec, xsum_length const n);
xsum_flt xsum_sum_kahan(xsum_flt const *__restrict__ vec, xsum_length const n);
xsum_flt xsum_sqnorm_double(xsum_flt const *__restrict__ vec, xsum_length const n);
xsum_flt xsum_sqnorm_double_not_ordered(xsum_flt const *__restrict__ vec, xsum_length const n);
xsum_flt xsum_dot_double(xsum_flt const *vec1, xsum_flt const *vec2, xsum_length const n);
xsum_flt xsum_dot_double_not_ordered(xsum_flt const *vec1, xsum_flt const *vec2, xsum_length const n);

#endif // EXACTSUM_HPP
