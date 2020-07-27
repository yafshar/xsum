#include "xsum.hpp"

#include <cstdio>
#include <cstring>

#include <algorithm>
#include <iostream>
#include <iomanip>

/* INITIALIZE A SMALL ACCUMULATOR TO ZERO. */

xsum_small::xsum_small() : _sacc(new xsum_small_accumulator) {}

xsum_small::~xsum_small() {}

xsum_small::xsum_small(xsum_small &&other)
{
    _sacc = std::move(other._sacc);
}

xsum_small &xsum_small::operator=(xsum_small &&other)
{
    _sacc = std::move(other._sacc);
    return *this;
}

void xsum_small::swap(xsum_small &other)
{
    _sacc.swap(other._sacc);
}

void xsum_small::reset()
{
    _sacc.reset(new xsum_small_accumulator);
}

void xsum_small::add(xsum_flt const value)
{
    xsum_small_accumulator *sacc = _sacc.get();

    if (sacc->adds_until_propagate == 0)
    {
        carry_propagate();
    }

    add_no_carry(value);

    sacc->adds_until_propagate -= 1;
}

void xsum_small::add(xsum_flt const *__restrict__ vec, xsum_length const n)
{
    if (n == 0)
    {
        return;
    }

    xsum_small_accumulator *sacc = _sacc.get();

    xsum_length c = n;
    while (c > 1)
    {
        if (sacc->adds_until_propagate == 0)
        {
            carry_propagate();
        }

        xsum_length const m = c - ((1 <= sacc->adds_until_propagate) ? c - 1 : sacc->adds_until_propagate);

        add_no_carry(vec, m + 1);

        sacc->adds_until_propagate -= m;

        vec += m;
        c -= m;
    }

    add(*vec);
}

void xsum_small::add_sqnorm(xsum_flt const *__restrict__ vec, xsum_length const n)
{
    if (n == 0)
    {
        return;
    }

    xsum_small_accumulator *sacc = _sacc.get();

    xsum_length c = n;
    while (c > 1)
    {
        if (sacc->adds_until_propagate == 0)
        {
            carry_propagate();
        }

        xsum_length const m = c - ((1 <= sacc->adds_until_propagate) ? c - 1 : sacc->adds_until_propagate);

        add_sqnorm_no_carry(vec, m + 1);

        sacc->adds_until_propagate -= m;

        vec += m;
        c -= m;
    }

    add(*vec * *vec);
}

void xsum_small::add_dot(xsum_flt const *vec1, xsum_flt const *vec2, xsum_length const n)
{
    if (n == 0)
    {
        return;
    }

    xsum_small_accumulator *sacc = _sacc.get();

    xsum_length c = n;
    while (c > 1)
    {
        if (sacc->adds_until_propagate == 0)
        {
            carry_propagate();
        }

        xsum_length const m = c - ((1 <= sacc->adds_until_propagate) ? c - 1 : sacc->adds_until_propagate);

        add_dot_no_carry(vec1, vec2, m + 1);

        vec1 += m;
        vec2 += m;

        sacc->adds_until_propagate -= m;

        c -= m;
    }

    add(*vec1 * *vec2);
}

xsum_flt xsum_small::round()
{
    if (xsum_debug)
    {
        std::cout << "Rounding small accumulator\n";
    }

    xsum_small_accumulator *sacc = _sacc.get();

    /* See if we have a NaN from one of the numbers being a NaN, in which
       case we return the NaN with largest payload. */

    union fpunion u;

    if (sacc->NaN != 0)
    {
        u.intv = sacc->NaN;
        return u.fltv;
    }

    /* Otherwise, if any number was infinite, we return +Inf, -Inf, or a Nan
       (if both +Inf and -Inf occurred).  Note that we do NOT return NaN if
       we have both an infinite number and a sum of other numbers that
       overflows with opposite sign, since there is no real ambiguity in
       such a case. */

    if (sacc->Inf != 0)
    {
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

    int i = carry_propagate();

    if (xsum_debug)
    {
        display();
    }

    xsum_int ivalue = sacc->chunk[i];

    /* Handle a possible denormalized number, including zero. */

    if (i <= 1)
    {
        /* Check for zero value, in which case we can return immediately. */

        if (ivalue == 0)
        {
            return 0.0;
        }

        /* Check if it is actually a denormalized number.  It always is if only
           the lowest chunk is non-zero.  If the lowest non-zero low chunk is
           the next-to-lowest, we check the magnitude of the absolute value.
           Note that the real exponent is 1 (not 0), so we need to shift right
           by 1 here, which also means there will be no overflow from the left
           shift below (but must view absolute value as unsigned). */

        if (i == 0)
        {
            u.intv = ((ivalue > 0) ? ivalue : -ivalue);
            u.intv >>= 1;
            if (ivalue < 0)
            {
                u.intv |= XSUM_SIGN_MASK;
            }
            return u.fltv;
        }
        else
        {
            u.intv = (ivalue << (XSUM_LOW_MANTISSA_BITS - 1)) + (sacc->chunk[0] >> 1);
            if (u.intv < 0)
            {
                u.intv = -u.intv;
            }

            if (u.uintv < static_cast<xsum_uint>(1) << XSUM_MANTISSA_BITS)
            {
                if (ivalue < 0)
                {
                    u.intv |= XSUM_SIGN_MASK;
                }
                return u.fltv;
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

    int e = (u.intv >> XSUM_MANTISSA_BITS) & XSUM_EXP_MASK;
    int more = 1 + XSUM_MANTISSA_BITS + XSUM_EXP_BIAS - e;

    if (xsum_debug)
    {
        std::cout << "e: " << e << ", more: " << more << ", ivalue: "
                  << std::hex << std::setfill('0') << std::setw(16)
                  << static_cast<long long>(ivalue)
                  << "\n";
    }

    /* Change 'ivalue' to put in 'more' bits from lower chunks into the bottom.
       Also set 'j' to the index of the lowest chunk from which these bits came,
       and 'lower' to the remaining bits of that chunk not now in 'ivalue'.
       We make sure that 'lower' initially has at least one bit in it, which
       we can later move into 'ivalue' if it turns out that one more bit is
       needed. */

    ivalue <<= more;
    if (xsum_debug)
    {
        std::cout << "after ivalue <<= more, ivalue: "
                  << std::hex << std::setfill('0') << std::setw(16)
                  << static_cast<long long>(ivalue)
                  << "\n";
    }

    int j = i - 1;

    /* must exist, since denormalized if i==0 */
    xsum_schunk lower = sacc->chunk[j];

    if (more >= XSUM_LOW_MANTISSA_BITS)
    {
        more -= XSUM_LOW_MANTISSA_BITS;
        ivalue += lower << more;
        if (xsum_debug)
        {
            std::cout << "after ivalue += lower << more, ivalue: "
                      << std::hex << std::setfill('0') << std::setw(16)
                      << static_cast<long long>(ivalue)
                      << "\n";
        }

        --j;

        lower = ((j < 0) ? 0 : sacc->chunk[j]);
    }

    ivalue += lower >> (XSUM_LOW_MANTISSA_BITS - more);
    lower &= (static_cast<xsum_schunk>(1) << (XSUM_LOW_MANTISSA_BITS - more)) - 1;

    if (xsum_debug)
    {
        std::cout << "j: " << j << ", new e: " << e << ", new |ivalue|: "
                  << std::hex << std::setfill('0') << std::setw(16)
                  << static_cast<long long>(ivalue < 0 ? -ivalue : ivalue)
                  << ", lower: "
                  << std::hex << std::setfill('0') << std::setw(16)
                  << static_cast<long long>(lower)
                  << "\n";
    }

    /* Check for a negative 'ivalue' that when negated doesn't contain a full
       mantissa's worth of bits, plus one to help rounding.  If so, move one
       more bit into 'ivalue' from 'lower' (and remove it from 'lower').
       Note that more than one additional bit will not be required because
       xsum_carry_propagate ensures the uppermost non-zero chunk is not -1. */

    if (ivalue < 0 && ((-ivalue) & (static_cast<xsum_int>(1) << (XSUM_MANTISSA_BITS + 1))) == 0)
    {
        int const pos = static_cast<xsum_schunk>(1) << (XSUM_LOW_MANTISSA_BITS - 1 - more);
        ivalue <<= 1;
        if (lower & pos)
        {
            ivalue |= 1;
            lower &= ~pos;
        }
        --e;
    }

    if (xsum_debug)
    {
        std::cout << "j: " << j << ", new e: " << e << ", new |ivalue|: "
                  << std::hex << std::setfill('0') << std::setw(16)
                  << static_cast<long long>(ivalue < 0 ? -ivalue : ivalue)
                  << ", lower: "
                  << std::hex << std::setfill('0') << std::setw(16)
                  << static_cast<long long>(lower)
                  << "\n";
    }

    /* Set u.intv to have just the correct sign bit (rest zeros), and 'ivalue'
       to now have the absolute value of the mantissa. */

    if (ivalue >= 0)
    {
        u.intv = 0;
    }
    else
    {
        ivalue = -ivalue;
        u.intv = XSUM_SIGN_MASK;
    }

    if (xsum_debug)
    {
        if ((ivalue >> (XSUM_MANTISSA_BITS + 1)) != 1)
        {
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
    if ((ivalue & 1) == 0)
    {
        if (xsum_debug)
        {
            std::cout << "round toward zero, since remainder magnitude is < 1/2\n";
        }

        goto done_rounding;
    }

    /* number is positive */
    if (u.intv == 0)
    {
        /* low bit 1 (odd) */
        if ((ivalue & 2) != 0)
        {
            if (xsum_debug)
            {
                std::cout << "round away from zero, since magnitude >= 1/2, goes to even\n";
            }

            goto round_away_from_zero;
        }

        if (lower != 0)
        {
            if (xsum_debug)
            {
                std::cout << "round away from zero, since magnitude > 1/2 (from 'lower')\n";
            }

            goto round_away_from_zero;
        }
    }
    /* number is negative */
    else
    {
        /* low bit 0 (even) */
        if ((ivalue & 2) == 0)
        {
            if (xsum_debug)
            {
                std::cout << "round toward zero, since magnitude <= 1/2, goes to even\n";
            }

            goto done_rounding;
        }

        if (lower != 0)
        {
            if (xsum_debug)
            {
                std::cout << "round toward zero, since magnitude < 1/2 (from 'lower')\n";
            }

            goto done_rounding;
        }
    }

    /* If we get here, 'lower' is zero.  We need to look at chunks lower down
       to see if any are non-zero. */

    while (j > 0)
    {
        --j;

        if (sacc->chunk[j] != 0)
        {
            lower = 1;
            break;
        }
    }

    /* number is positive, low bit 0 (even) */
    if (u.intv == 0)
    {
        if (lower != 0)
        {
            if (xsum_debug)
            {
                std::cout << "round away from zero, since magnitude > 1/2 (low chunks)\n";
            }

            goto round_away_from_zero;
        }
        else
        {
            if (xsum_debug)
            {
                std::cout << "round toward zero, magnitude == 1/2 (low chunks)\n";
            }

            goto done_rounding;
        }
    }
    else /* number is negative, low bit 1 (odd) */
    {
        if (lower != 0)
        {
            if (xsum_debug)
            {
                std::cout << "round toward zero, since magnitude < 1/2 (low chunks)\n";
            }

            goto done_rounding;
        }
        else
        {
            if (xsum_debug)
            {
                std::cout << "round away from zero, magnitude == 1/2 (low chunks)\n";
            }

            goto round_away_from_zero;
        }
    }

round_away_from_zero:
    /* Round away from zero, then check for carry having propagated out the
       top, and shift if so. */
    ivalue += 2;
    if (ivalue & (static_cast<xsum_int>(1) << (XSUM_MANTISSA_BITS + 2)))
    {
        ivalue >>= 1;
        ++e;
    }

done_rounding:;

    /* Get rid of the bottom bit that was used to decide on rounding. */

    ivalue >>= 1;

    /* Adjust to the true exponent, accounting for where this chunk is. */

    e += (i << XSUM_LOW_EXP_BITS) - XSUM_EXP_BIAS - XSUM_MANTISSA_BITS;

    /* If exponent has overflowed, change to plus or minus Inf and return. */

    if (e >= XSUM_EXP_MASK)
    {
        u.intv |= static_cast<xsum_int>(XSUM_EXP_MASK) << XSUM_MANTISSA_BITS;
        if (xsum_debug)
        {
            std::cout << "Final rounded result: "
                      << std::scientific << std::setprecision(17) << u.fltv
                      << " (overflowed)\n  ";
            pbinary_double(u.fltv);
            std::cout << "\n";
        }
        return u.fltv;
    }

    /* Put exponent and mantissa into u.intv, which already has the sign,
       then return u.fltv. */

    u.intv += static_cast<xsum_int>(e) << XSUM_MANTISSA_BITS;
    u.intv += ivalue & XSUM_MANTISSA_MASK; /* mask out the implicit 1 bit */

    if (xsum_debug)
    {
        if ((ivalue >> XSUM_MANTISSA_BITS) != 1)
        {
            std::abort();
        }
        std::cout << "Final rounded result: "
                  << std::scientific << std::setprecision(17) << u.fltv
                  << "\n  ";
        pbinary_double(u.fltv);
        std::cout << "\n";
    }

    return u.fltv;
}

void xsum_small::display()
{
    xsum_small_accumulator *sacc = _sacc.get();

    std::cout << "Small accumulator:"
              << (sacc->Inf ? "  Inf" : "")
              << (sacc->NaN ? "  NaN" : "")
              << "\n";

    for (int i = XSUM_SCHUNKS - 1, dots = 0; i >= 0; i--)
    {
        if (sacc->chunk[i] == 0)
        {
            if (!dots)
            {
                std::cout << "            ...\n";
            }
            dots = 1;
        }
        else
        {
            std::cout << std::setw(5) << i << " " << std::setw(5)
                      << static_cast<int>((i << XSUM_LOW_EXP_BITS) - XSUM_EXP_BIAS - XSUM_MANTISSA_BITS)
                      << " ";
            pbinary_int64(static_cast<std::int64_t>(sacc->chunk[i] >> 32), XSUM_SCHUNK_BITS - 32);
            std::cout << " ";
            pbinary_int64(static_cast<std::int64_t>(sacc->chunk[i] & 0xffffffff), 32);
            std::cout << "\n";
            dots = 0;
        }
    }
    std::cout << "\n";
}

int xsum_small::chunks_used()
{
    xsum_small_accumulator *sacc = _sacc.get();
    int c = 0;
    for (int i = 0; i < XSUM_SCHUNKS; ++i)
    {
        if (sacc->chunk[i] != 0)
        {
            ++c;
        }
    }
    return c;
}

inline int xsum_small::n_remaining_adds() { return _sacc->adds_until_propagate; }

inline xsum_small_accumulator *xsum_small::get() { return _sacc.get(); }

int xsum_small::carry_propagate()
{
    if (xsum_debug)
    {
        std::cout << "Carry propagating in small accumulator\n";
    }

    /* Set u to the index of the uppermost non-zero (for now) chunk, or
       return with value 0 if there is none. */

    bool found = false;

    int u = XSUM_SCHUNKS - 1;
    switch (XSUM_SCHUNKS & 0x3)
    {
    case 3:
        if (_sacc->chunk[u] != 0)
        {
            found = true;
            break;
        }
        --u;
    case 2:
        if (_sacc->chunk[u] != 0)
        {
            found = true;
            break;
        }
        --u;
    case 1:
        if (_sacc->chunk[u] != 0)
        {
            found = true;
            break;
        }
        --u;
    case 0:;
    }

    if (!found)
    {
        do
        {
            if (_sacc->chunk[u - 3] |
                _sacc->chunk[u - 2] |
                _sacc->chunk[u - 1] |
                _sacc->chunk[u])
            {
                found = true;
                break;
            }
            u -= 4;
        } while (u >= 0);
    }

    if (found)
    {
        while (_sacc->chunk[u] == 0)
        {
            --u;
        }
    }
    else
    {
        if (xsum_debug)
        {
            std::cout << "number is zero (1)\n";
        }

        _sacc->adds_until_propagate = XSUM_SMALL_CARRY_TERMS - 1;

        /* Return index of uppermost non-zero chunk. */
        return 0;
    }

    if (xsum_debug)
    {
        std::cout << "u: \n"
                  << u;
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
    while (i <= e)
    {
        if (_sacc->chunk[i] |
            _sacc->chunk[i + 1] |
            _sacc->chunk[i + 2] |
            _sacc->chunk[i + 3])
        {
            break;
        }
        i += 4;
    }

    xsum_schunk c;

    do
    {
        bool nonzero = false;

        /* Find the next non-zero chunk, or break out of loop if there is none. */
        do
        {
            c = _sacc->chunk[i];

            if (c != 0)
            {
                nonzero = true;
                break;
            }

            ++i;

            if (i > u)
            {
                break;
            }

            c = _sacc->chunk[i];

            if (c != 0)
            {
                nonzero = true;
                break;
            }

            ++i;
        } while (i <= u);

        if (!nonzero)
        {
            break;
        }

        /* Propagate possible carry from this chunk to next chunk up. */
        xsum_schunk const chigh = c >> XSUM_LOW_MANTISSA_BITS;
        if (chigh == 0)
        {
            uix = i;
            ++i;
            /* no need to change this chunk */
            continue;
        }

        if (u == i)
        {
            if (chigh == -1)
            {
                uix = i;
                /* don't propagate -1 into the region of all zeros above */
                break;
            }

            /* we will change chunk[u+1], so we'll need to look at it */
            u = i + 1;
        }

        xsum_schunk const clow = c & XSUM_LOW_MANTISSA_MASK;

        if (clow != 0)
        {
            uix = i;
        }

        /* We now change chunk[i] and add to chunk[i+1]. Note that i+1 should be
           in range (no bigger than XSUM_CHUNKS-1) because the number of chunks
           is big enough to hold any sum, and we do not store redundant chunks
           with values 0 or -1 above previously non-zero chunks. */

        if (xsum_debug)
        {
            if (i + 1 >= XSUM_SCHUNKS)
            {
                std::abort();
            }
        }

        _sacc->chunk[i] = clow;
        _sacc->chunk[i + 1] += chigh;

        ++i;
    } while (i <= u);

    if (xsum_debug)
    {
        std::cout << "  uix: " << uix << "  new u: " << u << "\n";
    }

    /* Check again for the number being zero, since carry propagation might
     have created zero from something that initially looked non-zero. */

    if (uix < 0)
    {
        if (xsum_debug)
        {
            std::cout << "number is zero (2)\n";
        }

        _sacc->adds_until_propagate = XSUM_SMALL_CARRY_TERMS - 1;

        /* Return index of uppermost non-zero chunk. */
        return 0;
    }

    /* While the uppermost chunk is negative, with value -1, combine it with
       the chunk below (if there is one) to produce the same number but with
       one fewer non-zero chunks. */

    while (_sacc->chunk[uix] == -1 && uix > 0)
    {
        _sacc->chunk[uix] = 0;
        --uix;
        _sacc->chunk[uix] += static_cast<xsum_schunk>(-1) << XSUM_LOW_MANTISSA_BITS;
    }

    /* We can now add one less than the total allowed terms before the
       next carry propagate. */

    _sacc->adds_until_propagate = XSUM_SMALL_CARRY_TERMS - 1;

    /* Return index of uppermost non-zero chunk. */
    return uix;
}

void xsum_small::add_inf_nan(xsum_int const ivalue)
{
    xsum_int const mantissa = ivalue & XSUM_MANTISSA_MASK;

    /* Inf */
    if (mantissa == 0)
    {
        if (_sacc->Inf == 0)
        {
            /* no previous Inf */
            _sacc->Inf = ivalue;
        }
        else if (_sacc->Inf != ivalue)
        {
            fpunion u;

            /* previous Inf was opposite sign */
            u.intv = ivalue;

            /* result will be a NaN */
            u.fltv = u.fltv - u.fltv;

            _sacc->Inf = u.intv;
        }
    }
    /* NaN */
    else
    {
        /* Choose the NaN with the bigger payload and clear its sign.  Using <=
           ensures that we will choose the first NaN over the previous zero. */
        if ((_sacc->NaN & XSUM_MANTISSA_MASK) <= mantissa)
        {
            _sacc->NaN = ivalue & ~XSUM_SIGN_MASK;
        }
    }
}

inline void xsum_small::add_no_carry(xsum_flt const value)
{
    if (xsum_debug)
    {
        std::cout << "ADD1 +" << std::setprecision(17)
                  << static_cast<double>(value) << "\n     ";
        pbinary_double((double)value);
        std::cout << "\n";
    }

    union fpunion u;

    /* Extract exponent and mantissa. */
    u.fltv = value;

    xsum_int const ivalue = u.intv;

    xsum_int mantissa = ivalue & XSUM_MANTISSA_MASK;
    xsum_expint exp = (ivalue >> XSUM_MANTISSA_BITS) & XSUM_EXP_MASK;

    /* Categorize number as normal, denormalized, or Inf/NaN according to
       the value of the exponent field. */

    /* normalized OR in implicit 1 bit at top of mantissa */
    if (exp != 0 && exp != XSUM_EXP_MASK)
    {
        mantissa |= static_cast<xsum_int>(1) << XSUM_MANTISSA_BITS;
    }
    /* zero or denormalized */
    else if (exp == 0)
    {
        /* If it's a zero (positive or negative), we do nothing. */
        if (mantissa == 0)
        {
            return;
        }

        /* Denormalized mantissa has no implicit 1, but exponent is 1 not 0. */
        exp = 1;
    }
    /* Inf or NaN */
    else
    {
        /* Just update flags in accumulator structure. */
        add_inf_nan(ivalue);
        return;
    }

    /* Separate high part of exponent, used as index of chunk, and low
       part of exponent, giving position within chunk. */

    xsum_expint const low_exp = exp & XSUM_LOW_EXP_MASK;
    xsum_expint const high_exp = exp >> XSUM_LOW_EXP_BITS;

    if (xsum_debug)
    {
        std::cout << "  high exp: ";
        pbinary_int64(high_exp, XSUM_HIGH_EXP_BITS);
        std::cout << "  low exp: ";
        pbinary_int64(low_exp, XSUM_LOW_EXP_BITS);
        std::cout << "\n";
    }

    xsum_schunk *const chunk_ptr = _sacc.get()->chunk + high_exp;
    xsum_schunk const chunk0 = chunk_ptr[0];
    xsum_schunk const chunk1 = chunk_ptr[1];

    /* Separate mantissa into two parts, after shifting, and add to (or
       subtract from) this chunk and the next higher chunk (which always
       exists since there are three extra ones at the top). */

    xsum_int const low_mantissa = (mantissa << low_exp) & XSUM_LOW_MANTISSA_MASK;
    xsum_int const high_mantissa = mantissa >> (XSUM_LOW_MANTISSA_BITS - low_exp);

    /* Add or subtract to or from the two affected chunks. */

    if (ivalue < 0)
    {
        chunk_ptr[0] = chunk0 - low_mantissa;
        chunk_ptr[1] = chunk1 - high_mantissa;

        if (xsum_debug)
        {
            std::cout << " -high man: ";
            pbinary_int64(-high_mantissa, XSUM_MANTISSA_BITS);
            std::cout << "\n  -low man: ";
            pbinary_int64(-low_mantissa, XSUM_LOW_MANTISSA_BITS);
            std::cout << "\n";
        }
    }
    else
    {
        chunk_ptr[0] = chunk0 + low_mantissa;
        chunk_ptr[1] = chunk1 + high_mantissa;

        if (xsum_debug)
        {
            std::cout << "  high man: ";
            pbinary_int64(high_mantissa, XSUM_MANTISSA_BITS);
            std::cout << "\n   low man: ";
            pbinary_int64(low_mantissa, XSUM_LOW_MANTISSA_BITS);
            std::cout << "\n";
        }
    }
}

inline void xsum_small::add_no_carry(xsum_flt const *__restrict__ vec,
                                     xsum_length const n)
{
    for (xsum_length i = 0; i < n - 1; ++i)
    {
        add_no_carry(vec[i]);
    }
}

inline void xsum_small::add_sqnorm_no_carry(xsum_flt const *__restrict__ vec,
                                            xsum_length const n)
{
    for (xsum_length i = 0; i < n - 1; ++i)
    {
        xsum_flt const f = vec[i];
        xsum_flt const g = f * f;
        add_no_carry(g);
    }
}

inline void xsum_small::add_dot_no_carry(xsum_flt const *__restrict__ vec1,
                                         xsum_flt const *__restrict__ vec2,
                                         xsum_length const n)
{
    for (xsum_length i = 0; i < n - 1; ++i)
    {
        xsum_flt const f = vec1[i];
        xsum_flt const g = vec2[i];
        xsum_flt const h = f * g;
        add_no_carry(h);
    }
}

/* LARGE ACCUMULATOR */

xsum_large_accumulator::xsum_large_accumulator()
{
    /* Since in two's complement representation, -1 consists of all 1 bits,
       we can initialize 16-bit values to -1 by initializing their component
       bytes to 0xff. */
    std::fill(count, count + XSUM_LCHUNKS, -1);
}

xsum_large::xsum_large() : _lacc(new xsum_large_accumulator) {}

xsum_large::~xsum_large() {}

xsum_large::xsum_large(xsum_large &&other)
{
    _lacc = std::move(other._lacc);
}

xsum_large &xsum_large::operator=(xsum_large &&other)
{
    _lacc = std::move(other._lacc);
    return *this;
}

void xsum_large::swap(xsum_large &other)
{
    _lacc.swap(other._lacc);
}

void xsum_large::reset()
{
    _lacc.reset(new xsum_large_accumulator);
}

void xsum_large::add(xsum_flt const value)
{
    if (xsum_debug)
    {
        std::cout << "LARGE ADD SINGLE NUMBER";
    }

    /* Version not manually optimized - maybe the compiler can do better. */

    fpunion u;

    /* Fetch the next number, and convert to integer form in u.uintv. */

    u.fltv = value;

    /* Isolate the upper sign+exponent bits that index the chunk. */

    xsum_expint ix = u.uintv >> XSUM_MANTISSA_BITS;

    /* Find the count for this chunk, and subtract one. */

    xsum_lcount count = _lacc->count[ix] - 1;

    if (count < 0)
    {
        /* If the decremented count is negative, it's either a special
           Inf/NaN chunk (in which case count will stay at -1), or one that
           needs to be transferred to the small accumulator, or one that
           has never been used before and needs to be initialized. */
        add_value_inf_nan(ix, u.uintv);
    }
    else
    {
        /* Store the decremented count of additions allowed before transfer,
           and add this value to the chunk. */
        _lacc->count[ix] = count;
        _lacc->chunk[ix] += u.uintv;
    }
}

void xsum_large::add(xsum_flt const *__restrict__ vec, xsum_length const n)
{
    if (n == 0)
    {
        return;
    }

    if (xsum_debug)
    {
        std::cout << "LARGE ADDV OF " << static_cast<long>(n) << " VALUES\n";
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
    xsum_flt f = *v;

    /* Unrolled loop processing two values each time around.  The loop is
       done as two nested loops, arranged so that the inner one will have
       no branches except for the one looping back.  This is achieved by
       a trick for combining three tests for negativity into one.  The
       last one or two values are not done here, so that the pre-fetching
       will not go past the end of the array (which would probably be OK,
       but is technically not allowed). */

    /* leave out last one or two, terminate when negative, for trick */
    xsum_length m = n - 3;
    while (m >= 0)
    {
        /* Loop processing two values at a time until we're done, or until
           one (or both) of the values result in a chunk needing to be processed.
           Updates are done here for both of these chunks, even though it is not
           yet known whether these updates ought to have been done.  We hope
           this allows for better memory pre-fetch and instruction scheduling. */
        do
        {
            ++v;

            u2.fltv = *v;
            u1.fltv = f;

            ix1 = u1.uintv >> XSUM_MANTISSA_BITS;
            count1 = _lacc->count[ix1] - 1;
            _lacc->count[ix1] = count1;
            _lacc->chunk[ix1] += u1.uintv;

            ++v;
            f = *v;

            ix2 = u2.uintv >> XSUM_MANTISSA_BITS;
            count2 = _lacc->count[ix2] - 1;
            _lacc->count[ix2] = count2;
            _lacc->chunk[ix2] += u2.uintv;

            m -= 2;
        } while ((static_cast<xsum_length>(count1) | static_cast<xsum_length>(count2) | m) >= 0);
        /* ... equivalent to while (count1 >= 0 && count2 >= 0 && m >= 0) */

        /* See if we were actually supposed to update these chunks.  If not,
               back out the changes and then process the chunks as they ought to
               have been processed. */

        if (count1 < 0 || count2 < 0)
        {
            _lacc->count[ix2] = count2 + 1;
            _lacc->chunk[ix2] -= u2.uintv;

            if (count1 < 0)
            {
                _lacc->count[ix1] = count1 + 1;
                _lacc->chunk[ix1] -= u1.uintv;

                add_value_inf_nan(ix1, u1.uintv);

                count2 = _lacc->count[ix2] - 1;
            }

            if (count2 < 0)
            {
                add_value_inf_nan(ix2, u2.uintv);
            }
            else
            {
                _lacc->count[ix2] = count2;
                _lacc->chunk[ix2] += u2.uintv;
            }
        }
    }

    /* Process the last one or two values, without pre-fetching. */

    m += 3;
    for (;;)
    {
        u1.fltv = f;
        ix1 = u1.uintv >> XSUM_MANTISSA_BITS;
        count1 = _lacc->count[ix1] - 1;

        if (count1 < 0)
        {
            add_value_inf_nan(ix1, u1.uintv);
        }
        else
        {
            _lacc->count[ix1] = count1;
            _lacc->chunk[ix1] += u1.uintv;
        }

        --m;
        if (m == 0)
        {
            break;
        }

        ++v;
        f = *v;
    }
}

void xsum_large::add_sqnorm(xsum_flt const *__restrict__ vec, xsum_length const n)
{
    if (n == 0)
    {
        return;
    }

    if (xsum_debug)
    {
        std::cout << "LARGE ADD_SQNORM OF " << static_cast<long>(n) << " VALUES\n";
    }

    fpunion u1;
    fpunion u2;

    int count1;
    int count2;

    xsum_expint ix1;
    xsum_expint ix2;

    xsum_flt const *v = vec;
    xsum_flt f = *v;

    /* Unrolled loop processing two squares each time around.  The loop is
       done as two nested loops, arranged so that the inner one will have
       no branches except for the one looping back.  This is achieved by
       a trick for combining three tests for negativity into one.  The
       last one or two squares are not done here, so that the pre-fetching
       will not go past the end of the array (which would probably be OK,
       but is technically not allowed). */

    /* leave out last one or two, terminate when negative, for trick */
    xsum_length m = n - 3;

    while (m >= 0)
    {
        /* Loop processing two squares at a time until we're done, or until
           one (or both) of them result in a chunk needing to be processed.
           Updates are done here for both of these chunks, even though it is not
           yet known whether these updates ought to have been done.  We hope
           this allows for better memory pre-fetch and instruction scheduling. */
        do
        {
            u1.fltv = f * f;

            ++v;
            f = *v;

            u2.fltv = f * f;

            ix1 = u1.uintv >> XSUM_MANTISSA_BITS;
            count1 = _lacc->count[ix1] - 1;
            _lacc->count[ix1] = count1;
            _lacc->chunk[ix1] += u1.uintv;

            ++v;
            f = *v;

            ix2 = u2.uintv >> XSUM_MANTISSA_BITS;
            count2 = _lacc->count[ix2] - 1;
            _lacc->count[ix2] = count2;
            _lacc->chunk[ix2] += u2.uintv;

            m -= 2;
        } while ((static_cast<xsum_length>(count1) | static_cast<xsum_length>(count2) | m) >= 0);
        /* ... equivalent to while (count1 >= 0 && count2 >= 0 && m >= 0) */

        /* See if we were actually supposed to update these chunks.  If not,
           back out the changes and then process the chunks as they ought to
           have been processed. */

        if (count1 < 0 || count2 < 0)
        {
            _lacc->count[ix2] = count2 + 1;
            _lacc->chunk[ix2] -= u2.uintv;

            if (count1 < 0)
            {
                _lacc->count[ix1] = count1 + 1;
                _lacc->chunk[ix1] -= u1.uintv;

                add_value_inf_nan(ix1, u1.uintv);

                count2 = _lacc->count[ix2] - 1;
            }

            if (count2 < 0)
            {
                add_value_inf_nan(ix2, u2.uintv);
            }
            else
            {
                _lacc->count[ix2] = count2;
                _lacc->chunk[ix2] += u2.uintv;
            }
        }
    }

    /* Process the last one or two squares, without pre-fetching. */

    m += 3;
    for (;;)
    {
        u1.fltv = f * f;
        ix1 = u1.uintv >> XSUM_MANTISSA_BITS;

        count1 = _lacc->count[ix1] - 1;
        if (count1 < 0)
        {
            add_value_inf_nan(ix1, u1.uintv);
        }
        else
        {
            _lacc->count[ix1] = count1;
            _lacc->chunk[ix1] += u1.uintv;
        }

        --m;
        if (m == 0)
        {
            break;
        }

        ++v;
        f = *v;
    }
}

void xsum_large::add_dot(xsum_flt const *__restrict__ vec1, xsum_flt const *__restrict__ vec2, xsum_length const n)
{
    if (n == 0)
    {
        return;
    }

    if (xsum_debug)
    {
        std::cout << "LARGE ADD_DOT OF " << static_cast<long>(n) << " VALUES\n";
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

    xsum_flt f1 = *v1;
    xsum_flt f2 = *v2;

    /* Unrolled loop processing two products each time around.  The loop is
       done as two nested loops, arranged so that the inner one will have
       no branches except for the one looping back.  This is achieved by
       a trick for combining three tests for negativity into one.  The
       last one or two products are not done here, so that the pre-fetching
       will not go past the end of the array (which would probably be OK,
       but is technically not allowed). */

    /* leave out last one or two, terminate when negative, for trick */
    xsum_length m = n - 3;

    while (m >= 0)
    {
        /* Loop processing two products at a time until we're done, or until
           one (or both) of them result in a chunk needing to be processed.
           Updates are done here for both of these chunks, even though it is not
           yet known whether these updates ought to have been done.  We hope
           this allows for better memory pre-fetch and instruction scheduling. */
        do
        {
            u1.fltv = f1 * f2;

            ++v1;
            f1 = *v1;

            ++v2;
            f2 = *v2;

            u2.fltv = f1 * f2;

            ix1 = u1.uintv >> XSUM_MANTISSA_BITS;
            count1 = _lacc->count[ix1] - 1;
            _lacc->count[ix1] = count1;
            _lacc->chunk[ix1] += u1.uintv;

            ++v1;
            f1 = *v1;

            ++v2;
            f2 = *v2;

            ix2 = u2.uintv >> XSUM_MANTISSA_BITS;
            count2 = _lacc->count[ix2] - 1;
            _lacc->count[ix2] = count2;
            _lacc->chunk[ix2] += u2.uintv;

            m -= 2;
        } while ((static_cast<xsum_length>(count1) | static_cast<xsum_length>(count2) | m) >= 0);
        /* ... equivalent to while (count1 >= 0 && count2 >= 0 && m >= 0) */

        /* See if we were actually supposed to update these chunks.  If not,
           back out the changes and then process the chunks as they ought to
           have been processed. */

        if (count1 < 0 || count2 < 0)
        {
            _lacc->count[ix2] = count2 + 1;
            _lacc->chunk[ix2] -= u2.uintv;

            if (count1 < 0)
            {
                _lacc->count[ix1] = count1 + 1;
                _lacc->chunk[ix1] -= u1.uintv;

                add_value_inf_nan(ix1, u1.uintv);

                count2 = _lacc->count[ix2] - 1;
            }

            if (count2 < 0)
            {
                add_value_inf_nan(ix2, u2.uintv);
            }
            else
            {
                _lacc->count[ix2] = count2;
                _lacc->chunk[ix2] += u2.uintv;
            }
        }
    }

    /* Process the last one or two products, without pre-fetching. */

    m += 3;
    for (;;)
    {
        u1.fltv = f1 * f2;

        ix1 = u1.uintv >> XSUM_MANTISSA_BITS;

        count1 = _lacc->count[ix1] - 1;
        if (count1 < 0)
        {
            add_value_inf_nan(ix1, u1.uintv);
        }
        else
        {
            _lacc->count[ix1] = count1;
            _lacc->chunk[ix1] += u1.uintv;
        }

        --m;
        if (m == 0)
        {
            break;
        }

        ++v1;
        f1 = *v1;

        ++v2;
        f2 = *v2;
    }
}

xsum_flt xsum_large::round()
{
    if (xsum_debug)
    {
        std::cout << "Rounding large accumulator\n";
    }

    xsum_used *p = _lacc->chunks_used;
    xsum_used *e = p + XSUM_LCHUNKS / 64;

    /* Very quickly skip some unused low-order blocks of chunks by looking
           at the used_used flags. */

    xsum_used uu = _lacc->used_used;
    if ((uu & 0xffffffff) == 0)
    {
        uu >>= 32;
        p += 32;
    }

    if ((uu & 0xffff) == 0)
    {
        uu >>= 16;
        p += 16;
    }

    if ((uu & 0xff) == 0)
    {
        p += 8;
    }

    /* Loop over remaining blocks of chunks. */
    xsum_used u;
    int ix;
    do
    {
        /* Loop to quickly find the next non-zero block of used flags, or finish
               up if we've added all the used blocks to the small accumulator. */

        for (;;)
        {
            u = *p;
            if (u != 0)
            {
                break;
            }

            ++p;
            if (p == e)
            {
                return _lacc->sacc.round();
            }

            u = *p;
            if (u != 0)
            {
                break;
            }

            ++p;
            if (p == e)
            {
                return _lacc->sacc.round();
            }

            u = *p;
            if (u != 0)
            {
                break;
            }

            ++p;
            if (p == e)
            {
                return _lacc->sacc.round();
            }

            u = *p;
            if (u != 0)
            {
                break;
            }

            ++p;
            if (p == e)
            {
                return _lacc->sacc.round();
            }
        }

        /* Find and process the chunks in this block that are used.  We skip
               forward based on the chunks_used flags until we're within eight
               bits of a chunk that is in use. */

        ix = (p - _lacc->chunks_used) << 6;
        if ((u & 0xffffffff) == 0)
        {
            u >>= 32;
            ix += 32;
        }

        if ((u & 0xffff) == 0)
        {
            u >>= 16;
            ix += 16;
        }

        if ((u & 0xff) == 0)
        {
            u >>= 8;
            ix += 8;
        }

        do
        {
            if (_lacc->count[ix] >= 0)
            {
                add_lchunk_to_small(ix);
            }

            ++ix;
            u >>= 1;
        } while (u != 0);

        ++p;
    } while (p != e);

    /* Finish now that all blocks have been added to the small accumulator
       by calling the small accumulator rounding function. */
    return _lacc->sacc.round();
}

void xsum_large::display()
{
    std::cout << "Large accumulator:\n";

    int dots = 0;
    for (int i = XSUM_LCHUNKS - 1; i >= 0; --i)
    {
        if (_lacc->count[i] < 0)
        {
            if (!dots)
            {
                std::cout << "            ...\n";
            }
            dots = 1;
        }
        else
        {
            std::cout << (i & 0x800 ? '-' : '+') << std::setw(4)
                      << (i & 0x7ff) << " "
                      << std::setw(5) << _lacc->count[i] << " ";
            pbinary_int64(static_cast<std::int64_t>(_lacc->chunk[i]) >> 32, XSUM_LCHUNK_BITS - 32);
            std::cout << " ";
            pbinary_int64(static_cast<std::int64_t>(_lacc->chunk[i]) & 0xffffffff, 32);
            std::cout << "\n";

            dots = 0;
        }
    }

    std::cout << "\nWithin large accumulator:  ";

    _lacc->sacc.display();
}

int xsum_large::chunks_used()
{
    int c = 0;
    for (int i = 0; i < XSUM_LCHUNKS; ++i)
    {
        if (_lacc->count[i] >= 0)
        {
            ++c;
        }
    }
    return c;
}

void xsum_large::add_lchunk_to_small(xsum_expint const ix)
{
    xsum_expint const count = _lacc->count[ix];

    /* Add to the small accumulator only if the count is not -1, which
       indicates a chunk that contains nothing yet. */
    if (count >= 0)
    {
        /* Propagate carries in the small accumulator if necessary. */

        if (_lacc->sacc.n_remaining_adds() == 0)
        {
            _lacc->sacc.carry_propagate();
        }

        /* Get the chunk we will add.  Note that this chunk is the integer sum
           of entire 64-bit floating-point representations, with sign, exponent,
           and mantissa, but we want only the sum of the mantissas. */

        xsum_lchunk chunk = _lacc->chunk[ix];

        if (xsum_debug)
        {
            std::cout << "Adding chunk " << static_cast<int>(ix)
                      << " to small accumulator (count "
                      << static_cast<int>(count) << ", chunk "
                      << std::hex << std::setfill('0') << std::setw(16)
                      << static_cast<long long>(chunk) << ")\n";
        }

        /* If we added the maximum number of values to 'chunk', the sum of
           the sign and exponent parts (all the same, equal to the index) will
           have overflowed out the top, leaving only the sum of the mantissas.
           If the count of how many more terms we could have summed is greater
           than zero, we therefore add this count times the index (shifted to
           the position of the sign and exponent) to get the unwanted bits to
           overflow out the top. */
        if (count > 0)
        {
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
        if (exp != 0)
        {
            mid_chunk += static_cast<xsum_lchunk>((1 << XSUM_LCOUNT_BITS) - count)
                         << (XSUM_MANTISSA_BITS - XSUM_LOW_MANTISSA_BITS + low_exp);
        }

        xsum_uint const high_chunk = mid_chunk >> XSUM_LOW_MANTISSA_BITS;
        mid_chunk &= XSUM_LOW_MANTISSA_MASK;

        auto lacc_sacc = _lacc->sacc.get();

        if (xsum_debug)
        {
            std::cout << "chunk div: low ";
            pbinary_int64(low_chunk, 64);
            std::cout << "\n"
                      << "           mid ";
            pbinary_int64(mid_chunk, 64);
            std::cout << "\n"
                      << "          high ";
            pbinary_int64(high_chunk, 64);
            std::cout << "\n";

            /* Add or subtract the three parts of the mantissa from three small
               accumulator chunks, according to the sign that is part of the index. */
            std::cout << "Small chunks "
                      << static_cast<int>(high_exp) << ", "
                      << static_cast<int>(high_exp) + 1 << ", "
                      << static_cast<int>(high_exp) + 2
                      << " before add or subtract:\n";
            pbinary_int64(lacc_sacc->chunk[high_exp], 64);
            std::cout << "\n";
            pbinary_int64(lacc_sacc->chunk[high_exp + 1], 64);
            std::cout << "\n";
            pbinary_int64(lacc_sacc->chunk[high_exp + 2], 64);
            std::cout << "\n";
        }

        if (ix & (1 << XSUM_EXP_BITS))
        {
            lacc_sacc->chunk[high_exp] -= low_chunk;
            lacc_sacc->chunk[high_exp + 1] -= mid_chunk;
            lacc_sacc->chunk[high_exp + 2] -= high_chunk;
        }
        else
        {
            lacc_sacc->chunk[high_exp] += low_chunk;
            lacc_sacc->chunk[high_exp + 1] += mid_chunk;
            lacc_sacc->chunk[high_exp + 2] += high_chunk;
        }

        if (xsum_debug)
        {
            std::cout << "Small chunks "
                      << static_cast<int>(high_exp) << ", "
                      << static_cast<int>(high_exp) + 1 << ", "
                      << static_cast<int>(high_exp) + 2
                      << " after add or subtract:\n";
            pbinary_int64(lacc_sacc->chunk[high_exp], 64);
            std::cout << "\n";
            pbinary_int64(lacc_sacc->chunk[high_exp + 1], 64);
            std::cout << "\n";
            pbinary_int64(lacc_sacc->chunk[high_exp + 2], 64);
            std::cout << "\n";
        }

        /* The above additions/subtractions reduce by one the number we can
           do before we need to do carry propagation again. */
        lacc_sacc->adds_until_propagate -= 1;
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

inline void xsum_large::add_value_inf_nan(xsum_expint const ix, xsum_lchunk const uintv)
{
    if ((ix & XSUM_EXP_MASK) == XSUM_EXP_MASK)
    {
        _lacc->sacc.add_inf_nan(uintv);
    }
    else
    {
        add_lchunk_to_small(ix);
        _lacc->count[ix] -= 1;
        _lacc->chunk[ix] += uintv;
    }
}

inline xsum_large_accumulator *xsum_large::get() { return _lacc.get(); }

// Helper functions

/* PRINT 64-BIT INTEGER IN BINARY.  Prints only the low-order n bits. */
void pbinary_int64(std::int64_t const v, int const n)
{
    for (int i = n - 1; i >= 0; --i)
    {
        std::cout << static_cast<int>((v >> i) & 1);
    }
}

/* PRINT DOUBLE-PRECISION FLOATING POINT VALUE IN BINARY. */
void pbinary_double(double const d)
{
    union
    {
        double f;
        std::int64_t i;
    } u;

    std::int64_t exp;

    u.f = d;

    std::cout << (u.i < 0 ? "- " : "+ ");

    exp = (u.i >> 52) & 0x7ff;

    pbinary_int64(exp, 11);

    if (exp == 0)
    {
        std::cout << " (denorm) ";
    }
    else if (exp == 0x7ff)
    {
        std::cout << " (InfNaN) ";
    }
    else
    {
        std::cout << " (+" << std::setfill('0') << std::setw(6) << static_cast<int>(exp - 1023) << ") ";
    }

    pbinary_int64(u.i & 0xfffffffffffffL, 52);
}

/* SUM A VECTOR WITH DOUBLE FP ACCUMULATOR. */
xsum_flt xsum_sum_double(xsum_flt const *__restrict__ vec, xsum_length const n)
{
    double s = 0.0;
    xsum_length j = 3;

    for (; j < n; j += 4)
    {
        s += vec[j - 3];
        s += vec[j - 2];
        s += vec[j - 1];
        s += vec[j];
    }

    j -= 3;
    for (; j < n; ++j)
    {
        s += vec[j];
    }

    return static_cast<xsum_flt>(s);
}

/* SUM A VECTOR WITH FLOAT128 ACCUMULATOR. */

#ifdef FLOAT128
#include <quadmath.h>

xsum_flt xsum_sum_float128(xsum_flt const *__restrict__ vec, xsum_length const n)
{
    __float128 s = 0.0;
    for (xsum_length j = 0; j < n; j++)
    {
        s += vec[j];
    }
    return static_cast<xsum_flt>(s);
}

#endif

/* SUM A VECTOR WITH DOUBLE FP, NOT IN ORDER. */

xsum_flt xsum_sum_double_not_ordered(xsum_flt const *__restrict__ vec, xsum_length const n)
{
    double s1 = 0.0;
    double s2 = 0.0;

    xsum_length j;
    for (j = 1; j < n; j += 2)
    {
        s1 += vec[j - 1];
        s2 += vec[j];
    }

    if (j == n)
    {
        s1 += vec[j - 1];
    }

    return static_cast<xsum_flt>(s1 + s2);
}

/* SUM A VECTOR WITH KAHAN'S METHOD. */

xsum_flt xsum_sum_kahan(xsum_flt const *__restrict__ vec, xsum_length const n)
{
    double t;
    double y;
    double s = 0.0;
    double c = 0.0;

    if (OPT_KAHAN_SUM)
    {
        for (xsum_length j = 1; j < n; j += 2)
        {
            y = vec[j - 1] - c;
            t = s;
            s += y;
            c = (s - t) - y;

            y = vec[j] - c;
            t = s;
            s += y;
            c = (s - t) - y;
        }

        for (xsum_length j = j - 1; j < n; ++j)
        {
            y = vec[j] - c;
            t = s;
            s += y;
            c = (s - t) - y;
        }
    }
    else
    {
        for (xsum_length j = 0; j < n; ++j)
        {
            y = vec[j] - c;
            t = s;
            s += y;
            c = (s - t) - y;
        }
    }

    return static_cast<xsum_flt>(s);
}

/* SQUARED NORM OF A VECTOR WITH DOUBLE FP ACCUMULATOR. */

xsum_flt xsum_sqnorm_double(xsum_flt const *__restrict__ vec, xsum_length const n)
{
    double s = 0.0;

    if (OPT_SIMPLE_SQNORM)
    {
        for (xsum_length j = 3; j < n; j += 4)
        {
            double const a = vec[j - 3];
            double const b = vec[j - 2];
            double const c = vec[j - 1];
            double const d = vec[j];

            s += a * a;
            s += b * b;
            s += c * c;
            s += d * d;
        }

        for (xsum_length j = j - 3; j < n; ++j)
        {
            double const a = vec[j];
            s += a * a;
        }
    }
    else
    {
        for (xsum_length j = 0; j < n; ++j)
        {
            double const a = vec[j];
            s += a * a;
        }
    }

    return static_cast<xsum_flt>(s);
}

/* SQUARED NORM OF A VECTOR WITH DOUBLE FP, NOT IN ORDER. */

xsum_flt xsum_sqnorm_double_not_ordered(xsum_flt const *__restrict__ vec, xsum_length const n)
{
    double s1 = 0.0;
    double s2 = 0.0;

    xsum_length j;
    for (j = 1; j < n; j += 2)
    {
        double const a = vec[j - 1];
        double const b = vec[j];
        s1 += a * a;
        s2 += b * b;
    }

    if (j == n)
    {
        double const a = vec[j - 1];
        s1 += a * a;
    }

    return static_cast<xsum_flt>(s1 + s2);
}

/* DOT PRODUCT OF VECTORS WITH DOUBLE FP ACCUMULATOR. */

xsum_flt xsum_dot_double(xsum_flt const *vec1, xsum_flt const *vec2, xsum_length const n)
{
    double s = 0.0;
    if (OPT_SIMPLE_DOT)
    {
        for (xsum_length j = 3; j < n; j += 4)
        {
            s += vec1[j - 3] * vec2[j - 3];
            s += vec1[j - 2] * vec2[j - 2];
            s += vec1[j - 1] * vec2[j - 1];
            s += vec1[j] * vec2[j];
        }

        for (xsum_length j = j - 3; j < n; j++)
        {
            s += vec1[j] * vec2[j];
        }
    }
    else
    {
        for (xsum_length j = 0; j < n; j++)
        {
            s += vec1[j] * vec2[j];
        }
    }

    return static_cast<xsum_flt>(s);
}

/* DOT PRODUCT OF VECTORS WITH DOUBLE FP, NOT IN ORDER. */

xsum_flt xsum_dot_double_not_ordered(xsum_flt const *vec1, xsum_flt const *vec2, xsum_length const n)
{
    double s1 = 0.0;
    double s2 = 0.0;

    xsum_length j;
    for (j = 1; j < n; j += 2)
    {
        s1 += vec1[j - 1] * vec2[j - 1];
        s2 += vec1[j] * vec2[j];
    }

    if (j == n)
    {
        s1 += vec1[j - 1] * vec2[j - 1];
    }

    return static_cast<xsum_flt>(s1 + s2);
}

#ifdef __restrict__
#undef __restrict__
#endif