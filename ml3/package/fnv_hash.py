"""
Implementation of Fowler/Noll/Vo hash algorithm in pure Python.
See http://isthe.com/chongo/tech/comp/fnv/
"""
# from: https://github.com/znerol/py-fnvhash/
# LICENCE: MIT
# Copyright (c) 2015 Lorenz Schori

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys

# ----------------------------------------------------------------------------
# A fast and effictive hash for the cases when crypto-grade strength is
# not important; e.g., bucketing items. Use FnvHash class below.

# ----------------------------------------------------------------------------

if sys.version_info[0] == 3:
    _get_byte = lambda c: c
else:
    _get_byte = ord

# ----------------------------------------------------------------------------

FNV_32_PRIME = 0x01000193
FNV_64_PRIME = 0x100000001B3

FNV0_32_INIT = 0
FNV0_64_INIT = 0
FNV1_32_INIT = 0x811C9DC5
FNV1_32A_INIT = FNV1_32_INIT
FNV1_64_INIT = 0xCBF29CE484222325
FNV1_64A_INIT = FNV1_64_INIT

# ----------------------------------------------------------------------------

def fnv(data, hval_init, fnv_prime, fnv_size):
    """
    Core FNV hash algorithm used in FNV0 and FNV1.
    """
    assert isinstance(data, bytes)

    hval = hval_init
    for byte in data:
        hval = (hval * fnv_prime) % fnv_size
        hval = hval ^ _get_byte(byte)
    return hval

# ----------------------------------------------------------------------------

def fnva(data, hval_init, fnv_prime, fnv_size):
    """
    Alternative FNV hash algorithm used in FNV-1a.
    """
    assert isinstance(data, bytes)

    hval = hval_init
    for byte in data:
        hval = hval ^ _get_byte(byte)
        hval = (hval * fnv_prime) % fnv_size
    return hval

# ----------------------------------------------------------------------------

def fnv0_32(data, hval_init=FNV0_32_INIT):
    """
    Returns the 32 bit FNV-0 hash value for the given data.
    """
    return fnv(data, hval_init, FNV_32_PRIME, 2 ** 32)

# ----------------------------------------------------------------------------

def fnv1_32(data, hval_init=FNV1_32_INIT):
    """
    Returns the 32 bit FNV-1 hash value for the given data.
    """
    return fnv(data, hval_init, FNV_32_PRIME, 2 ** 32)

# ----------------------------------------------------------------------------

def fnv1a_32(data, hval_init=FNV1_32_INIT):
    """
    Returns the 32 bit FNV-1a hash value for the given data.
    """
    return fnva(data, hval_init, FNV_32_PRIME, 2 ** 32)

# ----------------------------------------------------------------------------

def fnv0_64(data, hval_init=FNV0_64_INIT):
    """
    Returns the 64 bit FNV-0 hash value for the given data.
    """
    return fnv(data, hval_init, FNV_64_PRIME, 2 ** 64)

# ----------------------------------------------------------------------------

def fnv1_64(data, hval_init=FNV1_64_INIT):
    """
    Returns the 64 bit FNV-1 hash value for the given data.
    """
    return fnv(data, hval_init, FNV_64_PRIME, 2 ** 64)

# ----------------------------------------------------------------------------

def fnv1a_64(data, hval_init=FNV1_64_INIT):
    """
    Returns the 64 bit FNV-1a hash value for the given data.
    """
    return fnva(data, hval_init, FNV_64_PRIME, 2 ** 64)

# ----------------------------------------------------------------------------

class FnvHash:

    # ------------------------------------------------------------------------
    # Assumes s is either a string, or a number (int, float, etc.)

    @classmethod
    def compute(self, s, seed=FNV1_32_INIT):
        if isinstance(s, str):
            return self.compute_s(s, seed)

        return self.compute_n(s, seed)

    # ------------------------------------------------------------------------

    @classmethod
    def compute_s(self, s, seed=FNV1_32_INIT):
        return fnv1_32(s.encode(), hval_init=seed)

    # ------------------------------------------------------------------------

    @classmethod
    def compute_n_core(self, n, seed):
        return fnv1_32(n.to_bytes(max(4, (n.bit_length() + 7)) // 8, 'big'), hval_init=seed)

    # ------------------------------------------------------------------------

    @classmethod
    def compute_n(self, n, seed=FNV1_32_INIT):
        h1 = self.compute_n_core(n, seed)
        return self.compute_n_core(n, h1)
