# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython kernel for :func:`boolmore.boolean_functions.get_uni_rr`."""

from libc.stdlib cimport malloc, free
from cpython.bytes cimport PyBytes_FromStringAndSize


def get_uni_rr(str rr, bint max=True) -> str:
    """Canonical maximal / minimal rr (mask-order, index == mask).

    Same semantics as the pure-Python implementation: for each active
    implicant mask ``i``, fill every proper superset position.
    """
    cdef Py_ssize_t n = len(rr)
    cdef Py_ssize_t i, position
    cdef char fill_value
    cdef char c_one = 49  # '1'
    cdef char c_zero = 48  # '0'
    cdef char *uni
    cdef unsigned char *modified
    cdef bytes raw
    cdef const char *src
    cdef bytes out_b

    if n <= 1:
        return rr

    raw = rr.encode("ascii")
    src = raw
    uni = <char *>malloc(n)
    if uni is NULL:
        raise MemoryError("get_uni_rr buffer allocation failed")
    modified = <unsigned char *>malloc(n)
    if modified is NULL:
        free(uni)
        raise MemoryError("get_uni_rr flag allocation failed")

    try:
        for i in range(n):
            uni[i] = src[i]
            modified[i] = 0

        fill_value = c_one if max else c_zero

        for i in range(n):
            if uni[i] != c_one:
                continue
            if modified[i]:
                continue
            for position in range(i + 1, n):
                if (position & i) == i:
                    modified[position] = 1
                    uni[position] = fill_value

        out_b = PyBytes_FromStringAndSize(uni, n)
        return out_b.decode("ascii")
    finally:
        free(uni)
        free(modified)
