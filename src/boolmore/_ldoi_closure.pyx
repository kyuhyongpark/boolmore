# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython batch / single-seed LDOI closure over CSR watcher tables."""

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset


cdef inline void _force(
    signed char *state,
    int *queue,
    int *q_len,
    int node_id,
    signed char value,
) noexcept nogil:
    if state[node_id] != -1:
        return
    state[node_id] = value
    queue[q_len[0]] = node_id
    q_len[0] += 1


cdef void _closure_one(
    int n,
    int n_imp,
    int *targets,
    signed char *values,
    int *lit_counts,
    int *watch_ptr,
    int *watch_idx,
    int *empty_imp,
    int n_empty,
    int *seed_node,
    signed char *seed_value,
    int n_seed,
    signed char *out_row,
    int *residuals,
    int *queue,
    unsigned char *seen,
) noexcept nogil:
    cdef int i, node_id, imp_idx, head, q_len, start, end, value
    cdef signed char *state = out_row

    memset(state, -1, n)
    memcpy(residuals, lit_counts, n_imp * sizeof(int))
    q_len = 0

    for i in range(n_seed):
        state[seed_node[i]] = seed_value[i]

    for i in range(n_empty):
        imp_idx = empty_imp[i]
        _force(state, queue, &q_len, targets[imp_idx], values[imp_idx])

    for node_id in range(n):
        if state[node_id] != -1:
            queue[q_len] = node_id
            q_len += 1

    memset(seen, 0, n)
    head = 0
    for i in range(q_len):
        node_id = queue[i]
        if not seen[node_id]:
            seen[node_id] = 1
            queue[head] = node_id
            head += 1
    q_len = head

    head = 0
    while head < q_len:
        node_id = queue[head]
        head += 1
        value = state[node_id]
        start = watch_ptr[node_id * 2 + value]
        end = watch_ptr[node_id * 2 + value + 1]
        for i in range(start, end):
            imp_idx = watch_idx[i]
            residuals[imp_idx] -= 1
            if residuals[imp_idx] == 0:
                _force(
                    state, queue, &q_len,
                    targets[imp_idx], values[imp_idx],
                )


def batch_closure(
    int n,
    int n_imp,
    int[:] targets,
    signed char[:] values,
    int[:] lit_counts,
    int[:] watch_ptr,
    int[:] watch_idx,
    int[:] empty_imp,
    int[:] seed_ptr,
    int[:] seed_node,
    signed char[:] seed_value,
    signed char[:] out_state,
):
    """Run event-driven LDOI for each seed; write ``out_state`` as -1/0/1.

    ``out_state`` must have length ``n_seeds * n``. ``seed_ptr`` has length
    ``n_seeds + 1`` with half-open ranges into ``seed_node`` / ``seed_value``.
    """
    cdef int n_seeds = seed_ptr.shape[0] - 1
    cdef int n_empty = empty_imp.shape[0]
    cdef int s, seed_start, seed_end, n_seed
    cdef int *residuals
    cdef int *queue
    cdef unsigned char *seen

    if n < 0 or n_imp < 0:
        raise ValueError("n and n_imp must be non-negative")
    if n_seeds <= 0 or n == 0:
        return
    if out_state.shape[0] < n_seeds * n:
        raise ValueError("out_state shorter than n_seeds * n")
    if watch_ptr.shape[0] < 2 * n + 1:
        raise ValueError("watch_ptr shorter than 2*n+1")
    if targets.shape[0] < n_imp or values.shape[0] < n_imp or lit_counts.shape[0] < n_imp:
        raise ValueError("implicant arrays shorter than n_imp")

    # Pre-dedupe enqueue can hold empty-forced + seed echoes (up to 2n).
    residuals = <int *>malloc(n_imp * sizeof(int) if n_imp > 0 else sizeof(int))
    if residuals is NULL:
        raise MemoryError("ldoi residual allocation failed")
    queue = <int *>malloc(2 * n * sizeof(int))
    if queue is NULL:
        free(residuals)
        raise MemoryError("ldoi queue allocation failed")
    seen = <unsigned char *>malloc(n)
    if seen is NULL:
        free(residuals)
        free(queue)
        raise MemoryError("ldoi seen allocation failed")

    try:
        for s in range(n_seeds):
            seed_start = seed_ptr[s]
            seed_end = seed_ptr[s + 1]
            n_seed = seed_end - seed_start
            _closure_one(
                n,
                n_imp,
                &targets[0] if n_imp > 0 else NULL,
                &values[0] if n_imp > 0 else NULL,
                &lit_counts[0] if n_imp > 0 else NULL,
                &watch_ptr[0],
                &watch_idx[0] if watch_idx.shape[0] > 0 else NULL,
                &empty_imp[0] if n_empty > 0 else NULL,
                n_empty,
                &seed_node[seed_start] if n_seed > 0 else NULL,
                &seed_value[seed_start] if n_seed > 0 else NULL,
                n_seed,
                &out_state[s * n],
                residuals,
                queue,
                seen,
            )
    finally:
        free(residuals)
        free(queue)
        free(seen)
