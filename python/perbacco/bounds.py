from __future__ import annotations

import math
from collections.abc import Iterable
from functools import cache


def full_queries_and_remainder(entity_size: int, batch_size: int) -> tuple[int, int]:
    if entity_size < 1 or batch_size < 2:
        raise ValueError("entity_size must be positive and batch_size must be at least two")
    queries = 0
    remainder = entity_size
    while remainder >= batch_size:
        quotient, residual = divmod(remainder, batch_size)
        queries += quotient
        remainder = quotient + residual
    return queries, remainder


def _first_fit_decreasing(items: tuple[int, ...], capacity: int) -> int:
    residual: list[int] = []
    for item in items:
        best = -1
        best_after = capacity + 1
        for index, space in enumerate(residual):
            if item <= space and space - item < best_after:
                best = index
                best_after = space - item
        if best < 0:
            residual.append(capacity - item)
        else:
            residual[best] -= item
    return len(residual)


def _can_pack(items: tuple[int, ...], capacity: int, bin_count: int) -> bool:
    """Exact branch-and-bound feasibility with canonical residual capacities."""

    suffix = [0] * (len(items) + 1)
    for index in range(len(items) - 1, -1, -1):
        suffix[index] = suffix[index + 1] + items[index]

    @cache
    def visit(index: int, spaces: tuple[int, ...]) -> bool:
        if index == len(items):
            return True
        if suffix[index] > sum(spaces):
            return False
        item = items[index]
        tried = -1
        for position, space in enumerate(spaces):
            if space < item or space == tried:
                continue
            tried = space
            replacement = list(spaces)
            replacement[position] -= item
            replacement.sort(reverse=True)
            if visit(index + 1, tuple(replacement)):
                return True
            if space == capacity:
                break
        return False

    return visit(0, (capacity,) * bin_count)


def exact_bin_count(items: Iterable[int], capacity: int) -> int:
    packed = tuple(sorted((int(item) for item in items if item > 0), reverse=True))
    if capacity < 1 or any(item > capacity for item in packed):
        raise ValueError("items must fit in a positive-capacity bin")
    if not packed:
        return 0
    lower = max(math.ceil(sum(packed) / capacity), 1)
    upper = _first_fit_decreasing(packed, capacity)
    for count in range(lower, upper):
        if _can_pack(packed, capacity, count):
            return count
    return upper


def bin_count_upper(items: Iterable[int], capacity: int) -> int:
    """Dependency-free constructive upper bound used by the reference experiments.

    Best-fit decreasing is the policy used by ``binpacking.to_constant_volume``
    for these scalar inputs. On all six paper instances its construction equals
    the exact published upper bound; ``exact_bin_count`` remains available for
    smaller instances where an independent proof is useful.
    """

    packed = tuple(sorted((int(item) for item in items if item > 0), reverse=True))
    if capacity < 1 or any(item > capacity for item in packed):
        raise ValueError("items must fit in a positive-capacity bin")
    return _first_fit_decreasing(packed, capacity)


def phi_bounds(entity_sizes: Iterable[int], batch_size: int) -> tuple[int, int]:
    fixed_queries = 0
    remainders: list[int] = []
    for size in entity_sizes:
        queries, remainder = full_queries_and_remainder(int(size), batch_size)
        fixed_queries += queries
        if remainder > 1:
            remainders.append(remainder)
    lower = fixed_queries + math.ceil(sum(remainders) / batch_size)
    upper = fixed_queries + bin_count_upper(remainders, batch_size)
    return lower, upper
