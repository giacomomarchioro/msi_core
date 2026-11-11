from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Any, Callable, Iterator

from .result import Result
from .stack import Band, Stack


@dataclass(frozen=True, slots=True)
class PairwiseResult:
    first_index: int
    second_index: int
    result: Result


def iter_pairwise(
    function: Callable[..., Result],
    stack: Stack,
    *,
    params: Any = None,
    ordered: bool = False,
    **kwargs: Any,
) -> Iterator[PairwiseResult]:
    """Lazily apply an operation to each pair, excluding self-pairs.

    Unordered pairs follow stack order; ordered pairs include both directions.
    Computation and errors occur as the iterator is consumed. The caller is
    responsible for persisting each result if it must not accumulate in memory.
    """

    pairs = permutations(range(stack.n_bands), 2) if ordered else combinations(range(stack.n_bands), 2)
    for first, second in pairs:
        if params is None:
            result = function(stack[first], stack[second], **kwargs)
        else:
            result = function(stack[first], stack[second], params=params, **kwargs)
        yield PairwiseResult(first_index=first, second_index=second, result=result)


def apply_pairwise(
    function: Callable[..., Result],
    stack: Stack,
    *,
    params: Any = None,
    ordered: bool = False,
    **kwargs: Any,
) -> list[PairwiseResult]:
    """Apply an operation to every pair and return all results in memory.

    For large stacks use :func:`iter_pairwise` and persist each result as it is
    produced instead.
    """
    return list(iter_pairwise(function, stack, params=params, ordered=ordered, **kwargs))
