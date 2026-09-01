from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Any, Callable

from .result import Result
from .stack import Band, Stack


@dataclass(frozen=True, slots=True)
class PairwiseResult:
    first_index: int
    second_index: int
    result: Result


def apply_pairwise(
    function: Callable[..., Result],
    stack: Stack,
    *,
    params: Any = None,
    ordered: bool = False,
    **kwargs: Any,
) -> list[PairwiseResult]:
    """Apply a two-band operation to all combinations or ordered pairs."""

    pairs = permutations(range(stack.n_bands), 2) if ordered else combinations(range(stack.n_bands), 2)
    items: list[PairwiseResult] = []
    for first, second in pairs:
        if params is None:
            result = function(stack[first], stack[second], **kwargs)
        else:
            result = function(stack[first], stack[second], params=params, **kwargs)
        items.append(PairwiseResult(first_index=first, second_index=second, result=result))
    return items
