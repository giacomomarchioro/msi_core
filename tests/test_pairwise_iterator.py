import pytest

from msi_core import apply_pairwise, iter_pairwise
from msi_core.operations import subtraction


@pytest.mark.parametrize("ordered", [False, True])
def test_iterator_matches_existing_api(small_stack, ordered):
    expected = apply_pairwise(subtraction.run, small_stack, ordered=ordered, offset=3)
    actual = list(iter_pairwise(subtraction.run, small_stack, ordered=ordered, offset=3))
    assert [(x.first_index, x.second_index) for x in actual] == [
        (x.first_index, x.second_index) for x in expected
    ]
    for a, b in zip(actual, expected):
        assert (a.result.output.data == b.result.output.data).all()


def test_iteration_is_lazy_and_passes_params(small_stack):
    calls = []

    def operation(a, b, *, params):
        calls.append((a.metadata.index, b.metadata.index, params))
        return subtraction.run(a, b, params=params)

    iterator = iter_pairwise(operation, small_stack, params={"offset": 2})
    assert calls == []
    first = next(iterator)
    assert calls == [(0, 1, {"offset": 2})]
    assert (first.first_index, first.second_index) == (0, 1)


def test_empty_iterator_for_single_band(small_stack):
    assert list(iter_pairwise(subtraction.run, small_stack.select([0]))) == []


def test_operation_exception_propagates_during_iteration(small_stack):
    def operation(a, b):
        raise ValueError("failure")

    iterator = iter_pairwise(operation, small_stack)
    with pytest.raises(ValueError, match="failure"):
        next(iterator)
