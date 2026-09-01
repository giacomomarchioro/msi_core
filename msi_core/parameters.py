from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import BaseModel

from .errors import ParameterConflictError

P = TypeVar("P", bound=BaseModel)


def resolve_params(
    model: type[P],
    params: P | Mapping[str, Any] | None,
    kwargs: Mapping[str, Any],
) -> P:
    """Validate operation parameters from either an object/dict or keyword arguments."""

    if params is not None and kwargs:
        raise ParameterConflictError(
            "Pass either 'params' or keyword parameters, not both."
        )

    if params is None:
        return model.model_validate(dict(kwargs))
    if isinstance(params, model):
        return params
    return model.model_validate(dict(params))
