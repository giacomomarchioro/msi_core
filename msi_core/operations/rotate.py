from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ..attribution import OperationInfo
from ..parameters import resolve_params
from ..result import Result
from ..stack import Stack

INFO = OperationInfo(
    operation_id="rotate",
    operation_version="1.0.0",
    display_name="Rotate by quarter turns",
    description="Rotate every band by a multiple of 90 degrees counter-clockwise.",
    category="preprocessing",
    input_kind="stack",
)


class Params(BaseModel):
    quarter_turns: int = Field(
        default=0,
        ge=-3,
        le=3,
        description=(
            'Number of 90-degree counter-clockwise rotations; negative values rotate '
            'clockwise.'
        ),
    )


def run(
    stack: Stack,
    params: Params | dict[str, Any] | None = None,
    **kwargs: Any,
) -> Result:
    """Rotate every band by a multiple of 90 degrees counter-clockwise.

    Parameters
    ----------
    stack : Stack
        Input multispectral stack.
    params : Params or dict, optional
        Validated rotation configuration.
    **kwargs
        Individual operation parameters supplied directly as keyword
        arguments. Their types, defaults, constraints, and descriptions
        are defined by :class:`Params`.

    Returns
    -------
    Result
        Rotated stack with band metadata preserved.
    """
    config = resolve_params(Params, params, kwargs)
    output = np.rot90(stack.data, k=config.quarter_turns, axes=(1, 2)).copy()
    return Result(
        output=stack.with_data(output),
        operation_id=INFO.operation_id,
        operation_version=INFO.operation_version,
        operation_info=INFO,
        parameters=config.model_dump(mode="json"),
    )
