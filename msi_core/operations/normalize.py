from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from ..attribution import OperationInfo
from ..parameters import resolve_params
from ..result import Result
from ..stack import Stack

INFO = OperationInfo(
    operation_id="normalize",
    operation_version="1.0.0",
    display_name="Percentile normalization",
    description="Normalize a stack globally or independently per band using percentile bounds.",
    category="preprocessing",
    input_kind="stack",
)


class Params(BaseModel):
    low_percentile: float = Field(
        default=1.0,
        ge=0.0,
        le=100.0,
        description="Lower percentile used as the normalization black point.",
    )
    high_percentile: float = Field(
        default=99.0,
        ge=0.0,
        le=100.0,
        description=(
            'Upper percentile used as the normalization white point; must exceed '
            'low_percentile.'
        ),
    )
    mode: Literal["global", "per_band"] = Field(
        default="per_band",
        description=(
            'Use one percentile range for the whole stack (global) or estimate a separate '
            'range for each band (per_band).'
        ),
    )
    clip: bool = Field(default=True, description="Clip normalized values to the [0, 1] range.")

    @model_validator(mode="after")
    def validate_percentiles(self) -> "Params":
        if self.high_percentile <= self.low_percentile:
            raise ValueError("high_percentile must be greater than low_percentile.")
        return self


def run(
    stack: Stack,
    params: Params | dict[str, Any] | None = None,
    **kwargs: Any,
) -> Result:
    """Normalize multispectral intensities using percentile bounds.

    Parameters
    ----------
    stack : Stack
        Input multispectral stack.
    params : Params or dict, optional
        Validated normalization configuration. Parameters may alternatively be
        supplied directly as keyword arguments.
    **kwargs
        Individual operation parameters supplied directly as keyword
        arguments. Their types, defaults, constraints, and descriptions
        are defined by :class:`Params`.

    Returns
    -------
    Result
        Normalized float32 stack. Applied percentile bounds are stored in
        ``additional_data["percentile_bounds"]``.

    Notes
    -----
    This operation modifies scientific data. It is distinct from percentile
    stretching used only to create display renders.
    """
    config = resolve_params(Params, params, kwargs)
    data = stack.as_array(dtype=np.float32)
    output = np.empty_like(data, dtype=np.float32)

    if config.mode == "global":
        low, high = np.percentile(data, [config.low_percentile, config.high_percentile])
        denominator = float(high - low)
        if denominator <= 0:
            output.fill(0.0)
        else:
            output[:] = (data - low) / denominator
        bounds: Any = {"low": float(low), "high": float(high)}
    else:
        bounds = []
        for index in range(stack.n_bands):
            low, high = np.percentile(
                data[index], [config.low_percentile, config.high_percentile]
            )
            denominator = float(high - low)
            if denominator <= 0:
                output[index].fill(0.0)
            else:
                output[index] = (data[index] - low) / denominator
            bounds.append({"low": float(low), "high": float(high)})

    if config.clip:
        np.clip(output, 0.0, 1.0, out=output)

    return Result(
        output=stack.with_data(output),
        operation_id=INFO.operation_id,
        operation_version=INFO.operation_version,
        operation_info=INFO,
        parameters=config.model_dump(mode="json"),
        additional_data={"percentile_bounds": bounds},
    )
