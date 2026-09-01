from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy.ndimage import median_filter

from ..attribution import OperationInfo
from ..parameters import resolve_params
from ..result import Result
from ..stack import Stack

INFO = OperationInfo(
    operation_id="denoise",
    operation_version="1.0.0",
    display_name="Median denoise",
    description="Apply a spatial median filter independently to every band.",
    category="preprocessing",
    input_kind="stack",
)


class Params(BaseModel):
    size: int = Field(
        default=3,
        ge=1,
        description="Odd spatial kernel width used by the median filter.",
    )
    boundary_mode: Literal["reflect", "nearest", "mirror", "wrap"] = Field(
        default="reflect",
        description="Boundary handling mode passed to scipy.ndimage.median_filter.",
    )

    @model_validator(mode="after")
    def validate_size(self) -> "Params":
        if self.size % 2 == 0:
            raise ValueError("Median filter size must be odd.")
        return self


def run(
    stack: Stack,
    params: Params | dict[str, Any] | None = None,
    **kwargs: Any,
) -> Result:
    """Apply a spatial median filter independently to each band.

    Parameters
    ----------
    stack : Stack
        Input multispectral stack.
    params : Params or dict, optional
        Validated denoising configuration.
    **kwargs
        Individual operation parameters supplied directly as keyword
        arguments. Their types, defaults, constraints, and descriptions
        are defined by :class:`Params`.

    Returns
    -------
    Result
        Stack with the same bands and dtype as the input after median filtering.

    Raises
    ------
    ValueError
        If ``size`` is even.
    """
    config = resolve_params(Params, params, kwargs)
    output = np.empty_like(stack.data)
    for index in range(stack.n_bands):
        output[index] = median_filter(
            stack.data[index], size=config.size, mode=config.boundary_mode
        )
    return Result(
        output=stack.with_data(output),
        operation_id=INFO.operation_id,
        operation_version=INFO.operation_version,
        operation_info=INFO,
        parameters=config.model_dump(mode="json"),
    )
