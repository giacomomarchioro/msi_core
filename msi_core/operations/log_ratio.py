from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ..attribution import OperationInfo
from ..parameters import resolve_params
from ..result import Result
from ..stack import Band, BandMetadata, Stack

INFO = OperationInfo(
    operation_id="log_ratio",
    operation_version="1.0.0",
    display_name="Log band ratio",
    description="Compute the logarithm of the ratio between two selected bands.",
    category="arithmetic",
    input_kind="band_pair",
)


class Params(BaseModel):
    offset: float = Field(
        default=0.5,
        description="Constant added to both input bands before computing the ratio.",
    )
    epsilon: float = Field(
        default=1e-6,
        gt=0,
        description="Positive lower bound used when clip_nonpositive is enabled.",
    )
    clip_nonpositive: bool = Field(
        default=True,
        description=(
            'Clip shifted inputs to at least epsilon before taking the logarithm; '
            'otherwise non-positive values raise ValueError.'
        ),
    )


def run(
    band_a: Band,
    band_b: Band,
    params: Params | dict[str, Any] | None = None,
    **kwargs: Any,
) -> Result:
    """Compute the natural logarithm of the ratio between two bands.

    Parameters
    ----------
    band_a : Band
        Numerator band.
    band_b : Band
        Denominator band. Must match ``band_a`` spatially.
    params : Params or dict, optional
        Validated log-ratio configuration.
    **kwargs
        Individual operation parameters supplied directly as keyword
        arguments. Their types, defaults, constraints, and descriptions
        are defined by :class:`Params`.

    Returns
    -------
    Result
        Single-band float32 stack containing the natural log-ratio.

    Raises
    ------
    ValueError
        If band shapes differ, or if non-positive values are encountered while
        ``clip_nonpositive`` is false.
    """
    config = resolve_params(Params, params, kwargs)
    if band_a.data.shape != band_b.data.shape:
        raise ValueError("Band shapes must match.")
    numerator = band_a.data.astype(np.float32) + config.offset
    denominator = band_b.data.astype(np.float32) + config.offset
    if config.clip_nonpositive:
        numerator = np.maximum(numerator, config.epsilon)
        denominator = np.maximum(denominator, config.epsilon)
    elif np.any(numerator <= 0) or np.any(denominator <= 0):
        raise ValueError("Log-ratio inputs must be positive after applying offset.")
    output = np.log(numerator / denominator)
    name_a = band_a.name or f"band_{band_a.metadata.index}"
    name_b = band_b.name or f"band_{band_b.metadata.index}"
    stack = Stack(
        data=output[np.newaxis, ...].astype(np.float32),
        bands=[BandMetadata(index=0, name=f"log_{name_a}_div_{name_b}")],
        metadata={"input_bands": [band_a.metadata.model_dump(), band_b.metadata.model_dump()]},
    )
    return Result(
        output=stack,
        operation_id=INFO.operation_id,
        operation_version=INFO.operation_version,
        operation_info=INFO,
        parameters=config.model_dump(mode="json"),
    )
