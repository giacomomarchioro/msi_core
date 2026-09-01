from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ..attribution import OperationInfo
from ..parameters import resolve_params
from ..result import Result
from ..stack import Band, BandMetadata, Stack

INFO = OperationInfo(
    operation_id="subtraction",
    operation_version="1.0.0",
    display_name="Band subtraction",
    description="Subtract the second selected band from the first selected band.",
    category="arithmetic",
    input_kind="band_pair",
)


class Params(BaseModel):
    absolute: bool = Field(
        default=False,
        description="Return the absolute value of the difference after applying the offset.",
    )
    offset: float = Field(
        default=0.0,
        description=(
            'Constant added after subtraction and before optional absolute-value '
            'conversion.'
        ),
    )


def run(
    band_a: Band,
    band_b: Band,
    params: Params | dict[str, Any] | None = None,
    **kwargs: Any,
) -> Result:
    """Subtract one band from another.

    Parameters
    ----------
    band_a : Band
        Minuend band.
    band_b : Band
        Subtrahend band. Must have the same spatial shape as ``band_a``.
    params : Params or dict, optional
        Validated subtraction configuration.
    **kwargs
        Individual operation parameters supplied directly as keyword
        arguments. Their types, defaults, constraints, and descriptions
        are defined by :class:`Params`.

    Returns
    -------
    Result
        Single-band float32 stack containing ``band_a - band_b + offset``.

    Raises
    ------
    ValueError
        If the two bands have different spatial shapes.
    """
    config = resolve_params(Params, params, kwargs)
    _validate_pair(band_a, band_b)
    output = band_a.data.astype(np.float32) - band_b.data.astype(np.float32) + config.offset
    if config.absolute:
        output = np.abs(output)
    name_a = band_a.name or f"band_{band_a.metadata.index}"
    name_b = band_b.name or f"band_{band_b.metadata.index}"
    stack = Stack(
        data=output[np.newaxis, ...],
        bands=[BandMetadata(index=0, name=f"{name_a}_minus_{name_b}")],
        metadata={"input_bands": [band_a.metadata.model_dump(), band_b.metadata.model_dump()]},
    )
    return Result(
        output=stack,
        operation_id=INFO.operation_id,
        operation_version=INFO.operation_version,
        operation_info=INFO,
        parameters=config.model_dump(mode="json"),
    )


def _validate_pair(band_a: Band, band_b: Band) -> None:
    if band_a.data.shape != band_b.data.shape:
        raise ValueError(
            f"Band shapes must match; received {band_a.data.shape} and {band_b.data.shape}."
        )
