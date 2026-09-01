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
    operation_id="hot_pixel",
    operation_version="1.0.0",
    display_name="Local hot-pixel correction",
    description="Replace strong local median residuals independently in each band.",
    category="preprocessing",
    input_kind="stack",
)


class Params(BaseModel):
    window_size: int = Field(
        default=3,
        ge=3,
        description="Odd median-filter window used to estimate the local background.",
    )
    threshold_sigma: float = Field(
        default=6.0,
        gt=0,
        description=(
            'Robust-sigma multiplier used to classify large local residuals as candidate '
            'hot pixels.'
        ),
    )
    minimum_deviation: float = Field(
        default=0.0,
        ge=0,
        description=(
            'Absolute lower bound for the residual threshold used to identify candidate '
            'hot pixels.'
        ),
    )
    boundary_mode: Literal["reflect", "nearest", "mirror", "wrap"] = Field(
        default="reflect",
        description="Boundary handling mode passed to the median filter.",
    )

    @model_validator(mode="after")
    def validate_window(self) -> "Params":
        if self.window_size % 2 == 0:
            raise ValueError("window_size must be odd.")
        return self


def run(
    stack: Stack,
    params: Params | dict[str, Any] | None = None,
    **kwargs: Any,
) -> Result:
    """Replace strong local median residuals as candidate hot pixels.

    Parameters
    ----------
    stack : Stack
        Input multispectral stack. Correction is performed independently for
        each band.
    params : Params or dict, optional
        Validated correction configuration.
    **kwargs
        Individual operation parameters supplied directly as keyword
        arguments. Their types, defaults, constraints, and descriptions
        are defined by :class:`Params`.

    Returns
    -------
    Result
        Corrected stack. The number of replaced pixels per band is recorded in
        ``additional_data["replaced_pixels_per_band"]``.

    Notes
    -----
    This is a generic median-residual correction method. Parameters should be
    validated for the acquisition system and detector characteristics.
    """
    config = resolve_params(Params, params, kwargs)
    output = stack.data.copy()
    replaced: list[int] = []

    for index in range(stack.n_bands):
        image = stack.data[index].astype(np.float32)
        local_median = median_filter(
            image, size=config.window_size, mode=config.boundary_mode
        )
        residual = image - local_median
        center = float(np.median(residual))
        mad = float(np.median(np.abs(residual - center)))
        robust_sigma = 1.4826 * mad
        threshold = max(config.minimum_deviation, config.threshold_sigma * robust_sigma)
        if threshold <= 0:
            mask = np.zeros(image.shape, dtype=bool)
        else:
            mask = np.abs(residual - center) > threshold
        output[index][mask] = local_median[mask].astype(output.dtype, copy=False)
        replaced.append(int(mask.sum()))

    return Result(
        output=stack.with_data(output),
        operation_id=INFO.operation_id,
        operation_version=INFO.operation_version,
        operation_info=INFO,
        parameters=config.model_dump(mode="json"),
        additional_data={"replaced_pixels_per_band": replaced},
        metadata={
            "validation_note": "Generic median-residual correction; validate parameters for the acquisition system."
        },
    )
