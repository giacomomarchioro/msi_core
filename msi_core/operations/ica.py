from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field
from sklearn.decomposition import FastICA

from ..attribution import OperationInfo
from ..parameters import resolve_params
from ..result import Result
from ..stack import BandMetadata, Stack

INFO = OperationInfo(
    operation_id="ica",
    operation_version="1.0.0",
    display_name="Independent component analysis",
    description="Compute independent-component score images from pixel spectra.",
    category="decomposition",
    input_kind="stack",
)


class Params(BaseModel):
    n_components: int = Field(
        default=8,
        ge=1,
        description=(
            'Number of independent components to estimate; cannot exceed the number of '
            'input bands.'
        ),
    )
    algorithm: Literal["parallel", "deflation"] = Field(
        default="parallel",
        description=(
            'FastICA optimization strategy: estimate components simultaneously (parallel) '
            'or sequentially (deflation).'
        ),
    )
    fun: Literal["logcosh", "exp", "cube"] = Field(
        default="logcosh",
        description="Non-linearity used by FastICA to approximate neg-entropy.",
    )
    max_iter: int = Field(default=400, ge=1, description="Maximum number of FastICA iterations.")
    tolerance: float = Field(
        default=1e-4,
        gt=0,
        description="Convergence tolerance passed to FastICA.",
    )
    random_state: int | None = Field(
        default=0,
        description=(
            'Random seed used to initialize FastICA; use None for non-deterministic '
            'initialization.'
        ),
    )


def run(
    stack: Stack,
    params: Params | dict[str, Any] | None = None,
    **kwargs: Any,
) -> Result:
    """Perform Independent Component Analysis (ICA) on pixel spectra.

    Each pixel is treated as one observation and the multispectral bands are
    treated as features. FastICA estimates statistically independent source
    signals and returns one spatial image per estimated independent component.

    Parameters
    ----------
    stack : Stack
        Input multispectral stack with shape ``(bands, height, width)``.
    params : Params or dict, optional
        Validated ICA configuration. Parameters may alternatively be supplied
        directly as keyword arguments.
    **kwargs
        Individual operation parameters supplied directly as keyword
        arguments. Their types, defaults, constraints, and descriptions
        are defined by :class:`Params`.

    Returns
    -------
    Result
        Result whose output stack contains ``IC1``, ``IC2``, ... images.
        ``additional_data`` records the estimated unmixing components, mixing
        matrix, feature mean and number of iterations.

    Raises
    ------
    ValueError
        If ``n_components`` exceeds the number of bands.

    Notes
    -----
    ICA components have arbitrary sign and order; these properties should not
    be interpreted as physically meaningful without additional constraints.
    The implementation uses FastICA with ``whiten="unit-variance"``.

    Examples
    --------
    >>> result = run(stack, n_components=4, algorithm="parallel")
    """
    config = resolve_params(Params, params, kwargs)
    data = stack.as_array(dtype=np.float32)
    n_bands, height, width = data.shape
    if config.n_components > n_bands:
        raise ValueError(
            f"n_components={config.n_components} exceeds the number of bands ({n_bands})."
        )

    matrix = data.reshape(n_bands, -1).T
    model = FastICA(
        n_components=config.n_components,
        algorithm=config.algorithm,
        whiten="unit-variance",
        fun=config.fun,
        max_iter=config.max_iter,
        tol=config.tolerance,
        random_state=config.random_state,
    )
    sources = model.fit_transform(matrix)
    output_data = sources.T.reshape(config.n_components, height, width).astype(np.float32)
    output_bands = [
        BandMetadata(index=index, name=f"IC{index + 1}", extra={"component": index + 1})
        for index in range(config.n_components)
    ]
    output = Stack(
        data=output_data,
        bands=output_bands,
        metadata={**stack.metadata, "derived_from": "ica"},
    )
    return Result(
        output=output,
        operation_id=INFO.operation_id,
        operation_version=INFO.operation_version,
        operation_info=INFO,
        parameters=config.model_dump(mode="json"),
        metadata={"scientific_convention": "pixels_as_samples_bands_as_features"},
        additional_data={
            "components": model.components_.tolist(),
            "mixing": model.mixing_.tolist() if model.mixing_ is not None else None,
            "mean": model.mean_.tolist() if model.mean_ is not None else None,
            "n_iter": int(model.n_iter_),
        },
    )
