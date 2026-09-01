from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field
from sklearn.decomposition import PCA

from ..attribution import OperationInfo
from ..parameters import resolve_params
from ..result import Result
from ..stack import BandMetadata, Stack

INFO = OperationInfo(
    operation_id="pca",
    operation_version="1.0.0",
    display_name="Principal component analysis",
    description="Compute spectral score images or image-space component images.",
    category="decomposition",
    input_kind="stack",
)


class Params(BaseModel):
    n_components: int = Field(
        default=8,
        ge=1,
        description="Number of principal components to compute.",
    )
    mode: Literal["spectral", "image_space"] = Field(
        default="spectral",
        description=(
            'PCA orientation: spectral treats pixels as samples and bands as features; '
            'image_space treats bands as samples and pixel positions as features.'
        ),
    )
    standardize: bool = Field(
        default=False,
        description="Standardize features to unit standard deviation before PCA.",
    )
    whiten: bool = Field(
        default=False,
        description="Apply scikit-learn PCA whitening to the returned component scores.",
    )
    svd_solver: Literal["auto", "full", "randomized"] = Field(
        default="auto",
        description="Singular-value decomposition solver passed to scikit-learn PCA.",
    )
    random_state: int | None = Field(
        default=0,
        description=(
            'Random seed used by stochastic PCA solvers; use None for non-deterministic '
            'initialization.'
        ),
    )


def run(
    stack: Stack,
    params: Params | dict[str, Any] | None = None,
    **kwargs: Any,
) -> Result:
    """Perform Principal Component Analysis (PCA) on a multispectral stack.

    PCA transforms the input data into orthogonal components ordered by
    decreasing explained variance. Two orientations are supported because
    multispectral imaging uses both pixel-spectral and image-space
    formulations in practice.

    Parameters
    ----------
    stack : Stack
        Input multispectral stack with shape ``(bands, height, width)``.
    params : Params or dict, optional
        Validated PCA configuration. This form is primarily intended for
        programmatic workflows. Parameters may alternatively be supplied
        directly as keyword arguments.
    **kwargs
        Individual operation parameters supplied directly as keyword
        arguments. Their types, defaults, constraints, and descriptions
        are defined by :class:`Params`.

    Returns
    -------
    Result
        Result whose output stack contains ``PC1``, ``PC2``, ... images.
        ``additional_data`` includes explained variance information and either
        spectral loadings or image-space band scores, depending on ``mode``.

    Raises
    ------
    ValueError
        If ``n_components`` is larger than allowed by the input dimensions.

    Notes
    -----
    ``spectral`` and ``image_space`` are distinct PCA formulations and their
    component images have different interpretations. Standardization is not a
    display operation and should be selected deliberately as part of the
    scientific method.

    Examples
    --------
    >>> result = run(stack, n_components=6, mode="spectral")
    >>> result.output.n_bands
    6
    """
    config = resolve_params(Params, params, kwargs)
    data = stack.as_array(dtype=np.float32)
    n_bands, height, width = data.shape
    n_pixels = height * width
    maximum = min(n_bands, n_pixels)
    if config.n_components > maximum:
        raise ValueError(
            f"n_components={config.n_components} exceeds the maximum {maximum} for stack {data.shape}."
        )

    if config.mode == "spectral":
        matrix = data.reshape(n_bands, -1).T
        matrix, standardization = _standardize(matrix, config.standardize)
        model = PCA(
            n_components=config.n_components,
            whiten=config.whiten,
            svd_solver=config.svd_solver,
            random_state=config.random_state,
        )
        scores = model.fit_transform(matrix)
        output_data = scores.T.reshape(config.n_components, height, width).astype(np.float32)
        additional = {
            "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
            "explained_variance": model.explained_variance_.tolist(),
            "singular_values": model.singular_values_.tolist(),
            "spectral_loadings": model.components_.tolist(),
            "feature_mean": model.mean_.tolist(),
            "standardization": standardization,
        }
    else:
        matrix = data.reshape(n_bands, -1)
        matrix, standardization = _standardize(matrix, config.standardize)
        model = PCA(
            n_components=config.n_components,
            whiten=config.whiten,
            svd_solver=config.svd_solver,
            random_state=config.random_state,
        )
        band_scores = model.fit_transform(matrix)
        output_data = model.components_.reshape(config.n_components, height, width).astype(np.float32)
        additional = {
            "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
            "explained_variance": model.explained_variance_.tolist(),
            "singular_values": model.singular_values_.tolist(),
            "band_scores": band_scores.tolist(),
            "standardization": standardization,
            "note": "Large pixel-space component vectors are stored only as output images, not duplicated in metadata.",
        }

    output_bands = [
        BandMetadata(
            index=index,
            name=f"PC{index + 1}",
            extra={"component": index + 1, "pca_mode": config.mode},
        )
        for index in range(config.n_components)
    ]
    output = Stack(
        data=output_data,
        bands=output_bands,
        metadata={**stack.metadata, "derived_from": "pca", "pca_mode": config.mode},
    )
    return Result(
        output=output,
        operation_id=INFO.operation_id,
        operation_version=INFO.operation_version,
        operation_info=INFO,
        parameters=config.model_dump(mode="json"),
        metadata={
            "scientific_convention": (
                "pixels_as_samples_bands_as_features"
                if config.mode == "spectral"
                else "bands_as_samples_pixels_as_features"
            )
        },
        additional_data=additional,
    )


def _standardize(matrix: np.ndarray, enabled: bool) -> tuple[np.ndarray, dict[str, Any]]:
    if not enabled:
        return matrix, {"enabled": False}
    mean = matrix.mean(axis=0, keepdims=True)
    scale = matrix.std(axis=0, keepdims=True)
    safe_scale = np.where(scale > 0, scale, 1.0)
    standardized = (matrix - mean) / safe_scale
    # In image-space mode these vectors can contain one value per pixel. Avoid duplicating them in metadata.
    if matrix.shape[1] > 1024:
        metadata = {"enabled": True, "vectors_stored": False}
    else:
        metadata = {
            "enabled": True,
            "vectors_stored": True,
            "mean": mean.ravel().tolist(),
            "scale": safe_scale.ravel().tolist(),
        }
    return standardized.astype(np.float32, copy=False), metadata
