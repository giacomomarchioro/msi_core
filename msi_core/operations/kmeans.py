from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from sklearn.cluster import MiniBatchKMeans

from ..attribution import OperationInfo
from ..parameters import resolve_params
from ..result import Result
from ..stack import BandMetadata, Stack

INFO = OperationInfo(
    operation_id="kmeans",
    operation_version="1.0.0",
    display_name="Spectral K-means clustering",
    description="Cluster pixels using their multispectral band values.",
    category="segmentation",
    input_kind="stack",
)


class Params(BaseModel):
    n_clusters: int = Field(default=8, ge=2, description="Number of spectral clusters to estimate.")
    batch_size: int = Field(
        default=4096,
        ge=16,
        description="Mini-batch size used during MiniBatchKMeans optimization.",
    )
    max_iter: int = Field(
        default=100,
        ge=1,
        description="Maximum number of MiniBatchKMeans iterations.",
    )
    n_init: int = Field(
        default=3,
        ge=1,
        description="Number of independent centroid initializations.",
    )
    random_state: int | None = Field(
        default=0,
        description=(
            'Random seed for centroid initialization and mini-batch sampling; use None '
            'for non-deterministic behavior.'
        ),
    )


def run(
    stack: Stack,
    params: Params | dict[str, Any] | None = None,
    **kwargs: Any,
) -> Result:
    """Cluster pixels from their multispectral vectors using MiniBatch K-means.

    Parameters
    ----------
    stack : Stack
        Input multispectral stack. Pixels are observations and bands are
        clustering features.
    params : Params or dict, optional
        Validated clustering configuration. Parameters may alternatively be
        supplied directly as keyword arguments.
    **kwargs
        Individual operation parameters supplied directly as keyword
        arguments. Their types, defaults, constraints, and descriptions
        are defined by :class:`Params`.

    Returns
    -------
    Result
        A one-band integer stack named ``cluster_labels``. Cluster centers,
        inertia and iteration count are stored in ``additional_data``.

    Raises
    ------
    ValueError
        If the requested number of clusters exceeds the number of pixels.
    """
    config = resolve_params(Params, params, kwargs)
    data = stack.as_array(dtype=np.float32)
    n_bands, height, width = data.shape
    matrix = data.reshape(n_bands, -1).T
    if config.n_clusters > matrix.shape[0]:
        raise ValueError("n_clusters cannot exceed the number of pixels.")

    model = MiniBatchKMeans(
        n_clusters=config.n_clusters,
        batch_size=config.batch_size,
        max_iter=config.max_iter,
        n_init=config.n_init,
        random_state=config.random_state,
    )
    labels = model.fit_predict(matrix).reshape(1, height, width).astype(np.int32)
    output = Stack(
        data=labels,
        bands=[BandMetadata(index=0, name="cluster_labels")],
        metadata={**stack.metadata, "derived_from": "kmeans"},
    )
    return Result(
        output=output,
        operation_id=INFO.operation_id,
        operation_version=INFO.operation_version,
        operation_info=INFO,
        parameters=config.model_dump(mode="json"),
        metadata={"scientific_convention": "pixels_as_samples_bands_as_features"},
        additional_data={
            "cluster_centers": model.cluster_centers_.tolist(),
            "inertia": float(model.inertia_),
            "n_iter": int(model.n_iter_),
        },
    )
