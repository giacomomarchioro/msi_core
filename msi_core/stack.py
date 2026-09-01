from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class BandMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0)
    source_index: int | None = Field(default=None, ge=0)
    name: str | None = None
    wavelength_nm: float | None = Field(default=None, gt=0)
    source_file: str | None = None
    channel: int | str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Band:
    data: np.ndarray
    metadata: BandMetadata

    @property
    def name(self) -> str | None:
        return self.metadata.name

    @property
    def wavelength_nm(self) -> float | None:
        return self.metadata.wavelength_nm

    @property
    def source_file(self) -> str | None:
        return self.metadata.source_file


@dataclass(slots=True)
class Stack:
    """An in-memory multispectral stack with shape ``(bands, y, x)``."""

    data: np.ndarray
    bands: list[BandMetadata]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            raise TypeError("Stack data must be a NumPy array.")
        if self.data.ndim != 3:
            raise ValueError(
                f"Stack data must have shape (bands, y, x); received {self.data.shape}."
            )
        if any(size <= 0 for size in self.data.shape):
            raise ValueError("Stack dimensions must be positive.")
        if len(self.bands) != self.data.shape[0]:
            raise ValueError(
                f"Expected {self.data.shape[0]} band metadata records, received {len(self.bands)}."
            )
        expected = list(range(len(self.bands)))
        actual = [band.index for band in self.bands]
        if actual != expected:
            raise ValueError(
                f"BandMetadata.index values must be contiguous and ordered: expected {expected}, "
                f"received {actual}."
            )

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Band:
        if not isinstance(index, int):
            raise TypeError("Use Stack.select(...) to select multiple bands.")
        return Band(data=self.data[index], metadata=self.bands[index])

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.data.shape)

    @property
    def n_bands(self) -> int:
        return self.data.shape[0]

    @property
    def height(self) -> int:
        return self.data.shape[1]

    @property
    def width(self) -> int:
        return self.data.shape[2]

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data.dtype

    def as_array(
        self,
        *,
        dtype: np.dtype[Any] | type[Any] | str | None = None,
        copy: bool = False,
    ) -> np.ndarray:
        array = self.data if dtype is None else self.data.astype(dtype, copy=False)
        return array.copy() if copy else array

    def select(self, indices: Iterable[int]) -> "Stack":
        selected = list(indices)
        if not selected:
            raise ValueError("At least one band must be selected.")
        if len(set(selected)) != len(selected):
            raise ValueError("Band selection contains duplicate indices.")
        for index in selected:
            if index < 0 or index >= self.n_bands:
                raise IndexError(f"Band index {index} is outside 0..{self.n_bands - 1}.")

        bands = [
            self.bands[source_index].model_copy(
                update={
                    "index": new_index,
                    "source_index": (
                        self.bands[source_index].source_index
                        if self.bands[source_index].source_index is not None
                        else source_index
                    ),
                }
            )
            for new_index, source_index in enumerate(selected)
        ]
        return Stack(
            data=np.asarray(self.data[selected]),
            bands=bands,
            metadata={**self.metadata, "selected_bands": selected},
        )

    def with_data(
        self,
        data: np.ndarray,
        *,
        bands: list[BandMetadata] | None = None,
        metadata_update: dict[str, Any] | None = None,
    ) -> "Stack":
        output_bands = bands if bands is not None else [band.model_copy() for band in self.bands]
        metadata = dict(self.metadata)
        if metadata_update:
            metadata.update(metadata_update)
        return Stack(data=np.asarray(data), bands=output_bands, metadata=metadata)

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "bands": [band.model_dump(mode="json") for band in self.bands],
            "metadata": _json_safe(self.metadata),
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
