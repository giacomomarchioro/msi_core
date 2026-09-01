from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .attribution import OperationInfo
from .stack import Band, Stack, _json_safe


@dataclass(slots=True)
class Result:
    """Uniform image result returned by every operation."""

    output: Stack
    operation_id: str
    operation_version: str
    parameters: dict[str, Any]
    operation_info: OperationInfo | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    additional_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.operation_info is not None:
            if self.operation_info.operation_id != self.operation_id:
                raise ValueError("operation_info.operation_id does not match operation_id.")
            if self.operation_info.operation_version != self.operation_version:
                raise ValueError(
                    "operation_info.operation_version does not match operation_version."
                )

    @property
    def images(self) -> list[Band]:
        return [self.output[index] for index in range(self.output.n_bands)]

    @property
    def stack(self) -> Stack:
        return self.output

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "operation_version": self.operation_version,
            "parameters": _json_safe(self.parameters),
            "operation_info": (
                self.operation_info.model_dump(mode="json")
                if self.operation_info is not None
                else None
            ),
            "metadata": _json_safe(self.metadata),
            "additional_data": _json_safe(self.additional_data),
            "output": self.output.to_metadata_dict(),
        }
