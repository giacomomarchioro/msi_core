from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Person(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    affiliation: str | None = None
    orcid: str | None = None


class AlgorithmAttribution(BaseModel):
    """Scientific-method attribution, distinct from source-code authorship."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method_name: str = Field(min_length=1)
    authors: tuple[Person, ...] = ()
    status: Literal["established", "published", "validated", "experimental", "unpublished"]
    citation: str | None = None
    doi: str | None = None
    url: str | None = None
    license: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def require_authors_for_attributed_methods(self) -> "AlgorithmAttribution":
        if not self.authors:
            raise ValueError("At least one scientific method author is required.")
        return self


class OperationInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    operation_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    operation_version: str = Field(pattern=r"^\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?$")
    display_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    category: str = Field(min_length=1)
    input_kind: Literal["stack", "band_pair"]
    output_kind: Literal["stack"] = "stack"
    attribution: AlgorithmAttribution | None = None
