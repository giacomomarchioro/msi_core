from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..errors import BandOrderError


class ManifestBand(BaseModel):
    """One logical band produced by a TIFF source."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str | None = None
    wavelength_nm: float | None = Field(default=None, gt=0)
    channel: Literal["R", "G", "B"] | None = None


class ManifestSource(BaseModel):
    """One physical TIFF file and the logical bands it contributes."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    file: str = Field(min_length=1)
    type: Literal["single_band", "rgb"]
    name: str | None = None
    bands: list[ManifestBand] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_bands(self) -> "ManifestSource":
        if self.type == "single_band":
            if len(self.bands) != 1 or self.bands[0].channel is not None:
                raise ValueError("A single_band source must define exactly one non-channel band.")
        else:
            channels = [band.channel for band in self.bands]
            if channels != ["R", "G", "B"]:
                raise ValueError("An rgb source must define exactly R, G, B bands in that order.")
        return self


class RGBRepresentation(BaseModel):
    """A named RGB visualization assembled from three logical bands."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    kind: Literal["rgb"] = "rgb"
    label: str
    role: Literal["visible", "false_color", "other"] = "other"
    red_band: str
    green_band: str
    blue_band: str
    description: str | None = None


class DatasetManifest(BaseModel):
    """Portable description of TIFF sources, logical bands, and display representations."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "2.0"
    sources: list[ManifestSource] = Field(min_length=1)
    representations: list[RGBRepresentation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_manifest(self) -> "DatasetManifest":
        indices = [source.index for source in self.sources]
        if sorted(indices) != list(range(len(indices))):
            raise ValueError("Source indices must be contiguous from zero.")

        files = [source.file for source in self.sources]
        if len(files) != len(set(files)):
            raise ValueError("Manifest contains duplicate TIFF filenames.")

        band_ids = [band.id for source in self.sources for band in source.bands]
        if len(band_ids) != len(set(band_ids)):
            raise ValueError("Logical band ids must be unique.")
        known = set(band_ids)

        representation_ids = [rep.id for rep in self.representations]
        if len(representation_ids) != len(set(representation_ids)):
            raise ValueError("Representation ids must be unique.")
        for rep in self.representations:
            referenced = {rep.red_band, rep.green_band, rep.blue_band}
            missing = sorted(referenced - known)
            if missing:
                raise ValueError(
                    f"Representation '{rep.id}' references unknown band ids: {missing}."
                )
        return self

    @property
    def bands(self) -> list[ManifestBand]:
        return [band for source in self.sources for band in source.bands]


# Compatibility aliases for code that imported the 0.3.1 name.
BandsManifest = DatasetManifest


def read_manifest(path: Path) -> DatasetManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "bands" in payload and "sources" not in payload:
            payload = _upgrade_v1_manifest(payload)
        return DatasetManifest.model_validate(payload)
    except Exception as exc:
        raise BandOrderError(f"Invalid dataset manifest '{path}': {exc}") from exc


def write_manifest(manifest: DatasetManifest, path: str | Path) -> Path:
    output = Path(path)
    output.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def _upgrade_v1_manifest(payload: dict) -> dict:
    """Upgrade the simple 0.3.1 ``bands`` manifest in memory."""

    sources = []
    used_ids: set[str] = set()
    for item in sorted(payload.get("bands", []), key=lambda entry: entry["index"]):
        source_name = item.get("name") or Path(item["file"]).stem
        band_id = _unique_id(_slug(source_name), used_ids)
        sources.append(
            {
                "index": item["index"],
                "file": item["file"],
                "type": "single_band",
                "name": source_name,
                "bands": [
                    {
                        "id": band_id,
                        "name": item.get("name"),
                        "wavelength_nm": item.get("wavelength_nm"),
                        "channel": None,
                    }
                ],
            }
        )
    return {"schema_version": "2.0", "sources": sources, "representations": []}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "band"


def _unique_id(base: str, used: set[str]) -> str:
    candidate = base
    counter = 2
    while candidate in used:
        candidate = f"{base}_{counter}"
        counter += 1
    used.add(candidate)
    return candidate
