from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import tifffile
from pydantic import BaseModel, ConfigDict, Field

from ..errors import TiffFormatError
from .manifest import DatasetManifest, ManifestBand, ManifestSource, RGBRepresentation

_TIFF_SUFFIXES = {".tif", ".tiff"}
_PREFIX_WAVELENGTH = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:nm\b)?", re.IGNORECASE)
_EXPLICIT_NM = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*nm\b", re.IGNORECASE)
_BAND_INDEX = re.compile(r"(?:^|[^a-z0-9])band[_\-\s]?(\d+)(?:[^0-9]|$)", re.IGNORECASE)


class SourceProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str
    detected_type: Literal["single_band", "rgb"]
    name: str
    wavelength_nm: float | None = None
    explicit_band_index: int | None = None
    confidence: Literal["high", "medium"] = "high"


class RepresentationProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    role: Literal["visible", "false_color", "other"]
    red_file: str
    red_channel: Literal["R", "G", "B"] | None = None
    green_file: str
    green_channel: Literal["R", "G", "B"] | None = None
    blue_file: str
    blue_channel: Literal["R", "G", "B"] | None = None
    reason: str


class DatasetInspection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    folder: str
    sources: list[SourceProposal]
    suggested_representations: list[RepresentationProposal] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def inspect_tiff_folder(folder: str | Path) -> DatasetInspection:
    """Inspect TIFF files and propose a manifest without writing anything.

    Numeric prefixes such as ``365 UV.tif`` are interpreted as wavelengths in
    nanometres. RGB versus single-band classification comes from TIFF metadata,
    not the filename.
    """

    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.is_dir():
        raise FileNotFoundError(f"TIFF folder does not exist: {folder_path}")

    files = [
        path
        for path in folder_path.iterdir()
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in _TIFF_SUFFIXES
    ]
    if not files:
        raise FileNotFoundError(f"No TIFF files were found in {folder_path}.")

    proposals = [_inspect_source(path) for path in files]
    proposals = _deterministic_proposal_order(proposals)
    suggestions = _suggest_representations(proposals)

    return DatasetInspection(
        folder=str(folder_path),
        sources=proposals,
        suggested_representations=suggestions,
    )


def create_manifest(
    inspection: DatasetInspection,
    *,
    include_suggested_representations: bool = True,
) -> DatasetManifest:
    """Convert an inspection proposal into a portable, editable manifest."""

    used_ids: set[str] = set()
    sources: list[ManifestSource] = []
    band_lookup: dict[tuple[str, str | None], str] = {}

    for index, proposal in enumerate(inspection.sources):
        source_base = _slug(proposal.name)
        if proposal.detected_type == "single_band":
            if proposal.wavelength_nm is not None:
                base = f"wl_{_number_id(proposal.wavelength_nm)}"
            else:
                base = source_base
            band_id = _unique_id(base, used_ids)
            bands = [
                ManifestBand(
                    id=band_id,
                    name=proposal.name,
                    wavelength_nm=proposal.wavelength_nm,
                )
            ]
            band_lookup[(proposal.file, None)] = band_id
        else:
            bands = []
            for channel in ("R", "G", "B"):
                band_id = _unique_id(f"{source_base}_{channel.lower()}", used_ids)
                bands.append(
                    ManifestBand(
                        id=band_id,
                        name=f"{proposal.name} {channel}",
                        channel=channel,
                    )
                )
                band_lookup[(proposal.file, channel)] = band_id

        sources.append(
            ManifestSource(
                index=index,
                file=proposal.file,
                type=proposal.detected_type,
                name=proposal.name,
                bands=bands,
            )
        )

    representations: list[RGBRepresentation] = []
    if include_suggested_representations:
        for suggestion in inspection.suggested_representations:
            representations.append(
                RGBRepresentation(
                    id=suggestion.id,
                    label=suggestion.label,
                    role=suggestion.role,
                    red_band=band_lookup[(suggestion.red_file, suggestion.red_channel)],
                    green_band=band_lookup[(suggestion.green_file, suggestion.green_channel)],
                    blue_band=band_lookup[(suggestion.blue_file, suggestion.blue_channel)],
                    description=suggestion.reason,
                )
            )

    return DatasetManifest(sources=sources, representations=representations)


def _inspect_source(path: Path) -> SourceProposal:
    try:
        with tifffile.TiffFile(path) as tif:
            if len(tif.pages) != 1:
                raise TiffFormatError(
                    f"'{path.name}' contains {len(tif.pages)} pages; MultiTIFF input is not supported."
                )
            page = tif.pages[0]
            shape = tuple(int(v) for v in page.shape)
            photometric = getattr(page.photometric, "name", str(page.photometric)).upper()
            samples = int(getattr(page, "samplesperpixel", 1) or 1)
            if len(shape) == 2:
                kind = "single_band"
            elif len(shape) == 3 and photometric == "RGB" and samples == 3 and 3 in (shape[0], shape[-1]):
                kind = "rgb"
            else:
                raise TiffFormatError(
                    f"'{path.name}' has shape {shape} and photometric={photometric}; supported inputs are "
                    "single-page grayscale or 3-channel RGB TIFF."
                )
    except TiffFormatError:
        raise
    except Exception as exc:
        raise TiffFormatError(f"Cannot inspect TIFF '{path}': {exc}") from exc

    wavelength = None
    explicit_index = None
    if kind == "single_band":
        match = _EXPLICIT_NM.search(path.stem) or _PREFIX_WAVELENGTH.search(path.stem)
        if match:
            wavelength = float(match.group(1))
        index_match = _BAND_INDEX.search(path.stem)
        if index_match:
            explicit_index = int(index_match.group(1))

    name = _human_name(path.stem, wavelength)
    return SourceProposal(
        file=path.name,
        detected_type=kind,
        name=name,
        wavelength_nm=wavelength,
        explicit_band_index=explicit_index,
    )


def _deterministic_proposal_order(proposals: list[SourceProposal]) -> list[SourceProposal]:
    spectral = [p for p in proposals if p.wavelength_nm is not None]
    indexed = [p for p in proposals if p.wavelength_nm is None and p.explicit_band_index is not None]
    named = [p for p in proposals if p.wavelength_nm is None and p.explicit_band_index is None]

    wavelengths = [p.wavelength_nm for p in spectral]
    if len(wavelengths) != len(set(wavelengths)):
        raise ValueError("Duplicate wavelengths were detected while inspecting the TIFF folder.")
    indices = [p.explicit_band_index for p in indexed]
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate explicit band indices were detected while inspecting the TIFF folder.")

    # Named sources (for example VIS.tif or UVL.tif) follow wavelength/indexed
    # sources in a stable, case-insensitive order. Their actual band semantics are
    # made explicit in the generated manifest, so filesystem ordering is never used.
    return (
        sorted(spectral, key=lambda p: float(p.wavelength_nm))
        + sorted(indexed, key=lambda p: int(p.explicit_band_index))
        + sorted(named, key=lambda p: (p.file.casefold(), p.file))
    )


def _suggest_representations(proposals: list[SourceProposal]) -> list[RepresentationProposal]:
    suggestions: list[RepresentationProposal] = []

    # A source named VIS/visible is a strong candidate for the default visible image.
    visible_rgb = next(
        (
            p
            for p in proposals
            if p.detected_type == "rgb" and p.name.strip().casefold() in {"vis", "visible"}
        ),
        None,
    )
    if visible_rgb:
        suggestions.append(
            RepresentationProposal(
                id="visible_rgb",
                label="Visible image",
                role="visible",
                red_file=visible_rgb.file,
                red_channel="R",
                green_file=visible_rgb.file,
                green_channel="G",
                blue_file=visible_rgb.file,
                blue_channel="B",
                reason="Suggested from an RGB TIFF named VIS/Visible.",
            )
        )

    spectral = [p for p in proposals if p.wavelength_nm is not None]
    if len(spectral) >= 3:
        red = _nearest(spectral, 630.0)
        green = _nearest(spectral, 535.0)
        blue = _nearest(spectral, 475.0)
        if len({red.file, green.file, blue.file}) == 3:
            suggestions.append(
                RepresentationProposal(
                    id="visible_spectral",
                    label="Visible composite from spectral bands",
                    role="visible",
                    red_file=red.file,
                    green_file=green.file,
                    blue_file=blue.file,
                    reason=(
                        "Suggested by selecting available bands nearest to 630 nm (R), "
                        "535 nm (G), and 475 nm (B). Review before scientific use."
                    ),
                )
            )

    return suggestions


def _nearest(proposals: list[SourceProposal], target: float) -> SourceProposal:
    return min(proposals, key=lambda p: abs(float(p.wavelength_nm) - target))


def _human_name(stem: str, wavelength: float | None) -> str:
    value = stem.strip()
    if wavelength is not None:
        value = _PREFIX_WAVELENGTH.sub("", value, count=1).strip(" _-")
        if not value:
            value = f"{wavelength:g} nm"
    return value


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "band"


def _number_id(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _unique_id(base: str, used: set[str]) -> str:
    candidate = base
    counter = 2
    while candidate in used:
        candidate = f"{base}_{counter}"
        counter += 1
    used.add(candidate)
    return candidate
