from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import tifffile

from ..errors import BandOrderError, TiffFormatError
from ..region import Region
from ..stack import BandMetadata, Stack
from .manifest import DatasetManifest, ManifestBand, read_manifest
from .validate import validate_manifest

_TIFF_SUFFIXES = {".tif", ".tiff"}
_WAVELENGTH_PATTERN = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*nm(?![a-z])", re.IGNORECASE)
_PREFIX_WAVELENGTH_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s+", re.IGNORECASE)
_BAND_INDEX_PATTERN = re.compile(r"(?:^|[^a-z0-9])band[_\-\s]?(\d+)(?:[^0-9]|$)", re.IGNORECASE)
_RGB_CHANNELS = ("R", "G", "B")


@dataclass(frozen=True, slots=True)
class _SourceEntry:
    index: int
    path: Path
    source_index: int | None
    manifest_bands: tuple[ManifestBand, ...] | None = None
    inferred_name: str | None = None
    inferred_wavelength_nm: float | None = None


@dataclass(frozen=True, slots=True)
class _TiffInfo:
    spatial_shape: tuple[int, int]
    dtype: np.dtype
    kind: str
    sample_axis: int | None = None


@dataclass(frozen=True, slots=True)
class _BandEntry:
    source: _SourceEntry
    channel_index: int | None
    channel_name: str | None
    manifest_band: ManifestBand | None = None


def load_tiff_folder(
    folder: str | Path,
    *,
    region: Region | None = None,
    bands: Iterable[int] | None = None,
    manifest: DatasetManifest | str | Path | None = None,
    dtype: np.dtype | type | str | None = None,
) -> Stack:
    """Load TIFF sources into an in-memory multispectral stack.

    A grayscale TIFF contributes one band. A 3-channel RGB TIFF contributes
    three consecutive bands in canonical R, G, B order.

    ``manifest`` may be a :class:`DatasetManifest` (useful for a SaaS/database)
    or a JSON path. If omitted, ``bands.json`` is used when present. Without a
    manifest, every source must still have a deterministic wavelength or band
    index in its filename.
    """

    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.is_dir():
        raise FileNotFoundError(f"TIFF folder does not exist: {folder_path}")

    tiff_files = _find_tiff_files(folder_path)
    if not tiff_files:
        raise FileNotFoundError(f"No TIFF files were found in {folder_path}.")

    sources, ordering_source, manifest_label = _resolve_sources(
        folder_path=folder_path,
        tiff_files=tiff_files,
        manifest=manifest,
    )

    source_infos = [_inspect_tiff(source.path) for source in sources]
    _validate_source_types_against_manifest(sources, source_infos)

    first_info = source_infos[0]
    for source, info in zip(sources[1:], source_infos[1:]):
        if info.spatial_shape != first_info.spatial_shape:
            raise TiffFormatError(
                f"Spatial shape mismatch: '{sources[0].path.name}' has {first_info.spatial_shape}, "
                f"but '{source.path.name}' has {info.spatial_shape}."
            )
        if dtype is None and info.dtype != first_info.dtype:
            raise TiffFormatError(
                f"Data type mismatch: '{sources[0].path.name}' has {first_info.dtype}, "
                f"but '{source.path.name}' has {info.dtype}. Pass dtype=... to explicitly "
                "convert all input bands to a common type."
            )

    full_height, full_width = first_info.spatial_shape
    selected_region = region or Region(x0=0, x1=full_width, y0=0, y1=full_height)
    selected_region.validate_within(width=full_width, height=full_height)

    expanded = _expand_sources(sources, source_infos)
    selected_indices = list(range(len(expanded))) if bands is None else list(bands)
    if not selected_indices:
        raise ValueError("At least one band must be selected.")
    if len(set(selected_indices)) != len(selected_indices):
        raise ValueError("Band selection contains duplicate indices.")
    for index in selected_indices:
        if index < 0 or index >= len(expanded):
            raise IndexError(f"Band index {index} is outside 0..{len(expanded) - 1}.")

    arrays: list[np.ndarray] = []
    metadata: list[BandMetadata] = []
    read_modes: dict[str, str] = {}

    for output_index, expanded_index in enumerate(selected_indices):
        entry = expanded[expanded_index]
        info = source_infos[entry.source.index]
        image, read_mode = _read_region(
            entry.source.path,
            selected_region,
            info=info,
            channel_index=entry.channel_index,
        )
        if dtype is not None:
            image = _safe_cast(image, dtype, source_name=entry.source.path.name)
        arrays.append(image)

        read_key = entry.source.path.name
        if entry.channel_name is not None:
            read_key = f"{read_key}:{entry.channel_name}"
        read_modes[read_key] = read_mode

        if entry.manifest_band is not None:
            band = entry.manifest_band
            name = band.name
            wavelength = band.wavelength_nm
            channel = band.channel
            band_id = band.id
        else:
            name = entry.source.inferred_name
            wavelength = entry.source.inferred_wavelength_nm
            channel = entry.channel_name
            band_id = None
            if channel is not None:
                name = f"{name} {channel}" if name else f"{entry.source.path.stem} {channel}"

        metadata.append(
            BandMetadata(
                index=output_index,
                source_index=entry.source.source_index,
                name=name,
                wavelength_nm=wavelength,
                source_file=entry.source.path.name,
                channel=channel,
                extra={"band_id": band_id} if band_id else {},
            )
        )

    return Stack(
        data=np.stack(arrays, axis=0),
        bands=metadata,
        metadata={
            "source_folder": str(folder_path),
            "full_spatial_shape": [full_height, full_width],
            "region": selected_region.as_dict(),
            "band_order_source": ordering_source,
            "manifest": manifest_label,
            "read_modes": read_modes,
            "rgb_split_order": list(_RGB_CHANNELS),
        },
    )


def _safe_cast(
    image: np.ndarray,
    dtype: np.dtype | type | str,
    *,
    source_name: str,
) -> np.ndarray:
    target = np.dtype(dtype)
    values = np.asarray(image)
    if np.issubdtype(target, np.floating):
        finite = np.isfinite(values)
        if finite.any():
            info = np.finfo(target)
            finite_max = float(info.max)
            minimum = float(values[finite].min())
            maximum = float(values[finite].max())
            if minimum < -finite_max or maximum > finite_max:
                raise ValueError(
                    f"TIFF values in '{source_name}' cannot be safely represented as {target}. "
                    f"Observed range is [{minimum:g}, {maximum:g}], while {target} finite range is "
                    f"approximately [{-finite_max:g}, {finite_max:g}]. Use dtype='float32' or wider."
                )
    converted = values.astype(target, copy=False)
    if np.issubdtype(target, np.floating) and not np.isfinite(converted).all():
        raise ValueError(
            f"Conversion of '{source_name}' to {target} produced non-finite values. "
            "Use a wider processing dtype such as float32."
        )
    return converted


def _find_tiff_files(folder: Path) -> list[Path]:
    return [
        path
        for path in folder.iterdir()
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in _TIFF_SUFFIXES
    ]


def _resolve_sources(
    *,
    folder_path: Path,
    tiff_files: list[Path],
    manifest: DatasetManifest | str | Path | None,
) -> tuple[list[_SourceEntry], str, str | None]:
    parsed: DatasetManifest | None = None
    manifest_label: str | None = None

    if isinstance(manifest, DatasetManifest):
        parsed = manifest
        manifest_label = "in_memory"
    else:
        if manifest is None:
            manifest_path = folder_path / "bands.json"
        else:
            supplied = Path(manifest).expanduser()
            manifest_path = supplied.resolve() if supplied.is_absolute() else (folder_path / supplied).resolve()
        if manifest_path.exists():
            parsed = read_manifest(manifest_path)
            manifest_label = str(manifest_path)

    if parsed is not None:
        validate_manifest(parsed, folder_path)
        sources: list[_SourceEntry] = []
        for item in sorted(parsed.sources, key=lambda source: source.index):
            path = (folder_path / item.file).resolve()
            try:
                path.relative_to(folder_path)
            except ValueError as exc:
                raise BandOrderError(f"Manifest file '{item.file}' resolves outside the TIFF folder.") from exc
            sources.append(
                _SourceEntry(
                    index=item.index,
                    path=path,
                    source_index=item.index,
                    manifest_bands=tuple(item.bands),
                    inferred_name=item.name,
                )
            )
        return sources, "manifest", manifest_label

    # Minimal standalone fallback: all files must follow one deterministic naming rule.
    wavelengths: list[float | None] = []
    for path in tiff_files:
        match = _WAVELENGTH_PATTERN.search(path.stem) or _PREFIX_WAVELENGTH_PATTERN.search(path.stem)
        wavelengths.append(float(match.group(1)) if match else None)

    if all(value is not None for value in wavelengths):
        numeric = [float(value) for value in wavelengths if value is not None]
        if len(numeric) != len(set(numeric)):
            raise BandOrderError("Duplicate wavelengths were found in TIFF filenames.")
        ordered = sorted(zip(tiff_files, numeric), key=lambda pair: pair[1])
        return (
            [
                _SourceEntry(
                    index=index,
                    path=path,
                    source_index=index,
                    inferred_name=f"{wavelength:g} nm",
                    inferred_wavelength_nm=wavelength,
                )
                for index, (path, wavelength) in enumerate(ordered)
            ],
            "filename_wavelength",
            None,
        )

    if any(value is not None for value in wavelengths):
        raise BandOrderError(
            "Only some TIFF filenames expose wavelengths. Generate bands.json with "
            "inspect_tiff_folder()/create_manifest() or provide a complete manifest."
        )

    index_values: list[int | None] = []
    for path in tiff_files:
        match = _BAND_INDEX_PATTERN.search(path.stem)
        index_values.append(int(match.group(1)) if match else None)
    if all(value is not None for value in index_values):
        numeric_indices = [int(value) for value in index_values if value is not None]
        if len(numeric_indices) != len(set(numeric_indices)):
            raise BandOrderError("Duplicate band indices were found in TIFF filenames.")
        ordered = sorted(zip(tiff_files, numeric_indices), key=lambda pair: pair[1])
        return (
            [
                _SourceEntry(
                    index=index,
                    path=path,
                    source_index=source_index,
                    inferred_name=f"Band {source_index}",
                )
                for index, (path, source_index) in enumerate(ordered)
            ],
            "filename_band_index",
            None,
        )

    raise BandOrderError(
        "Band order is ambiguous. Generate a DatasetManifest, provide bands.json, include a "
        "wavelength in every filename, or use explicit names such as band_001.tif."
    )


def _inspect_tiff(path: Path) -> _TiffInfo:
    try:
        with tifffile.TiffFile(path) as tif:
            if len(tif.pages) != 1:
                raise TiffFormatError(
                    f"'{path.name}' contains {len(tif.pages)} pages; MultiTIFF input is not supported."
                )
            page = tif.pages[0]
            shape = tuple(int(value) for value in page.shape)
            dtype = np.dtype(page.dtype)
            if len(shape) == 2:
                return _TiffInfo(spatial_shape=(shape[0], shape[1]), dtype=dtype, kind="gray")

            photometric = getattr(page.photometric, "name", str(page.photometric)).upper()
            samples_per_pixel = int(getattr(page, "samplesperpixel", 1) or 1)
            if len(shape) == 3 and photometric == "RGB" and samples_per_pixel == 3:
                if shape[-1] == 3:
                    return _TiffInfo((shape[0], shape[1]), dtype, "rgb", sample_axis=2)
                if shape[0] == 3:
                    return _TiffInfo((shape[1], shape[2]), dtype, "rgb", sample_axis=0)

            raise TiffFormatError(
                f"'{path.name}' has shape {shape} and photometric={photometric}; supported inputs are "
                "single-page 2-D grayscale TIFF or single-page 3-channel RGB TIFF."
            )
    except TiffFormatError:
        raise
    except Exception as exc:
        raise TiffFormatError(f"Cannot inspect TIFF '{path}': {exc}") from exc


def _validate_source_types_against_manifest(
    sources: list[_SourceEntry], infos: list[_TiffInfo]
) -> None:
    for source, info in zip(sources, infos):
        if source.manifest_bands is None:
            continue
        expected = "rgb" if len(source.manifest_bands) == 3 else "gray"
        if expected != info.kind:
            raise TiffFormatError(
                f"Manifest definition for '{source.path.name}' does not match the TIFF layout."
            )


def _expand_sources(sources: list[_SourceEntry], infos: list[_TiffInfo]) -> list[_BandEntry]:
    expanded: list[_BandEntry] = []
    for source, info in zip(sources, infos):
        if info.kind == "gray":
            manifest_band = source.manifest_bands[0] if source.manifest_bands else None
            expanded.append(_BandEntry(source, None, None, manifest_band))
        elif info.kind == "rgb":
            for channel_index, channel_name in enumerate(_RGB_CHANNELS):
                manifest_band = source.manifest_bands[channel_index] if source.manifest_bands else None
                expanded.append(_BandEntry(source, channel_index, channel_name, manifest_band))
        else:  # pragma: no cover
            raise TiffFormatError(f"Unsupported TIFF kind: {info.kind}")
    return expanded


def _read_region(
    path: Path,
    region: Region,
    *,
    info: _TiffInfo,
    channel_index: int | None,
) -> tuple[np.ndarray, str]:
    try:
        mapped = tifffile.memmap(path, mode="r")
        return _extract_region(mapped, region, info=info, channel_index=channel_index), "memmap_roi"
    except TiffFormatError:
        raise
    except Exception:
        image = tifffile.imread(path)
        return _extract_region(image, region, info=info, channel_index=channel_index), "decoded_then_cropped"


def _extract_region(
    image: np.ndarray,
    region: Region,
    *,
    info: _TiffInfo,
    channel_index: int | None,
) -> np.ndarray:
    if info.kind == "gray":
        if image.ndim != 2 or channel_index is not None:
            raise TiffFormatError("Internal TIFF layout mismatch while reading a grayscale image.")
        return np.array(image[region.slices], copy=True)

    if info.kind == "rgb":
        if channel_index is None or channel_index not in (0, 1, 2):
            raise TiffFormatError("An RGB TIFF must be read using one of the R, G, B channels.")
        if info.sample_axis == 2 and image.ndim == 3 and image.shape[-1] == 3:
            return np.array(image[region.slices[0], region.slices[1], channel_index], copy=True)
        if info.sample_axis == 0 and image.ndim == 3 and image.shape[0] == 3:
            return np.array(image[channel_index, region.slices[0], region.slices[1]], copy=True)
        raise TiffFormatError("Internal TIFF RGB layout does not match the inspected file metadata.")

    raise TiffFormatError(f"Unsupported TIFF kind: {info.kind}")
