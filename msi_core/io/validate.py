from __future__ import annotations

from pathlib import Path

from ..errors import BandOrderError, TiffFormatError
from .inspect import inspect_tiff_folder
from .manifest import DatasetManifest


def validate_manifest(manifest: DatasetManifest, folder: str | Path) -> None:
    """Validate that a manifest matches the TIFF files and detected source types."""

    folder_path = Path(folder).expanduser().resolve()
    inspection = inspect_tiff_folder(folder_path)
    detected = {proposal.file: proposal for proposal in inspection.sources}
    listed = {source.file: source for source in manifest.sources}

    missing = sorted(set(listed) - set(detected))
    unlisted = sorted(set(detected) - set(listed))
    if missing or unlisted:
        raise BandOrderError(
            "The manifest must list every TIFF exactly once. "
            f"Missing files: {missing or 'none'}; unlisted TIFF files: {unlisted or 'none'}."
        )

    for filename, source in listed.items():
        proposal = detected[filename]
        if source.type != proposal.detected_type:
            raise TiffFormatError(
                f"Manifest declares '{filename}' as {source.type}, but TIFF metadata indicates "
                f"{proposal.detected_type}."
            )
