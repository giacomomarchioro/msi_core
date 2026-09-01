"""Command-line interface for dataset manifest inspection and creation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import BandOrderError, TiffFormatError
from .io import (
    create_manifest,
    inspect_tiff_folder,
    read_manifest,
    validate_manifest,
    write_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="msi-core",
        description="Utilities for multispectral TIFF datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bands = subparsers.add_parser("bands", help="Inspect and manage dataset band manifests.")
    bands_sub = bands.add_subparsers(dest="bands_command", required=True)

    init_parser = bands_sub.add_parser(
        "init",
        help="Inspect a TIFF folder and create bands.json.",
    )
    init_parser.add_argument("folder", nargs="?", default=".", help="TIFF folder (default: current directory).")
    init_parser.add_argument(
        "-o",
        "--output",
        help="Manifest path (default: <folder>/bands.json).",
    )
    init_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Write without an interactive confirmation prompt.",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing manifest.",
    )
    init_parser.add_argument(
        "--no-representations",
        action="store_true",
        help="Do not include automatically suggested display representations.",
    )
    init_parser.set_defaults(handler=_cmd_bands_init)

    show_parser = bands_sub.add_parser(
        "show",
        help="Show how a TIFF folder is interpreted without modifying it.",
    )
    show_parser.add_argument("folder", nargs="?", default=".", help="TIFF folder (default: current directory).")
    show_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the inspection proposal as JSON.",
    )
    show_parser.set_defaults(handler=_cmd_bands_show)

    validate_parser = bands_sub.add_parser(
        "validate",
        help="Validate bands.json against the TIFF files in a folder.",
    )
    validate_parser.add_argument("folder", nargs="?", default=".", help="TIFF folder (default: current directory).")
    validate_parser.add_argument(
        "-m",
        "--manifest",
        help="Manifest path (default: <folder>/bands.json).",
    )
    validate_parser.set_defaults(handler=_cmd_bands_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args) or 0)
    except (BandOrderError, TiffFormatError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


def _cmd_bands_show(args: argparse.Namespace) -> int:
    inspection = inspect_tiff_folder(args.folder)
    if args.json:
        print(json.dumps(inspection.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return 0

    _print_inspection(inspection)
    return 0


def _cmd_bands_init(args: argparse.Namespace) -> int:
    folder = Path(args.folder).expanduser().resolve()
    inspection = inspect_tiff_folder(folder)
    manifest = create_manifest(
        inspection,
        include_suggested_representations=not args.no_representations,
    )
    output = Path(args.output).expanduser().resolve() if args.output else folder / "bands.json"

    _print_inspection(inspection)
    print()
    _print_manifest_representations(manifest)

    if output.exists() and not args.force:
        raise ValueError(f"Manifest already exists: {output}. Use --force to overwrite it.")

    if not args.yes:
        answer = input(f"\nWrite manifest to {output}? [Y/n] ").strip().casefold()
        if answer not in {"", "y", "yes"}:
            print("No changes made.")
            return 1

    write_manifest(manifest, output)
    print(f"Manifest written: {output}")
    return 0


def _cmd_bands_validate(args: argparse.Namespace) -> int:
    folder = Path(args.folder).expanduser().resolve()
    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else folder / "bands.json"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    manifest = read_manifest(manifest_path)
    validate_manifest(manifest, folder)
    print(
        f"OK: {manifest_path} is valid for {folder} "
        f"({len(manifest.sources)} sources, {len(manifest.bands)} logical bands, "
        f"{len(manifest.representations)} representations)."
    )
    return 0


def _print_inspection(inspection) -> None:
    print(f"Scanning: {inspection.folder}")
    print()
    for source in inspection.sources:
        if source.detected_type == "rgb":
            detail = "RGB -> R, G, B"
        elif source.wavelength_nm is not None:
            detail = f"{source.wavelength_nm:g} nm"
        elif source.explicit_band_index is not None:
            detail = f"band index {source.explicit_band_index}"
        else:
            detail = "single band"
        print(f"  {source.file:<32} {detail:<20} {source.name}")

    if inspection.suggested_representations:
        print("\nSuggested representations:")
        for rep in inspection.suggested_representations:
            print(f"  {rep.id:<24} {rep.label}")
            print(f"    {rep.reason}")

    if inspection.warnings:
        print("\nWarnings:")
        for warning in inspection.warnings:
            print(f"  - {warning}")


def _print_manifest_representations(manifest) -> None:
    if not manifest.representations:
        print("No display representations will be added to the manifest.")
        return
    print("Representations to include:")
    for rep in manifest.representations:
        print(
            f"  {rep.id}: R={rep.red_band}, G={rep.green_band}, B={rep.blue_band} "
            f"({rep.role})"
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
