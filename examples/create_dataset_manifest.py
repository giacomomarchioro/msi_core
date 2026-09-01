"""Inspect a TIFF acquisition folder and write an editable bands.json manifest."""

from pathlib import Path

from msi_core import create_manifest, inspect_tiff_folder, write_manifest

folder = Path("./acquisition")
inspection = inspect_tiff_folder(folder)

print("Detected sources:")
for source in inspection.sources:
    print(
        f"- {source.file}: type={source.detected_type}, "
        f"wavelength_nm={source.wavelength_nm}"
    )

print("Suggested representations:")
for representation in inspection.suggested_representations:
    print(f"- {representation.id}: {representation.label}")

manifest = create_manifest(inspection)
write_manifest(manifest, folder / "bands.json")
print(f"Wrote {folder / 'bands.json'}")
