# Dataset manifests

`msi_core` separates three concepts:

1. **Source** — a physical TIFF file.
2. **Band** — a logical image channel contributed by a source.
3. **Representation** — a named RGB view assembled from three logical bands.

This distinction matters because a grayscale TIFF contributes one band while an RGB TIFF contributes three bands.

## Inspecting an acquisition folder

```python
from msi_core import inspect_tiff_folder

inspection = inspect_tiff_folder("./acquisition")

for source in inspection.sources:
    print(
        source.file,
        source.detected_type,
        source.wavelength_nm,
    )
```

For names such as:

```text
365 UV.tif
385 UV.tif
400 Deep Violet.tif
...
940 IR.tif
UVL.tif
VIS.tif
```

numeric prefixes are interpreted as wavelengths in nanometres. `VIS.tif` and `UVL.tif` are classified from the TIFF metadata itself rather than their filenames.

Inspection does not write or modify any file.

## Creating a manifest

```python
from msi_core import create_manifest, inspect_tiff_folder, write_manifest

inspection = inspect_tiff_folder("./acquisition")
manifest = create_manifest(inspection)
write_manifest(manifest, "./acquisition/bands.json")
```

The same `DatasetManifest` can be serialized to a Django `JSONField` instead of being written to disk.

## Example

A manifest can contain:

```json
{
  "schema_version": "2.0",
  "sources": [
    {
      "index": 0,
      "file": "475 Blue.tif",
      "type": "single_band",
      "name": "Blue",
      "bands": [
        {
          "id": "wl_475",
          "name": "Blue",
          "wavelength_nm": 475.0,
          "channel": null
        }
      ]
    },
    {
      "index": 1,
      "file": "VIS.tif",
      "type": "rgb",
      "name": "VIS",
      "bands": [
        {"id": "vis_r", "name": "VIS R", "wavelength_nm": null, "channel": "R"},
        {"id": "vis_g", "name": "VIS G", "wavelength_nm": null, "channel": "G"},
        {"id": "vis_b", "name": "VIS B", "wavelength_nm": null, "channel": "B"}
      ]
    }
  ],
  "representations": [
    {
      "id": "visible_rgb",
      "kind": "rgb",
      "label": "Visible image",
      "role": "visible",
      "red_band": "vis_r",
      "green_band": "vis_g",
      "blue_band": "vis_b",
      "description": "Suggested from an RGB TIFF named VIS/Visible."
    }
  ]
}
```

## Visible representations

A dataset may define more than one valid visible view. For example:

- the `R/G/B` channels of an acquired `VIS.tif`;
- a composite built from spectral bands such as 630 nm, 535 nm, and 475 nm.

`inspect_tiff_folder()` may suggest both. Suggestions are metadata proposals, not scientific assertions. Applications should allow users to review or edit them before treating a representation as canonical.

## SaaS use

A SaaS does not need to write `bands.json`:

```python
manifest = DatasetManifest.model_validate(dataset.manifest_json)
stack = load_tiff_folder(
    dataset.storage_path,
    manifest=manifest,
    region=region,
)
```

This keeps the scientific dataset model independent of Django while allowing Django to persist exactly the same schema.
