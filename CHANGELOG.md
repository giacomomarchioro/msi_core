# Changelog

## 0.3.6 - 2026-08-28

- Made each operation's Pydantic `Params` model the single source of truth for parameter documentation.
- Added non-empty `Field(..., description=...)` metadata for every public operation parameter so JSON Schema, Django forms, and other consumers receive the same documentation.
- Simplified `run()` docstrings to focus on algorithm behavior, inputs/outputs, scientific notes, examples, and references instead of duplicating parameter definitions.
- Replaced the old docstring-name synchronization test with a contract test requiring every public `Params` field to have a description.
- Kept operation APIs and runtime behavior unchanged.

## 0.3.5 - 2026-08-28

- Expanded NumPy-style docstrings for every public scientific operation, including all `Params` fields, return semantics, important exceptions, and scientific interpretation where relevant.
- Added documentation-contract tests requiring each public `Params` field to be mentioned by the corresponding `run()` docstring.
- Kept the existing isolated registry API (`create_default_registry()` and `OperationRegistry.clone()`) as the recommended extensibility pattern.
- Strengthened export tests to preserve per-render mapping metadata (`method`, source range, percentile settings, and 8-bit output range).
- No workflow/session abstraction or `ResultCollection` was added; multi-operation orchestration remains the responsibility of `msi_workflow`.

## 0.3.4 - 2026-08-28

- Added `create_default_registry()` and `OperationRegistry.clone()` so workflow engines, Django applications, plugins, and tests can use isolated registries; the module-level registry remains only a convenience API.
- Added the `visualization` layer with `render_grayscale()`, `render_rgb()`, and `save_render()`.
- Changed `save_result()` to preserve scientific TIFFs under `data/` and, by default, create 8-bit PNG renders under `renders/`.
- Added optional JPEG renders and min/max or percentile display mappings; all mappings are recorded in `metadata.json`.
- Added direct rendering of RGB representations declared by `DatasetManifest`.
- Added realistic PCA tests, including the large image-space standardization path and full-range uint16 data converted to float32.
- Added explicit protection against unsafe TIFF casts such as full-range uint16 to float16; the loader now raises a clear error instead of allowing infinities.
- Expanded RGB/ROI/render/export test coverage.

## 0.3.3

- Added the installed `msi-core` command-line entry point.
- Added `msi-core bands show [FOLDER]` to inspect TIFF sources without modifying them.
- Added `msi-core bands init [FOLDER]` to create `bands.json` from the same inspection and manifest APIs used by Python.
- Added `msi-core bands validate [FOLDER]` to validate a manifest against the current TIFF directory.
- Added `--yes`, `--force`, `--no-representations`, custom output/manifest paths, and JSON inspection output.
- Kept all band detection and manifest semantics in the public Python API; the CLI contains no independent scientific detection logic.

## 0.3.2 - 2026-08-27

### Added

- `DatasetManifest` schema with explicit physical TIFF `sources`, logical `bands`, and named RGB `representations`.
- `inspect_tiff_folder()` to inspect grayscale/RGB TIFF sources without writing files.
- Numeric-prefix wavelength detection for acquisition names such as `365 UV.tif` in addition to `365nm`.
- `create_manifest()` to turn an inspection into an editable manifest suitable for standalone JSON or SaaS/database storage.
- `validate_manifest()` and `write_manifest()` helpers.
- Automatic visible-representation suggestions:
  - RGB `VIS.tif`/`Visible.tif` when available;
  - a reviewable spectral composite using available bands nearest 630/535/475 nm.
- `load_tiff_folder(..., manifest=DatasetManifest(...))`, allowing a SaaS to pass a manifest directly from its database without creating a temporary JSON file.

### Changed

- New manifests use schema version `2.0` and model physical sources separately from logical bands.
- Version 1.0 `bands.json` files from 0.3.1 are upgraded in memory for backward compatibility.

## 0.3.1

- Added first-class single-page RGB TIFF ingestion.
- RGB TIFF files are split automatically into three consecutive `R`, `G`, `B` stack bands.
- Added per-band RGB channel metadata while preserving the source TIFF filename.
- Applied ROI extraction consistently to grayscale and RGB source files.
- Kept deterministic source-file ordering and canonical RGB channel ordering.
- Added tests for RGB splitting, ROI extraction, mixed grayscale/RGB source ordering, and unsupported multichannel TIFF rejection.

## 0.3.0

- Rebuilt the package as a small standalone stack-processing library.
- Added one module per scientific operation.
- Added typed `Region`, `BandMetadata`, `Stack`, and `Result` contracts.
- Added deterministic TIFF band ordering with explicit `BandOrderError`.
- Added TIFF-folder ROI loading and TIFF + JSON result export.
- Added dual parameter API: keyword arguments or Pydantic `Params` objects.
- Added explicit PCA modes: `spectral` and `image_space`.
- Added operation metadata, versioning, registry descriptions, and scientific method attribution.
- Added normalization, PCA, ICA, K-means, arithmetic, rotation, denoising, and hot-pixel operations.
- Removed workflow, Django, database, asset-reference, DAG, and Zarr concerns from the core package.
