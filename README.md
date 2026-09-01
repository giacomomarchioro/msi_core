# msi-core

`msi-core` is a small, standalone Python library for processing multispectral image
stacks. Its central workflow is deliberately simple:

```text
TIFF folder -> optional ROI -> Stack -> operation -> Result -> scientific TIFF + 8-bit render + JSON
```

The library is intended to remain usable from scripts, notebooks, tests, and future
workflow systems. It does **not** depend on Django, a database, a task queue, Zarr,
or a workflow engine.

## Design principles

- One scientific operation per Python module.
- A small, direct API for researchers and notebooks.
- Pydantic parameter models for validation and future UI generation.
- Deterministic band ordering; ambiguous folders fail explicitly.
- Typed regions and per-band metadata.
- Scientific method attribution is distinct from source-code authorship.
- Processing functions calculate and return data; they do not write to databases.

## Installation

```bash
python -m pip install -e '.[test]'
```

Optional image-processing dependencies:

```bash
python -m pip install -e '.[image,test]'
```

## Quick start

Assume a folder containing grayscale spectral TIFF files. Single-page RGB TIFF files are also accepted and are automatically split into consecutive R, G, B bands:

```text
dataset/
├── band_365nm.tif
├── band_450nm.tif
├── band_550nm.tif
└── band_650nm.tif
```

Load a rectangular region and run spectral PCA:

```python
from msi_core import Region, load_tiff_folder, save_result
from msi_core.operations import normalize, pca

region = Region(x0=810, x1=2700, y0=5070, y1=6680)
stack = load_tiff_folder("./dataset", region=region)

normalized = normalize.run(
    stack,
    low_percentile=1,
    high_percentile=99,
    mode="per_band",
)

result = pca.run(
    normalized.output,
    n_components=4,
    mode="spectral",
)

save_result(result, "./output/pca")
```

The same operation can also receive an explicitly validated parameter object:

```python
params = pca.Params.model_validate(
    {"n_components": 4, "mode": "spectral"}
)
result = pca.run(stack, params=params)
```

This dual interface is intentional:

- scripts and notebooks can use keyword arguments;
- a future recipe engine or Django UI can validate JSON through `Params`.

## Dataset manifest CLI

The same dataset-inspection API used by Python and future SaaS integrations is also
available from the command line:

```bash
msi-core bands show .
msi-core bands init .
msi-core bands validate .
```

`bands show` inspects the folder without changing it. `bands init` proposes a
`DatasetManifest`, shows detected TIFF sources and suggested RGB representations,
and writes `bands.json` after confirmation. `bands validate` checks an existing
manifest against the TIFF files currently present in the folder.

For scripts and automated workflows, confirmation can be skipped explicitly:

```bash
msi-core bands init . --yes
```

Existing manifests are never overwritten unless `--force` is supplied. The CLI is
intentionally a thin wrapper around `inspect_tiff_folder()`, `create_manifest()`,
`validate_manifest()`, and `write_manifest()`, so Django or another SaaS layer can
use the exact same logic without invoking a subprocess.

## Deterministic band ordering

`load_tiff_folder()` never falls back to an ordinary alphabetical sort. The order is
resolved using the following policy:

1. `bands.json`, when present;
2. numeric wavelengths in every filename, such as `365nm`;
3. explicit numeric band identifiers in every filename, such as `band_001`;
4. otherwise `BandOrderError` is raised.

This prevents a silent change in spectral order across machines or filesystems. Within an RGB TIFF, channel order is always canonical R, G, B and the three channels remain consecutive.

A manifest is recommended when filenames are not self-describing. The current
manifest schema separates physical TIFF sources, logical bands, and display
representations. For example:

```json
{
  "schema_version": "2.0",
  "sources": [
    {
      "index": 0,
      "file": "365 UV.tif",
      "type": "single_band",
      "name": "UV",
      "bands": [
        {"id": "wl_365", "name": "UV", "wavelength_nm": 365, "channel": null}
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
      "blue_band": "vis_b"
    }
  ]
}
```

When a manifest is present, it must list every TIFF source file in the folder exactly
once. A listed RGB TIFF expands automatically into three consecutive R, G, B bands.

## RGB TIFF input

A single-page TIFF declared as RGB contributes three bands to the stack in canonical
`R, G, B` order. `tifffile` is used directly, so no OpenCV BGR conversion is involved.
Each generated `BandMetadata` record keeps the original TIFF filename and a `channel`
value of `R`, `G`, or `B`.

For example, if `band_002.tif` is RGB, the logical stack order around that source is:

```text
... -> band_002:R -> band_002:G -> band_002:B -> ...
```

RGB source files follow the same deterministic source-file ordering rules as grayscale
TIFF files. Multi-page TIFF and non-RGB multichannel TIFF files remain unsupported.

## Regions

Coordinates use half-open intervals:

```python
Region(x0=100, x1=400, y0=200, y1=500)
```

means `x = 100..399` and `y = 200..499`. The resulting image has width `300` and
height `300`.

`msi-core` first attempts memory-mapped ROI access for TIFF files that support it.
For formats that cannot be memory mapped, it falls back to decoding the image and
then cropping it. This fallback is correct but may be slower for compressed or
strip-organized TIFF files.

## PCA conventions

`pca.Params.mode` makes the scientific convention explicit:

- `spectral` (default): pixels are samples and bands are features. Output images are
  score images, which is the conventional spectral decorrelation formulation.
- `image_space`: bands are samples and pixel positions are features. Output images
  are reshaped component vectors, matching the formulation used by the legacy
  script on which this proof of concept was based.

Both are mathematically valid decompositions, but they answer different questions.
The selected mode is recorded in the result metadata.

## Operations included in the proof of concept

- `normalize`
- `pca`
- `ica`
- `kmeans`
- `subtraction`
- `division`
- `log_ratio`
- `rotate`
- `denoise`
- `hot_pixel`

Pairwise operations operate on two selected bands:

```python
from msi_core.operations import subtraction

result = subtraction.run(stack[0], stack[3])
```

They can also be applied to all combinations or ordered pairs:

```python
from msi_core import apply_pairwise
from msi_core.operations import subtraction

items = apply_pairwise(subtraction.run, stack, ordered=True)
```

## Operation metadata and scientific attribution

Every operation module exposes:

```text
INFO
Params
run(...)
```

`INFO` includes a stable operation identifier and version. A method attributable to
specific scientific authors can also provide structured attribution:

```python
from msi_core import AlgorithmAttribution, OperationInfo, Person

INFO = OperationInfo(
    operation_id="author_method",
    operation_version="1.0.0",
    display_name="Author Method",
    description="A method proposed for a specific scientific purpose.",
    category="enhancement",
    input_kind="stack",
    attribution=AlgorithmAttribution(
        method_name="Author Method",
        authors=[
            Person(
                name="Jane Doe",
                affiliation="Example University",
                orcid="0000-0000-0000-0000",
            )
        ],
        status="published",
        citation="Doe, J. (2026). ...",
        doi="10.0000/example",
    ),
)
```

This attribution describes the **scientific method**, not the people who implemented
the Python code. Code contributions remain attributable through version control and
project contributor records.

When an operation returns its `INFO` in the `Result`, the same attribution is also
written to the exported `metadata.json` sidecar, so it remains attached to data that
leave the registry or web application.

The registry exposes operation information and Pydantic JSON Schema. For simple
standalone use, `registry`/`default_registry` is available as a convenience:

```python
from msi_core import registry

schema = registry.describe("pca")["parameter_schema"]
```

Workflow engines, Django applications, plugins, and isolated tests should create
their own registry instead of mutating the shared convenience instance:

```python
from msi_core import create_default_registry

registry = create_default_registry()
registry.register_module(my_custom_operation)
```

`OperationRegistry.clone()` is also available when an existing configured registry
needs to be copied. A future Django interface can use the parameter schema to
generate forms without adding Django as a dependency of `msi-core`.

## Saving scientific data and renders

```python
save_result(result, "./output/pca")
```

creates two explicit layers:

```text
output/pca/
├── data/
│   ├── pca_000_PC1.tif
│   └── ...
├── renders/
│   ├── pca_000_PC1.png
│   └── ...
└── metadata.json
```

Files under `data/` preserve the scientific result values and dtype, including
negative floating-point PCA/ICA values. Files under `renders/` are 8-bit display
representations intended for ordinary image viewers, web interfaces, figures, and
publication workflows. Rendering never modifies the scientific TIFF files.

The default render uses a linear min/max mapping. Percentile-based display mapping
is available without turning it into a processing operation:

```python
save_result(
    result,
    "./output/pca",
    render_stretch="percentile",
    render_percentiles=(1, 99),
)
```

JPEG is also available when compact output is preferred:

```python
save_result(result, "./output/pca", render_format="jpeg", jpeg_quality=95)
```

Every mapping used to create a render is recorded in `metadata.json`. Renders can
be disabled with `save_render=False`.

## RGB and grayscale visualization

Visualization is separate from scientific processing. A scientific array can be
rendered without modifying it:

```python
from msi_core import render_grayscale, save_render

image8, mapping = render_grayscale(result.output[0].data)
save_render(image8, "PC1.png")
```

Dataset RGB representations declared in `bands.json` can be rendered directly from
the loaded stack:

```python
from msi_core import render_rgb, save_render

representation = manifest.representations[0]
rgb8, render_metadata = render_rgb(stack, representation)
save_render(rgb8, "visible.png")
```

`render_rgb()` supports both independent channel scaling and a shared RGB scale.
The selected display mapping is returned as metadata so a SaaS can persist it.

## Intended future layering

```text
Django / SaaS
      |
msi_workflow
      |
msi_core
      |
NumPy / SciPy / scikit-learn / TIFF I/O
```

`msi_workflow` may later add recipes, provenance, storage references, asynchronous
jobs, caching, and distributed execution. These features should wrap the stable
`Stack`, `Result`, `Params`, and operation metadata contracts rather than complicate
the scientific functions themselves.

## Current limitations

- Single-page 2-D grayscale and 3-channel RGB TIFF inputs are supported; MultiTIFF and other multichannel layouts are rejected.
- The selected ROI is materialized in memory.
- PCA, ICA, and clustering operate in memory and are intended for manageable ROIs.
- TIFF window efficiency depends on the physical TIFF layout.
- No workflow engine, database integration, or asynchronous execution is included.
- Hot-pixel correction is a general median-residual method and should be validated
  against the acquisition system before being treated as a diagnostic protocol.

See `docs/SCIENTIFIC_CONVENTIONS.md` and `docs/ADDING_AN_OPERATION.md` for more detail.

## Dataset inspection and `bands.json`

For acquisition folders with mixed spectral and RGB TIFF sources, use the same Python API that a future SaaS can call:

```python
from msi_core import create_manifest, inspect_tiff_folder, write_manifest

inspection = inspect_tiff_folder("./acquisition")
manifest = create_manifest(inspection)
write_manifest(manifest, "./acquisition/bands.json")
```

`inspect_tiff_folder()` recognizes numeric-prefix acquisition names such as `365 UV.tif`, classifies RGB sources from TIFF metadata, and can suggest visible representations. `DatasetManifest` models physical sources, logical bands, and RGB representations separately.

A web application can persist `manifest.model_dump(mode="json")` directly in a database and later pass the validated `DatasetManifest` object to `load_tiff_folder()`. Writing a JSON file is optional.

See [`docs/DATASET_MANIFEST.md`](docs/DATASET_MANIFEST.md) for the full schema and SaaS-oriented usage.
