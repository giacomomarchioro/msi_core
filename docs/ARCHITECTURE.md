# Architecture

## Scope

`msi-core` is a scientific computation library. It owns:

- in-memory stack and result models;
- deterministic TIFF-folder ingestion, including automatic RGB channel splitting;
- ROI validation and extraction;
- scientific operations;
- parameter validation;
- operation metadata and attribution;
- simple TIFF + JSON export.

It does not own users, projects, databases, queues, storage catalogs, recipes, or
collaboration features.

## Dependency direction

```text
Django / SaaS -> msi_workflow -> msi_core -> numerical libraries
```

Dependencies must never point from `msi_core` to Django or a workflow package.

## Core contracts

### Stack

A `Stack` stores one NumPy array with shape `(bands, y, x)` and a `BandMetadata`
record for every band. A grayscale TIFF contributes one band; a supported RGB TIFF
contributes three consecutive bands with `channel` metadata `R`, `G`, and `B`.

### Result

Every operation returns a `Result`. `Result.output` is another `Stack`, including
single-image results represented as a one-band stack. This keeps result handling
uniform while preserving direct NumPy access.

### Operation module

Each operation module exports:

- `INFO`: stable identity, version, description, input kind, optional attribution;
- `Params`: Pydantic model;
- `run(...)`: scientific computation.

The operation function accepts either a `Params` instance/dictionary or user-friendly
keyword arguments. Parameter validation occurs exactly once in the operation.

## Why this remains extensible

A workflow wrapper can resolve an operation through the registry, validate recipe
JSON with `Params`, execute `run`, save `Result`, and record `INFO` in provenance.
None of those tasks requires the scientific function to understand the workflow.


## Visualization boundary

Scientific operations return scientific arrays. Visualization is a separate, non-destructive layer.
`save_result()` persists the original arrays as TIFF files under `data/` and can create 8-bit PNG/JPEG renders under `renders/`. `render_grayscale()` and `render_rgb()` can also be called directly by standalone scripts or a future SaaS viewer.

## Registry isolation

The module-level `default_registry` is a convenience for simple standalone use. Systems that add custom operations or require isolation should call `create_default_registry()` or clone an existing registry and inject that instance into the higher-level workflow layer.
