# Validation report

Release: `msi-core 0.3.1`

## Automated tests

The complete test suite was executed with Python 3.13.5:

```text
34 passed
```

The tests cover:

- typed region validation and bounds checking;
- stack metadata invariants and band selection;
- deterministic band order from wavelengths, explicit band indices, and manifests;
- explicit failure for ambiguous ordering and invalid TIFF inputs;
- ROI extraction, RGB channel splitting, and band-subset metadata preservation;
- keyword and Pydantic parameter APIs;
- spectral and image-space PCA conventions;
- ICA, K-means, pairwise arithmetic, rotation, denoising, and hot-pixel correction;
- pairwise combination and permutation strategies;
- operation registry descriptions and JSON Schema generation;
- scientific method attribution in a custom registry;
- raw floating-point TIFF export and optional 16-bit scaled export.

## End-to-end processing check

Synthetic TIFF inputs, including grayscale and RGB sources, were exercised through the loader and processing tests. A four-band grayscale folder was also processed end to end through:

```text
wavelength ordering -> ROI -> hot-pixel correction -> median denoise
-> rotation -> percentile normalization -> spectral PCA -> TIFF + JSON export
```

The check confirmed:

- deterministic output order `365, 450, 550, 650 nm`;
- ROI stack shape `(4, 20, 20)`;
- PCA output shape `(3, 20, 20)`;
- three exported TIFF images and a valid metadata sidecar;
- registry access to the PCA parameter JSON Schema.

## Packaging check

The wheel was built and installed into an isolated target directory. A separate smoke test
loaded TIFF files, extracted an ROI, normalized the stack, ran PCA, and inspected the
registry using only the installed wheel.

```text
ISOLATED WHEEL SMOKE TEST OK (2, 6, 8)
```

## Scope of this validation

The validation uses small synthetic arrays and TIFF files. It demonstrates software
correctness for the tested contracts, but it is not a scientific validation of a particular
processing method or parameter set on cultural-heritage acquisitions. Algorithm-specific
validation must use representative calibration targets and manuscript data.

## Command-line interface

The installed `msi-core` entry point is covered by tests for `bands show`, JSON
inspection output, non-interactive `bands init`, manifest creation, manifest
validation, and overwrite protection.
