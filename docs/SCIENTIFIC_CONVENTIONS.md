# Scientific conventions

## Axis order

All stacks use:

```text
(bands, y, x)
```

A pixel spectrum is therefore `stack.data[:, y, x]`.

## Spatial regions

`Region` uses half-open coordinates:

```text
[x0, x1) x [y0, y1)
```

This matches NumPy slicing: `image[y0:y1, x0:x1]`.

## Band order

Band order is a scientific property. It is resolved from a complete `bands.json`
manifest, wavelengths in all filenames, or explicit band numbers in all filenames.
Ambiguous or partially encoded folders raise `BandOrderError`. RGB source files retain their source-file position and expand locally in canonical `R, G, B` order.

## RGB source images

A supported RGB TIFF is treated as three measurement channels and added to the stack
in canonical `R, G, B` order. Each channel keeps the same source filename and receives
explicit channel metadata. No nominal wavelength is invented for RGB channels: wavelength
metadata is inherited only when it was explicitly provided for the source in the manifest
or filename convention.

## PCA modes

### Spectral mode

```text
samples = pixels
features = spectral bands
```

The output images are per-pixel component scores. Component loadings describe the
contribution of spectral bands.

### Image-space mode

```text
samples = bands
features = pixel positions
```

The output images are reshaped PCA component vectors in pixel space. This mode is
provided to reproduce and compare the decomposition pattern used by the legacy
processing script.

The modes are not interchangeable. The selected convention must be reported when
interpreting or publishing results.

## Sign ambiguity

The signs of PCA and ICA components are not intrinsically meaningful: a component
and its negation describe the same axis/source up to sign. Visual comparisons should
account for this property.

## Scaling for export

Raw floating-point output should be retained for scientific analysis. Per-image
min-max conversion to 16-bit TIFF is intended for viewing and exchange only and may
remove information about absolute ranges across components.

## Hot-pixel correction

The included operation replaces strong local median residuals. It is a generic
algorithmic convenience, not an acquisition-system-specific validated protocol.
Thresholds should be tested on representative calibration and manuscript data.
