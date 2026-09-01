# Migration from the legacy processing script

The legacy script combined loading, ROI cropping, preprocessing, decomposition,
pairwise arithmetic, preview generation, export, and metadata writing in a single
function. `msi-core` preserves the same practical flow while separating each concern.

## Legacy pattern

```python
process_image_stack_files(folder, roi, processing_modes=("normalize", "pca"))
```

## New pattern

```python
from msi_core import Region, load_tiff_folder, save_result
from msi_core.operations import normalize, pca

stack = load_tiff_folder(
    folder,
    region=Region(x0=810, x1=2700, y0=5070, y1=6680),
)

normalized = normalize.run(stack, mode="global")
result = pca.run(normalized.output, n_components=8, mode="image_space")
save_result(result, output_folder)
```

Use `mode="image_space"` to obtain the same sample/feature orientation as the legacy
PCA implementation. Use `mode="spectral"` for pixels-as-samples spectral PCA.

Rotation, denoising, and hot-pixel correction are explicit operations so they can be
recorded later in a workflow recipe instead of occurring silently during loading.
