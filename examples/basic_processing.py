from pathlib import Path

from msi_core import Region, load_tiff_folder, save_result
from msi_core.operations import normalize, pca

input_folder = Path("./dataset")
output_folder = Path("./output")

stack = load_tiff_folder(
    input_folder,
    region=Region(x0=810, x1=2700, y0=5070, y1=6680),
)

normalized = normalize.run(
    stack,
    low_percentile=1,
    high_percentile=99,
    mode="per_band",
)

result = pca.run(
    normalized.output,
    n_components=min(8, normalized.output.n_bands),
    mode="spectral",
)

save_result(result, output_folder / "pca")
