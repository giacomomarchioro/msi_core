from pathlib import Path

from msi_core import Region, load_tiff_folder, save_result
from msi_core.operations import normalize, pca,ica,kmeans

input_folder = Path("/Volumes/AntComCAPSD/D46_sup_multispettrale/136r")
output_folder = Path("./output")
print("Creating stack")
stack = load_tiff_folder(
    input_folder,
    region=Region(3300,5270,7960,9410),
    dtype="float32"
)
print("Runnin pca")
pcaparams = pca.Params()
pcaparams.mode = 'image_space'
pcaparams.svd_solver = 'auto'


result = pca.run(
    stack,
    params=pcaparams
)
print("Saving...")
save_result(result, output_folder / "pca")

icaparams = ica.Params()
icaparams.n_components = 8

result2 = ica.run(
    stack=stack,
    params=icaparams
)

save_result(result=result2,output_dir=output_folder/"ica")

kmparams = kmeans.Params()
kmparams.n_clusters = 8

save_result(kmeans.run(stack,kmparams),output_dir=output_folder/"kmeans")

print("Process ended.")
