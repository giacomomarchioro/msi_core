from .attribution import AlgorithmAttribution, OperationInfo, Person
from .errors import (
    BandOrderError,
    MsiCoreError,
    OperationNotFoundError,
    ParameterConflictError,
    RegionError,
    TiffFormatError,
)
from .io import (
    DatasetInspection,
    DatasetManifest,
    ManifestBand,
    ManifestSource,
    RGBRepresentation,
    SourceProposal,
    create_manifest,
    inspect_tiff_folder,
    load_tiff_folder,
    read_manifest,
    save_result,
    save_stack,
    validate_manifest,
    write_manifest,
)
from .pairwise import PairwiseResult, apply_pairwise, iter_pairwise
from .region import Region
from .registry import (
    OperationRegistry,
    RegisteredOperation,
    create_default_registry,
    default_registry,
    registry,
)
from .result import Result
from .stack import Band, BandMetadata, Stack
from .visualization import RenderMapping, render_grayscale, render_rgb, save_render

__version__ = "0.3.6"

__all__ = [
    "AlgorithmAttribution",
    "Band",
    "BandMetadata",
    "BandOrderError",
    "DatasetInspection",
    "DatasetManifest",
    "ManifestBand",
    "ManifestSource",
    "MsiCoreError",
    "OperationInfo",
    "OperationNotFoundError",
    "OperationRegistry",
    "PairwiseResult",
    "ParameterConflictError",
    "Person",
    "RGBRepresentation",
    "Region",
    "RegionError",
    "RegisteredOperation",
    "RenderMapping",
    "Result",
    "SourceProposal",
    "Stack",
    "TiffFormatError",
    "apply_pairwise",
    "create_default_registry",
    "create_manifest",
    "default_registry",
    "inspect_tiff_folder",
    "iter_pairwise",
    "load_tiff_folder",
    "read_manifest",
    "render_grayscale",
    "render_rgb",
    "registry",
    "save_render",
    "save_result",
    "save_stack",
    "validate_manifest",
    "write_manifest",
]
