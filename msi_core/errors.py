class MsiCoreError(Exception):
    """Base exception for msi-core."""


class BandOrderError(MsiCoreError):
    """Raised when a deterministic spectral band order cannot be established."""


class TiffFormatError(MsiCoreError):
    """Raised when a TIFF does not satisfy the single-band input contract."""


class RegionError(MsiCoreError, ValueError):
    """Raised when a spatial region is invalid or outside image bounds."""


class ParameterConflictError(MsiCoreError, ValueError):
    """Raised when both a parameter object and keyword parameters are supplied."""


class OperationNotFoundError(MsiCoreError, KeyError):
    """Raised when an operation identifier is not present in a registry."""
