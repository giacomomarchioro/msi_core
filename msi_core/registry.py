from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable

from pydantic import BaseModel

from .attribution import OperationInfo
from .errors import OperationNotFoundError
from .operations import BUILTIN_OPERATION_MODULES
from .result import Result


@dataclass(frozen=True, slots=True)
class RegisteredOperation:
    info: OperationInfo
    params_model: type[BaseModel]
    run: Callable[..., Result]
    module_name: str

    def describe(self) -> dict[str, Any]:
        return {
            **self.info.model_dump(mode="json"),
            "parameter_schema": self.params_model.model_json_schema(),
            "module": self.module_name,
        }


class OperationRegistry:
    def __init__(self) -> None:
        self._operations: dict[str, RegisteredOperation] = {}

    def register_module(self, module: ModuleType) -> None:
        info = getattr(module, "INFO", None)
        params_model = getattr(module, "Params", None)
        run = getattr(module, "run", None)
        if not isinstance(info, OperationInfo):
            raise TypeError(f"Module {module.__name__} does not expose a valid INFO object.")
        if not isinstance(params_model, type) or not issubclass(params_model, BaseModel):
            raise TypeError(f"Module {module.__name__} does not expose a Pydantic Params model.")
        if not callable(run):
            raise TypeError(f"Module {module.__name__} does not expose a callable run function.")
        if info.operation_id in self._operations:
            raise ValueError(f"Operation '{info.operation_id}' is already registered.")
        self._operations[info.operation_id] = RegisteredOperation(
            info=info,
            params_model=params_model,
            run=run,
            module_name=module.__name__,
        )

    def resolve(self, operation_id: str) -> RegisteredOperation:
        try:
            return self._operations[operation_id]
        except KeyError as exc:
            raise OperationNotFoundError(f"Unknown operation '{operation_id}'.") from exc

    def list_available(self) -> list[RegisteredOperation]:
        return [self._operations[key] for key in sorted(self._operations)]

    def describe(self, operation_id: str) -> dict[str, Any]:
        return self.resolve(operation_id).describe()

    def clone(self) -> "OperationRegistry":
        """Return an independent registry containing the same registrations."""
        copied = OperationRegistry()
        copied._operations = dict(self._operations)
        return copied


def create_default_registry() -> OperationRegistry:
    """Create a fresh registry populated only with built-in operations."""
    created = OperationRegistry()
    for module in BUILTIN_OPERATION_MODULES:
        created.register_module(module)
    return created


# Convenience registry for simple standalone use. Workflow engines and tests that
# need isolation should call ``create_default_registry()`` or ``clone()``.
default_registry = create_default_registry()
registry = default_registry  # Backward-compatible convenience alias.
