from msi_core.operations import BUILTIN_OPERATION_MODULES


def test_every_public_parameter_has_a_pydantic_description():
    """Params is the single source of truth for operation parameter documentation."""
    for module in BUILTIN_OPERATION_MODULES:
        missing = [
            name
            for name, field in module.Params.model_fields.items()
            if not field.description or not field.description.strip()
        ]
        assert not missing, (
            f"{module.__name__}.Params fields have no description: {missing}"
        )


def test_every_operation_run_has_core_numpy_docstring_sections():
    for module in BUILTIN_OPERATION_MODULES:
        docstring = module.run.__doc__ or ""
        assert "Parameters\n----------" in docstring, module.__name__
        assert "Returns\n-------" in docstring, module.__name__


def test_every_operation_run_points_keyword_parameters_to_params_model():
    for module in BUILTIN_OPERATION_MODULES:
        docstring = module.run.__doc__ or ""
        assert "**kwargs" in docstring, module.__name__
        assert ":class:`Params`" in docstring, module.__name__
