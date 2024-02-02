import importlib


def get_class(base_module, class_name: str):
    """Get a class from a module. If the name contains a dot, the class will be
    imported as follows `"custom_package.optimizers/CustomOptimizer" ->
    from custom_package.optimizers import CustomOptimizer`.

    Args:
        base_module: Base module to import from, it is the fallback module if no dot is present in the class name.
        class_name (str): Class name.

    Returns:
        Class: Imported class.
    """

    module = base_module

    if "." in class_name:
        parts = class_name.split(".")
        base_module_name = ".".join(parts[:-1])
        class_name = parts[-1]

        module = importlib.import_module(base_module_name)
    elif base_module is None:
        raise ValueError(f"No parent module provided for class {class_name}")

    return getattr(module, class_name)
