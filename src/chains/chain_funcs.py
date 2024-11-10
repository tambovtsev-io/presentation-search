from typing import Any


def get_param_or_default(
    param_dict: dict,
    param_name: str,
    default_value: Any
) -> Any:
    """Get parameter from dictionary with fallback to default value

    Args:
        param_dict: Dictionary containing parameters
        param_name: Name of the parameter to get

    Returns:
        Parameter value from dict, or default value, or None
    """
    # Get item of a dict
    out = param_dict.get(param_name, default_value)

    # In case key is present but set to None
    if out is None:
        out = default_value
    return out
