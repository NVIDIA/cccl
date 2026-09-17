from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .._bindings import InitKind
from .protocols import is_device_array

if TYPE_CHECKING:
    import numpy as np

    from ..typing import DeviceArrayLike, GpuStruct

__all__ = ["get_init_kind", "sanitize_identifier"]


def get_init_kind(
    init_value: np.ndarray | DeviceArrayLike | GpuStruct | None,
) -> InitKind:
    match init_value:
        case None:
            return InitKind.NO_INIT
        case _ if is_device_array(init_value):
            return InitKind.FUTURE_VALUE_INIT
        case _:
            return InitKind.VALUE_INIT


def sanitize_identifier(name: str) -> str:
    """Sanitize a name to be a valid Python/LLVM identifier.

    This replaces any character that isn't alphanumeric or underscore with
    an underscore. This is needed because:
    - Lambda functions have __name__ = "<lambda>" which contains angle brackets
    - Python identifiers and LLVM/NVVM global names don't allow special characters

    Args:
        name: The name to sanitize (e.g., function __name__)

    Returns:
        A sanitized name safe for use as a Python identifier or LLVM symbol
    """
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)
