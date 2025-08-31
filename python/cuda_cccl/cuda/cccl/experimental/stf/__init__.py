from ._stf_bindings import (
    context,
    data_place,
    dep,
    exec_place,
)
from .decorator import jit  # Python-side kernel launcher

__all__ = [
    "context",
    "dep",
    "exec_place",
    "data_place",
    "jit",
]

def has_torch() -> bool:
    import importlib.util
    return importlib.util.find_spec("torch") is not None

def has_numba() -> bool:
    import importlib.util
    return importlib.util.find_spec("numba") is not None
