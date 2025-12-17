import inspect

import numba
from numba import cuda
from numba.core.extending import as_numba_type
from numpy.typing import DTypeLike

from .typing import GpuStruct


def get_inferred_return_type(op, args: tuple):
    _, return_type = cuda.compile(op, args)
    return return_type


def to_numba_type(tp: GpuStruct | DTypeLike) -> numba.types.Type:
    """
    Convert a GpuStruct or DtypeLike to a numba type.
    """
    if value := as_numba_type.lookup.get(tp):
        return value
    return numba.from_dtype(tp)


def signature_from_annotations(func) -> numba.core.typing.Signature:
    """
    Create a numba signature from the annotations of a function.
    Annotations can be provided as either numpy dtypes or numba types.
    The function is assumed to have only positional arguments.

    If no annotations are provided or if any argument (or the return type) has
    no type annotation, raise a ValueError.
    """

    if len(func.__annotations__) == 0:
        raise ValueError(
            "Function must provide type annotations for all arguments and the return type"
        )

    argspec = inspect.getfullargspec(func)
    num_args = len(argspec.args)
    try:
        ret_ann = argspec.annotations["return"]
    except KeyError:
        raise ValueError("Function has incomplete annotations: missing return type")
    retty = to_numba_type(ret_ann)
    if num_args != len(argspec.annotations) - 1:  # -1 for the return type
        raise ValueError("One or more arguments are missing type annotations")
    argtys = tuple(
        to_numba_type(tp)
        for name, tp in func.__annotations__.items()
        if name != "return"
    )
    return retty(*argtys)
