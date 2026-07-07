# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Device-less placeholders for ahead-of-time (no-GPU) compilation.

``ProxyArray`` / ``ProxyValue`` describe *only* the dtype (and, for arrays,
shape/contiguity) of an argument, with no backing GPU allocation. Pass them to
``make_<algo>(...)`` together with ``compute_capability=`` to compile an
algorithm on a machine that has no GPU (or no live data), then ``serialize()``
the result. The real device arrays / scalars are supplied later at ``__call__``.

Accessing a proxy's data pointer raises ``RuntimeError`` — a proxy can be used
to *build* an algorithm but never to *run* one.
"""

from __future__ import annotations

import numpy as np

_PROXY_DATA_ERROR = (
    "ProxyArray has no GPU data — it is a build-time placeholder only. "
    "Pass a real device array when calling the compiled algorithm."
)

_PROXY_VALUE_DATA_ERROR = (
    "ProxyValue has no data — it is a build-time placeholder only. "
    "Pass a real scalar or numpy array when calling the compiled algorithm."
)


class _ProxyCAI(dict):
    """CAI dict whose 'data' key raises on access."""

    def __missing__(self, key):
        if key == "data":
            raise RuntimeError(_PROXY_DATA_ERROR)
        raise KeyError(key)


class ProxyArray:
    """Dtype-only placeholder for a device array.

    Use in place of a real device array when calling ``make_<algo>()`` to
    trigger ahead-of-time compilation without allocating GPU memory — for
    example, on a build machine that has no GPU or no live data.

    Satisfies the ``DeviceArrayLike`` protocol:

    * ``is_device_array(proxy)`` -> ``True``
    * ``get_dtype(proxy)``       -> the dtype supplied at construction
    * ``get_data_pointer(proxy)``-> raises ``RuntimeError``
    * ``is_contiguous(proxy)``   -> ``True``

    Accessing the data pointer raises ``RuntimeError``; passing a
    ``ProxyArray`` to a compiled algorithm's ``__call__`` is not supported.

    Example::

        from cuda.compute import ProxyArray, make_reduce_into, OpKind
        import numpy as np

        reducer = make_reduce_into(
            d_in=ProxyArray(np.float32),
            d_out=ProxyArray(np.float32),
            op=OpKind.PLUS,
            h_init=np.zeros(1, dtype=np.float32),
            compute_capability=[80, 90],
        )
        reducer.serialize()
    """

    __slots__ = ("_dtype",)

    def __init__(self, dtype):
        self._dtype = np.dtype(dtype)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def __cuda_array_interface__(self) -> dict:
        return _ProxyCAI(
            {
                "shape": (1,),
                "typestr": self._dtype.str,
                "version": 3,
                "strides": None,  # C-contiguous
                # "data" is intentionally absent — accessing it raises RuntimeError
            }
        )

    def __repr__(self) -> str:
        return f"ProxyArray(dtype={self._dtype})"


class ProxyValue:
    """Dtype-only placeholder for a scalar / initial-value argument.

    Use in place of a real numpy scalar or array when calling ``make_<algo>()``
    to trigger ahead-of-time compilation without real data — for example, for
    the ``h_init`` argument of :func:`~cuda.compute.make_reduce_into`.

    Accessing the data of a ``ProxyValue`` raises ``RuntimeError``; passing
    one to a compiled algorithm's ``__call__`` is not supported.
    """

    __slots__ = ("_dtype",)

    def __init__(self, dtype):
        self._dtype = np.dtype(dtype)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __repr__(self) -> str:
        return f"ProxyValue(dtype={self._dtype})"


def is_proxy(obj) -> bool:
    """True if *obj* is a build-time placeholder (ProxyArray/ProxyValue)."""
    return isinstance(obj, (ProxyArray, ProxyValue))
