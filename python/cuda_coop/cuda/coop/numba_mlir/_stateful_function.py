# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe a block Scan prefix callback and its mutable state type.

``StatefulFunction`` holds compile-time callback information. Callers pass a
separate one-item ThreadData or local array to each Scan call. Its value can
carry a running prefix across tiles without recompiling the callback. Planning
checks that the payload matches the descriptor before lowering its pointer.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import GenericAlias
from typing import Any

from ._semantic import _normalize_numba_callable


@dataclass(frozen=True, slots=True)
class StatefulFunction:
    """Pair a scan prefix callback with its state dtype.

    Pass this descriptor as ``prefix_op`` to
    :func:`cuda.coop.numba_mlir.scan` or its block scan variants, with the
    state as the third positional argument.

    Parameters
    ----------
    op : callable
        Device callback ``op(state, aggregate)``. It may update ``state[0]``
        and returns the tile prefix in the scanned value dtype. Accepts a
        decorated Numba device function or supported Python device callable.
        A functor class may define ``__call__(state, aggregate)`` instead. Its
        first argument receives the device state pointer; the compiler does
        not construct a Python instance.
    dtype : dtype-like
        Numeric dtype of the one-item state payload. It must exactly match
        the supplied state array, but may differ from the scanned value dtype.
    name : str, optional
        Nonempty diagnostic label, or ``None``. Does not affect the callback's
        identity or generated symbol.

    Notes
    -----
    Allocate a one-item :func:`cuda.coop.ThreadData` or local array and
    initialize every thread's state identically. CUB may call each lane of
    the first warp but applies lane zero's prefix; read the final state from
    block thread zero. See :ref:`coop-prefix-callbacks` for synchronization.
    Structured state and stateful binary scan operators are unsupported.

    Examples
    --------
    :func:`cuda.coop.numba_mlir.exclusive_sum` carries a running prefix across
    tiles with this descriptor.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`
        C++ Block Scan overloads accepting a block-prefix callback.
    """

    __class_getitem__ = classmethod(GenericAlias)

    op: Any
    dtype: Any
    name: str | None = None

    def __post_init__(self) -> None:
        """Normalize the callable and reject incomplete descriptor metadata.

        Unwrap an outer Numba dispatcher so compiler paths share one callback
        identity. Check that a dtype is supplied and any diagnostic label is a
        nonempty string. Numeric dtype and state-array checks run later, when
        the descriptor is used to plan or build a Scan provider.
        """

        normalized = _normalize_numba_callable(self.op)
        if not callable(normalized):
            raise TypeError("StatefulFunction op must be callable")
        if self.dtype is None:
            raise TypeError("StatefulFunction dtype must be provided")
        if self.name is not None and (
            not isinstance(self.name, str) or not self.name
        ):
            raise ValueError("StatefulFunction name must be a non-empty string")
        object.__setattr__(self, "op", normalized)


__all__ = ["StatefulFunction"]
