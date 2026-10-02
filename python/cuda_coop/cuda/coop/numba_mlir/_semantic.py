# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Give compiler dtypes stable identities for specialization and symbol reuse.

Two equivalent compiler types should select the same compiled operation even
when their internal bookkeeping differs. Adapt top-level Numba types to a
tagged class-and-name description before using the shared semantic encoder.
All other values follow that encoder's normal rules. The resulting hashable
descriptions feed cache keys and symbol hashes; they do not serialize compiler
objects for later reconstruction.
"""

from collections.abc import Hashable

import numba_cuda_mlir.numba_cuda.types as numba_types

from cuda.coop._core import semantic_token


def _numba_semantic_token(value: object) -> Hashable:
    """Describe a compiler dtype without serializing its implementation state.

    The common semantic-token encoder can inspect arbitrary object attributes.
    For a Numba type, cache and symbol identity instead use its concrete type
    class and printed spelling, so compiler bookkeeping does not become part
    of the specialization key. Only a top-level Numba type is normalized here;
    other values are passed directly to the common encoder.

    Parameters
    ----------
    value : object
        Compiler dtype or other semantic key component. A Numba ``Type`` is
        replaced with a tagged tuple of its class module, class qualified name,
        and ``str(value)`` before encoding.

    Returns
    -------
    Hashable
        Semantic description suitable for compiler cache and symbol
        keys. This is a description, not a digest or a reversible encoding.

    Raises
    ------
    TypeError
        The common encoder encounters an unsupported callable value.
    """

    if isinstance(value, numba_types.Type):
        value = (
            "numba-cuda-mlir-type",
            type(value).__module__,
            type(value).__qualname__,
            str(value),
        )
    return semantic_token(value)
