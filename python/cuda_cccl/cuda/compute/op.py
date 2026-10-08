# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from __future__ import annotations

import sys
import sysconfig
import warnings

from . import types as cccl_types
from ._bindings import Op, OpKind, TypeEnum
from ._caching import CachableFunction, cache_with_registered_key_functions
from ._device_code import DeviceCode

try:
    from ._build_info import USING_V2  # type: ignore[import-not-found]
except ImportError:
    USING_V2 = False


def _native_layout(td: cccl_types.TypeDescriptor) -> tuple[list[int], int, int]:
    """Return (field offsets, size, alignment) as native CUDA/C++ would
    compute them for ``td``. ``offsets`` is ``[]`` for a non-struct."""
    if not isinstance(td, cccl_types.StructTypeDescriptor):
        if td.dtype is not None and td.dtype.subdtype is not None:
            # Fixed-size subarray (e.g. complex64[2]): takes its element
            # type's native alignment, and its size scales with the shape.
            base_dtype, shape = td.dtype.subdtype
            _, size, alignment = _native_layout(cccl_types.from_numpy_dtype(base_dtype))
            for dim in shape:
                size *= dim
            return [], size, alignment
        if td.dtype is not None and td.dtype.kind == "c":
            # libcu++ aligns cuda::std::complex<T> to 2 * sizeof(T) (its own
            # itemsize); NumPy only aligns it to sizeof(T).
            return [], td.dtype.itemsize, td.dtype.itemsize
        return [], td.size, td.alignment

    # Standard C struct packing: place each field at its own alignment, then
    # pad the whole struct up to a multiple of its largest field's alignment.
    offset = 0
    offsets = []
    struct_alignment = 1
    for field_type in td.fields.values():
        _, size, alignment = _native_layout(field_type)
        struct_alignment = max(struct_alignment, alignment)
        offset = -(-offset // alignment) * alignment
        offsets.append(offset)
        offset += size
    itemsize = -(-offset // struct_alignment) * struct_alignment
    return offsets, itemsize, struct_alignment


_UNAFFECTED = "Python-callable operators are unaffected."


def _check_raw_op_layout(
    td: cccl_types.TypeDescriptor, *, path: str = "<value>"
) -> None:
    """Raise if ``td``'s cuda.compute layout disagrees with the layout
    native CUDA/C++ would give it (issue #11347). A RawOp casts a raw
    pointer to a native struct; if cuda.compute built the value with a
    different layout, the RawOp reads or writes the wrong bytes.
    """
    if isinstance(td, cccl_types.PointerTypeDescriptor):
        # The pointee is a separate allocation: check it as its own root.
        _check_raw_op_layout(td.pointee, path=f"*{path}")
        return

    if not isinstance(td, cccl_types.StructTypeDescriptor):
        # Bare argument/result: no offsets, but alignment can still be wrong.
        _, _, native_alignment = _native_layout(td)
        if td.alignment != native_alignment:
            raise TypeError(
                f"{path} ({td.dtype}) is only {td.alignment}-byte aligned, "
                f"but native CUDA/C++ code needs {native_alignment}-byte "
                f"alignment. {_UNAFFECTED}"
            )
        return

    assert td.dtype is not None

    # Check every field's absolute offset first.
    _check_inline_fields(td, path, actual_base=0, native_base=0)

    # Then check the struct's own overall size and alignment: nothing
    # follows the root struct to absorb a mismatch there.
    _, expected_itemsize, expected_alignment = _native_layout(td)
    if td.dtype.itemsize != expected_itemsize or td.alignment != expected_alignment:
        raise TypeError(
            f"{path} has itemsize {td.dtype.itemsize} and alignment "
            f"{td.alignment}, but native CUDA/C++ code would give it "
            f"itemsize {expected_itemsize} and alignment "
            f"{expected_alignment}. {_UNAFFECTED}"
        )


def _check_inline_fields(
    td: cccl_types.StructTypeDescriptor,
    path: str,
    *,
    actual_base: int,
    native_base: int,
) -> None:
    """Check every field's absolute offset against native CUDA/C++,
    recursing into nested structs and pointer pointees."""
    assert td.dtype is not None
    expected_offsets, _, _ = _native_layout(td)
    for (field_name, field_type), native_offset in zip(
        td.fields.items(), expected_offsets
    ):
        actual_offset = int(td.dtype.fields[field_name][1])
        actual_abs = actual_base + actual_offset
        native_abs = native_base + native_offset
        field_path = f"{path}.{field_name}"

        if actual_abs != native_abs:
            raise TypeError(
                f"'{field_path}' sits at absolute offset {actual_abs} in "
                f"cuda.compute's layout, but native CUDA/C++ code would "
                f"place it at {native_abs}. {_UNAFFECTED}"
            )

        if isinstance(field_type, cccl_types.PointerTypeDescriptor):
            # The pointee is a separate allocation: check it as its own root.
            _check_raw_op_layout(field_type, path=field_path)
        elif isinstance(field_type, cccl_types.StructTypeDescriptor):
            # A nested struct's own size/alignment isn't checked here: a
            # following field in the enclosing struct can legitimately
            # absorb a difference there. Only its fields' offsets matter.
            _check_inline_fields(
                field_type, field_path, actual_base=actual_abs, native_base=native_abs
            )


def _is_well_known_op(op: OpKind) -> bool:
    return isinstance(op, OpKind) and op not in (OpKind.STATELESS, OpKind.STATEFUL)


class _OpAdapter:
    """
    Provides a unified interface for operators, whether they are:
    - Well-known operations (OpKind.PLUS, OpKind.MAXIMUM, etc.)
    - Stateless user-provided callables
    - Stateful user-provided callables
    """

    def compile(self, input_types, output_type=None) -> Op:
        """
        Compile this operator to an Op for CCCL interop.

        Args:
            input_types: Tuple of TypeDescriptors for input arguments
            output_type: Optional TypeDescriptor for return value (inferred if None)

        Returns:
            Compiled Op object for C++ interop
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def is_stateful(self) -> bool:
        """Return True if this op has runtime state."""
        return False

    def get_state(self) -> bytes:
        """
        Return the op's state bytes.
        """
        return b""

    @property
    def state_alignment(self) -> int:
        """Return the alignment requirement of the op's state bytes."""
        return 1

    def get_return_type(self, input_types):
        """Get the return type for this op given input types."""
        raise NotImplementedError(
            f"get_return_type not implemented for {self.__class__.__name__}"
        )


class _WellKnownOp(_OpAdapter):
    """Internal wrapper for well-known OpKind values."""

    __slots__ = ["_kind"]

    def __init__(self, kind: OpKind):
        if not _is_well_known_op(kind):
            raise ValueError(
                f"OpKind.{kind.name} is not a well-known operation. "
                "Use OpKind.PLUS, OpKind.MAXIMUM, etc."
            )
        self._kind = kind

    def compile(self, input_types, output_type=None) -> Op:
        # V2 supports some built-in operations on storage types, such as IDENTITY.
        if not USING_V2:
            for t in (*input_types, output_type):
                if t is not None and t.info.typenum == TypeEnum.STORAGE:
                    raise TypeError(
                        f"OpKind.{self._kind.name} is not supported for struct or other "
                        f"opaque types ({t.dtype}). Provide a custom operator instead."
                    )
        return Op(
            operator_type=self._kind,
            name="",
            ltoir=b"",
            state_alignment=1,
            state=b"",
        )

    @property
    def kind(self) -> OpKind:
        """The underlying OpKind."""
        return self._kind

    def __eq__(self, other):
        if not isinstance(other, _WellKnownOp):
            return False
        return self._kind == other._kind

    def __hash__(self):
        return hash(self._kind)


class RawOp(_OpAdapter):
    """
    ``RawOp`` lets you supply pre-compiled device code (LTO-IR) implementing a
    custom operator, bypassing the default Numba-based JIT pipeline.

    Example:
        Supplying C++ device code compiled to LTO-IR via NVRTC:

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/raw_op/cpp_stateless.py
            :language: python
            :start-after: # example-begin

    Args:
        name: The ABI name of the operator.
        ltoir: Raw ``bytes`` of pre-compiled LTO-IR implementing the operator
            (for example, produced by ``nvcc -dlto`` or NVRTC).
        state: Optional bytes representing the operator's state.
        state_alignment: Alignment requirement for the state bytes (default: 1).
        extra_ltoirs: Optional list of additional LTO-IR ``bytes`` to link.

    Notes:
        - The provided code must define a function with the specified name and the correct signature.
        - The function must use untyped pointers for all parameters and return type. The function body
          is responsible for correctly interpreting the pointer arguments based on the expected input and output types.
          For stateless operators, the signature is

             void func(void* arg1, void* arg2, ..., void* result)`

          For stateful operators, the first parameter must be a pointer to the state:

             void func(void* state, void* arg1, void* arg2, ...)
    """

    __slots__ = [
        "_ltoir",
        "_name",
        "_state",
        "_state_alignment",
        "_extra_ltoirs",
    ]

    def __init__(
        self,
        *,
        ltoir: bytes | DeviceCode,
        name: str,
        state: bytes = b"",
        state_alignment: int = 1,
        extra_ltoirs: list[bytes | DeviceCode] | None = None,
    ):
        if (
            not isinstance(state_alignment, int)
            or state_alignment < 1
            or (state_alignment & (state_alignment - 1)) != 0
        ):
            raise ValueError(
                "state_alignment must be a positive power of two, "
                f"got {state_alignment!r}"
            )
        self._ltoir = ltoir
        self._name = name
        self._state = state
        self._state_alignment = state_alignment
        self._extra_ltoirs = extra_ltoirs or []

    def compile(self, input_types, output_type=None) -> Op:
        # RawOp bodies are precompiled and may cast their raw pointer
        # arguments to a native CUDA/C++ struct, so their layout must match
        # exactly. Python-callable operators don't need this: numba-cuda-mlir
        # builds and reads its own values consistently either way.
        for t in (*input_types, output_type):
            if t is not None:
                _check_raw_op_layout(t)

        # Determine if stateful based on whether state is provided
        op_kind = OpKind.STATEFUL if self._state else OpKind.STATELESS

        return Op(
            operator_type=op_kind,
            name=self._name,
            ltoir=self._ltoir,
            state=self._state,
            state_alignment=self._state_alignment,
            extra_ltoirs=self._extra_ltoirs,
        )

    def get_state(self) -> bytes:
        """Return the op's state bytes."""
        return self._state

    @property
    def state_alignment(self) -> int:
        """Return the alignment requirement of the op's state bytes."""
        return self._state_alignment

    @property
    def _identity(self):
        # The actual *value* of the state bytes never affects the compiled
        # LTO-IR/glue code -- only their length (which fixes offsets baked
        # into generated deref code, see TransformIterator) and alignment
        # do. Keying the cache on the state's value would force a full
        # rebuild for every distinct runtime state (e.g. every distinct `n`
        # in a `sum(x) * (1/n)` mean), defeating the point of passing it as
        # state rather than baking it into the LTO-IR. Mirrors how iterator
        # state_bytes are excluded from IteratorBase.kind for the same
        # reason.
        return (
            self._ltoir,
            self._name,
            len(self._state),
            self._state_alignment,
            tuple(self._extra_ltoirs),
        )

    def __eq__(self, other):
        if not isinstance(other, RawOp):
            return False
        return self._identity == other._identity

    def __hash__(self):
        return hash(self._identity)


# Public aliases
OpAdapter = _OpAdapter


def _jit_op_adapter_factory():
    # helper that tries to import `_jit.py`. If it fails,
    # returns a function that raises an appropriate error when called.
    try:
        from ._jit import to_jit_op_adapter

        return to_jit_op_adapter
    except ModuleNotFoundError as e:
        # The minimal extras ship no JIT backend at all, so this is the error a
        # minimal-install user sees when they pass a Python callable. Prefer the
        # structured module name; fall back to the message for errors raised
        # without one.
        if "numba_cuda_mlir" in (e.name or str(e)):

            def _missing_jit_adapter(op):
                raise ImportError(
                    "numba-cuda-mlir is required to JIT compile Python callables"
                )

            return _missing_jit_adapter
        raise


# Resolved lazily on the first Python-callable operator (see
# _get_jit_op_adapter) so that `import cuda.compute` never imports the JIT
# backend. Importing it eagerly would make every consumer pay its import cost,
# would turn a broken backend installation into a package-wide import failure,
# and would fail outright on the minimal extras, which do not install it --
# even for users who only ever pass OpKind/RawOp operators.
_jit_adapter = None


def _get_jit_op_adapter():
    global _jit_adapter
    if _jit_adapter is None:
        # A concurrent first call may run the factory twice; that is benign
        # (the factory is idempotent) so no lock is taken.
        gil_was_off = (
            sysconfig.get_config_var("Py_GIL_DISABLED")
            and not getattr(sys, "_is_gil_enabled", lambda: True)()
        )
        _jit_adapter = _jit_op_adapter_factory()
        if gil_was_off and sys._is_gil_enabled():
            warnings.warn(
                "Compiling a Python callable operator imported a module that "
                "re-enabled the GIL for this process. To keep free-threaded "
                "execution, use OpKind or RawOp (pre-compiled LTO-IR) "
                "operators instead of Python callables.",
                RuntimeWarning,
            )
    return _jit_adapter


def make_op_adapter(op) -> OpAdapter:
    """
    Create an Op from a callable or well-known OpKind.

    Args:
        op: Callable or OpKind

    Returns:
        A value with appropriate subtype of _BaseOp
    """
    # Already an _OpAdapter instance:
    if isinstance(op, _OpAdapter):
        return op

    # Well-known operation
    if isinstance(op, OpKind):
        return _WellKnownOp(op)

    # It's a Python callable
    return _get_jit_op_adapter()(op)


cache_with_registered_key_functions.register(
    _WellKnownOp, lambda op: (op._kind.name, op._kind.value)
)

cache_with_registered_key_functions.register(
    OpKind, lambda kind: (kind.name, kind.value)
)

cache_with_registered_key_functions.register(
    type(lambda: None), lambda func: CachableFunction(func)
)

cache_with_registered_key_functions.register(RawOp, lambda op: op._identity)


__all__ = [
    "OpAdapter",
    "OpKind",
    "make_op_adapter",
    "RawOp",
]
