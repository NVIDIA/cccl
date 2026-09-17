# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Callable, cast

import numpy as np

from .. import _bindings
from .. import _cccl_interop as cccl
from .._caching import cache_build_results, cache_with_registered_key_functions
from .._cccl_interop import (
    get_value_type,
    make_pointer_object,
    set_cccl_iterator_state,
    to_cccl_value_state,
)
from .._serialization import (
    BUILD_RESULTS,
    CONDITIONAL,
    ENUM,
    ITER,
    OP,
    VALUE,
    Serializable,
)
from .._utils import get_init_kind
from .._utils.protocols import get_data_pointer, get_dtype, validate_and_get_stream
from .._utils.temp_storage_buffer import TempStorageBuffer
from ..determinism import Determinism
from ..op import OpAdapter, make_op_adapter
from ..typing import (
    DeviceArrayLike,
    GpuStruct,
    IteratorBase,
    IteratorT,
    Operator,
    _Struct,
)


# Perf: get_data_pointer() (_utils/protocols.py) is a generic fallback chain
# -- for a CuPy array it unconditionally tries arr.data_ptr() (PyTorch's
# convention) FIRST, which raises AttributeError every single call before
# falling through to the working arr.data.ptr line. Raising/catching an
# AttributeError is not free in CPython (exception object construction,
# stack unwind) -- measured ~6.7x slower than a direct attribute read for a
# CuPy array. Since which accessor works depends only on the array's type
# (not its value), cache the choice per-type once and skip the doomed
# try/except on every execute() call. Process-wide, not per-reducer: the
# right accessor for e.g. cupy.ndarray never changes.
_PTR_GETTER_CACHE: dict = {}


def _fast_data_pointer(arr):
    t = type(arr)
    getter = _PTR_GETTER_CACHE.get(t)
    if getter is None:
        if hasattr(t, "data_ptr"):
            getter = lambda a: a.data_ptr()  # noqa: E731
        elif hasattr(t, "data"):
            getter = lambda a: a.data.ptr  # noqa: E731
        else:
            getter = lambda a: a.__cuda_array_interface__["data"][0]  # noqa: E731
        _PTR_GETTER_CACHE[t] = getter
    return getter(arr)


# Sentinel for "stream not yet validated" -- distinct from None, which is
# itself a valid stream argument (the null stream).
_UNSET_STREAM = object()


class _Reduce(Serializable):
    __slots__ = [
        "_bound_build_result",
        "d_in_cccl",
        "d_out_cccl",
        "h_init_cccl",
        "op_cccl",
        "init_kind",
        "build_results",
        "loaded_build_result",
        "device_reduce_fn",
        # Perf: __call__ used to unconditionally redo three things that are
        # data-independent (device_reduce_fn only depends on which build
        # result got loaded; a *stateless* op's state is always b"" by
        # definition). Caching them costs one extra `is` check but skips a
        # `Determinism(...)` reconstruction / adapter re-derivation on every
        # call in the common "same reducer, same op, repeated calls" case.
        #
        # These hold strong references (not bare ids) so identity can never
        # false-positive against a GC'd-and-reused object. h_init is
        # deliberately NOT cached this way: it's a plain mutable
        # numpy array/GpuStruct the caller may mutate in place between calls
        # while reusing the same object, and identity alone can't detect
        # that -- so h_init state is still re-marshaled every call.
        "_last_loaded_build_result",
        "_last_stateless_op",
        # Perf: is_kind_pointer() reads fixed metadata (iter_data.type) baked
        # into the iterator descriptor when it was built -- it can never
        # change for a given d_in_cccl/d_out_cccl object, so re-checking it
        # on every execute() call (inside set_cccl_iterator_state) is
        # redundant. Cached once here; execute() uses it to skip straight to
        # the pointer-update fast path instead of re-deriving the branch.
        "_d_in_is_ptr",
        "_d_out_is_ptr",
        # Perf: make_pointer_object() allocates a new Pointer (Cython
        # __cinit__) every call. Pointer.ptr/.ref are plain `cdef` fields
        # (not `cdef public`), so a pure-Python caller can't mutate an
        # existing Pointer's fields directly -- Pointer.rebind() (added
        # alongside this) does that from Cython. Hold one reusable Pointer
        # per pointer-kind iterator (only when _d_in_is_ptr/_d_out_is_ptr is
        # True) and rebind() + reassign .state on it every execute() call
        # instead of allocating a fresh one. Safe to reuse across calls:
        # the iterator's `.state` setter copies the raw pointer value out of
        # the Pointer wrapper into its own C struct immediately on
        # assignment (see Iterator.state's setter) -- it never holds a live
        # reference to the wrapper's mutable state, so rebind-then-reassign
        # each call is equivalent to allocating fresh each time.
        "_d_in_ptr_obj",
        "_d_out_ptr_obj",
        # Perf: validate_and_get_stream() re-derives the same int handle
        # every call for a fixed stream object (its __cuda_stream__()
        # protocol result can't change over that object's lifetime). Cache
        # by identity, same pattern as _last_loaded_build_result/
        # _last_stateless_op above. Uses a private sentinel (not None) as
        # the "never validated yet" marker, because stream=None is itself a
        # valid, meaningful input (the null stream) -- None can't double as
        # both "unset" and "a real cached value" the way it does for the
        # other caches above, where None is never a legitimate op/build
        # result.
        "_last_stream_obj",
        "_last_stream_handle",
    ]

    __serialization_schema__ = (
        ("init_kind", ENUM(_bindings.InitKind)),
        ("d_in_cccl", ITER),
        ("d_out_cccl", ITER),
        ("op_cccl", OP),
        (
            "h_init_cccl",
            CONDITIONAL(
                "init_kind",
                {
                    _bindings.InitKind.NO_INIT: None,
                    _bindings.InitKind.VALUE_INIT: VALUE,
                },
            ),
        ),
        ("build_results", BUILD_RESULTS(_bindings.DeviceReduceBuildResult)),
    )

    # TODO: constructor shouldn't require concrete `d_in`, `d_out`:
    def __init__(
        self,
        d_in: DeviceArrayLike | IteratorT,
        d_out: DeviceArrayLike | IteratorT,
        op: OpAdapter,
        h_init: np.ndarray | GpuStruct | None,
        determinism: Determinism,
        compute_capability=None,
    ):
        self.d_in_cccl = cccl.to_cccl_input_iter(d_in)
        self.d_out_cccl = cccl.to_cccl_output_iter(d_out)
        self._d_in_is_ptr = self.d_in_cccl.is_kind_pointer()
        self._d_out_is_ptr = self.d_out_cccl.is_kind_pointer()
        # Placeholder Pointer objects (arg=0), rebound to the real address on
        # the first execute() call -- see the __slots__ comment for why this
        # is reused rather than allocated fresh every call.
        self._d_in_ptr_obj = make_pointer_object(0, None) if self._d_in_is_ptr else None
        self._d_out_ptr_obj = make_pointer_object(0, None) if self._d_out_is_ptr else None
        self._last_stream_obj = _UNSET_STREAM
        self._last_stream_handle = None

        self.init_kind = get_init_kind(h_init)

        self.h_init_cccl: _bindings.Value | None

        match self.init_kind:
            case _bindings.InitKind.NO_INIT:
                # The non-deterministic (atomic) kernel folds the initial value
                # into block zero's aggregate, so it requires one to exist.
                if Determinism(determinism) is Determinism.NOT_GUARANTEED:
                    raise ValueError(
                        "h_init=None is not supported with Determinism.NOT_GUARANTEED"
                    )
                self.h_init_cccl = None
                value_type = get_value_type(d_in)
                init_value_type_info = self.d_in_cccl.value_type

            case _bindings.InitKind.FUTURE_VALUE_INIT:
                raise ValueError(
                    "Passing a device array as h_init is not supported for reduction"
                )

            case _bindings.InitKind.VALUE_INIT:
                h_init_typed = cast(np.ndarray | GpuStruct, h_init)
                self.h_init_cccl = cccl.to_cccl_value(h_init_typed)
                value_type = get_value_type(h_init_typed)
                init_value_type_info = self.h_init_cccl.type

        # Compile the op with value types
        self.op_cccl = op.compile((value_type, value_type), value_type)

        # loaded_build_result / device_reduce_fn are bound lazily on the first
        # __call__ (see _bind_device_reduce_fn).
        self._last_loaded_build_result = None
        self._last_stateless_op = None
        self.build_results, self._bound_build_result = cache_build_results(
            _bindings.DeviceReduceBuildResult,
            d_in,
            d_out,
            op,
            h_init,
            determinism,
            self.init_kind,
            compute_capability=compute_capability,
            builder=lambda: cccl.build_for_ccs(
                _bindings.DeviceReduceBuildResult,
                self.d_in_cccl,
                self.d_out_cccl,
                self.op_cccl,
                init_value_type_info,
                self.init_kind,
                determinism,
                compute_capability=compute_capability,
            ),
        )

    def _after_deserialize(self) -> None:
        # deserialize() bypasses __init__ (see Serializable.deserialize),
        # where these two caches are otherwise initialized -- without this,
        # a deserialized _Reduce would raise AttributeError the first time
        # _bind_device_reduce_fn/__call__ reads them.
        self._last_loaded_build_result = None
        self._last_stateless_op = None
        # d_in_cccl/d_out_cccl ARE part of the serialization schema and are
        # already restored by the time _after_deserialize runs (schema
        # members are set in order before this hook fires) -- safe to read.
        self._d_in_is_ptr = self.d_in_cccl.is_kind_pointer()
        self._d_out_is_ptr = self.d_out_cccl.is_kind_pointer()
        self._d_in_ptr_obj = make_pointer_object(0, None) if self._d_in_is_ptr else None
        self._d_out_ptr_obj = make_pointer_object(0, None) if self._d_out_is_ptr else None
        self._last_stream_obj = _UNSET_STREAM
        self._last_stream_handle = None

    def _bind_device_reduce_fn(self) -> None:
        # Derived from the loaded build result (not serialized); bound at __call__
        # once resolve_build_result picks + loads the current device's build result.
        # compute() handles both init kinds: it ignores h_init for NO_INIT builds.
        #
        # Perf: device_reduce_fn is a pure function of `loaded_build_result`'s
        # identity -- nothing else it's derived from can change without
        # `loaded_build_result` itself changing first. resolve_build_result()
        # already returns the same object every call on the construction-time
        # bound fast path, so this guard turns a `Determinism(...)`
        # reconstruction + two attribute writes into a no-op on repeat calls.
        # For deserialized/unbound wrappers (resolved fresh per call), this
        # still rebinds whenever the resolved result actually changes (e.g.
        # a different device), matching prior behavior exactly.
        if self.loaded_build_result is self._last_loaded_build_result:
            return
        if (
            Determinism(self.loaded_build_result.determinism)
            is Determinism.NOT_GUARANTEED
        ):
            self.device_reduce_fn = self.loaded_build_result.compute_nondeterministic
        else:
            self.device_reduce_fn = self.loaded_build_result.compute
        self._last_loaded_build_result = self.loaded_build_result

    def set_op(self, op: Callable | OpAdapter) -> None:
        """Explicitly rebind this reducer's operator.

        ``execute()`` (unlike ``__call__``) never re-derives op state on its
        own -- if the operator or its captured state has changed since
        construction or the last ``set_op()``, you must call this first, or
        ``execute()`` will silently launch with stale op state. Call this
        only when it has actually changed; it is not needed after every
        ``execute()``.
        """
        op_adapter = make_op_adapter(op)
        self.op_cccl.state = op_adapter.get_state()

    def set_h_init(self, h_init: np.ndarray | GpuStruct) -> None:
        """Explicitly rebind this reducer's initial value.

        ``execute()`` never re-reads ``h_init`` on its own -- if its value
        has changed (including in-place mutation of the same array object)
        since construction or the last ``set_h_init()``, you must call this
        first, or ``execute()`` will silently launch with the stale value.
        """
        if self.init_kind is _bindings.InitKind.VALUE_INIT:
            self.h_init_cccl = cast(_bindings.Value, self.h_init_cccl)
            self.h_init_cccl.state = to_cccl_value_state(h_init)

    def execute(
        self,
        *,
        temp_storage,
        d_in,
        d_out,
        num_items: int,
        stream=None,
    ):
        """Minimal per-call path: updates only the ``d_in``/``d_out`` pointer
        state and issues the launch. Unlike ``__call__``, this never
        re-derives op state or ``h_init`` state -- construct with the right
        initial values (or call ``set_op``/``set_h_init`` beforehand) and
        reuse this for repeat calls against the same op/h_init. This is the
        deliberately-unsafe-by-default counterpart to ``__call__``: it trades
        the implicit per-call re-derivation (and its correctness margin for
        mutated/varying op or h_init) for the lower fixed cost of a call
        where op and h_init are call-invariant, which is the common case in
        a hot loop. Use ``__call__`` (via ``reduce_into``/the object
        returned by ``make_reduce_into``) if you can't make that guarantee.
        """
        self.loaded_build_result = cccl.resolve_build_result(
            self.build_results, self._bound_build_result
        )
        self._bind_device_reduce_fn()

        # Perf: skip set_cccl_iterator_state's is_kind_pointer() re-check --
        # _d_in_is_ptr/_d_out_is_ptr were already determined once at
        # construction (see __init__/_after_deserialize) and cannot change
        # for these iterator objects. Falls back to the general helper for
        # the (rarer) non-pointer-kind case (a custom Iterator/Transform/Zip
        # input), which still needs its own state-reading logic.
        #
        # For the pointer-kind case: _fast_data_pointer skips the doomed
        # try/except get_data_pointer() would otherwise redo every call
        # (see module docstring), and rebind() mutates the already-allocated
        # Pointer wrapper instead of allocating a new one via
        # make_pointer_object -> Pointer.__cinit__.
        if self._d_in_is_ptr:
            self._d_in_ptr_obj.rebind(_fast_data_pointer(d_in), d_in)
            self.d_in_cccl.state = self._d_in_ptr_obj
        else:
            set_cccl_iterator_state(self.d_in_cccl, d_in)
        if self._d_out_is_ptr:
            self._d_out_ptr_obj.rebind(_fast_data_pointer(d_out), d_out)
            self.d_out_cccl.state = self._d_out_ptr_obj
        else:
            set_cccl_iterator_state(self.d_out_cccl, d_out)

        # Perf: validate_and_get_stream() re-derives the same int handle
        # every call for a fixed stream object -- its __cuda_stream__()
        # result can't change over that object's lifetime. Cache by
        # identity (held reference, not id()), same as the other caches on
        # this class; _UNSET_STREAM (not None) marks "never validated yet"
        # since stream=None is itself a valid, meaningful input.
        if stream is self._last_stream_obj:
            stream_handle = self._last_stream_handle
        else:
            stream_handle = validate_and_get_stream(stream)
            self._last_stream_obj = stream
            self._last_stream_handle = stream_handle

        if temp_storage is None:
            temp_storage_bytes = 0
            d_temp_storage = 0
        else:
            temp_storage_bytes = temp_storage.nbytes
            d_temp_storage = get_data_pointer(temp_storage)

        temp_storage_bytes = self.device_reduce_fn(
            d_temp_storage,
            temp_storage_bytes,
            self.d_in_cccl,
            self.d_out_cccl,
            num_items,
            self.op_cccl,
            self.h_init_cccl,
            stream_handle,
        )
        return temp_storage_bytes

    def __call__(
        self,
        *,
        temp_storage,
        d_in,
        d_out,
        num_items: int,
        op: Callable | OpAdapter,
        h_init: np.ndarray | GpuStruct | None,
        stream=None,
    ):
        # Select (and lazily load) the current device's build result, then bind the
        # derived compute fn from it.
        self.loaded_build_result = cccl.resolve_build_result(
            self.build_results, self._bound_build_result
        )
        self._bind_device_reduce_fn()

        set_cccl_iterator_state(self.d_in_cccl, d_in)
        set_cccl_iterator_state(self.d_out_cccl, d_out)

        # Update op state for stateful ops.
        #
        # Perf: make_op_adapter(op) is already cheap when `op` is already an
        # _OpAdapter (the common case: the isinstance check short-circuits,
        # no construction). What's skippable is re-deriving and re-assigning
        # .get_state(): for a *stateless* op that's a constant (b"" by
        # definition -- see _OpAdapter.get_state), so once we've seen this
        # exact op object with is_stateful=False, re-running that derivation
        # on a repeat call is pure waste. Stateful ops are NOT cached this
        # way: get_state() can legitimately return different bytes across
        # calls for the same object (e.g. an op capturing evolving runtime
        # state), so those always re-derive, unchanged from prior behavior.
        op_adapter = make_op_adapter(op)
        if op_adapter.is_stateful or op is not self._last_stateless_op:
            self.op_cccl.state = op_adapter.get_state()
            self._last_stateless_op = None if op_adapter.is_stateful else op

        if self.init_kind is _bindings.InitKind.VALUE_INIT:
            # We know that h_init_cccl is a Value here, so this cast tells MyPy
            # what the actual type is. cast() is a no-op at runtime, which makes
            # it better than isinstance() since this is a hot path and we have
            # to minimize the work we do prior to calling the kernel.
            self.h_init_cccl = cast(_bindings.Value, self.h_init_cccl)
            self.h_init_cccl.state = to_cccl_value_state(
                cast(np.ndarray | GpuStruct, h_init)
            )

        stream_handle = validate_and_get_stream(stream)

        if temp_storage is None:
            temp_storage_bytes = 0
            d_temp_storage = 0
        else:
            temp_storage_bytes = temp_storage.nbytes
            d_temp_storage = get_data_pointer(temp_storage)

        temp_storage_bytes = self.device_reduce_fn(
            d_temp_storage,
            temp_storage_bytes,
            self.d_in_cccl,
            self.d_out_cccl,
            num_items,
            self.op_cccl,
            self.h_init_cccl,
            stream_handle,
        )
        return temp_storage_bytes


@cache_with_registered_key_functions
def make_reduce_into(
    *,
    d_in: DeviceArrayLike | IteratorT,
    d_out: DeviceArrayLike | IteratorT,
    op: Operator,
    h_init: np.ndarray | GpuStruct | None = None,
    **kwargs,
):
    """Computes a device-wide reduction using the specified binary ``op`` and initial value ``init``.

    Example:
        Below, ``make_reduce_into`` is used to create a reduction object that can be reused.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/reduction/reduce_object.py
            :language: python
            :start-after: # example-begin


    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_out: Device array (of size 1) or iterator that will store the result of the reduction
        op: Binary operator to apply.
            The signature is ``(T, T) -> T``, where ``T`` is the data type of
            the initial value ``h_init`` (or of the input items when ``h_init``
            is None).
        h_init: Numpy array or GPU struct storing initial value of the
            reduction, or None for no initial value
        compute_capability: Compute capability, or list of capabilities, to
            build for ahead of time. Accepts a packed int (e.g. ``90``), a
            ``(major, minor)`` pair, a string (e.g. ``"9.0"``), or a list
            thereof. When ``None`` (the default), the current device's
            architecture is used.

    Returns:
        A callable object that can be used to perform the reduction
    """
    if h_init is not None:
        try:
            accum_dtype = get_dtype(h_init)
        except (AttributeError, TypeError) as e:
            raise TypeError(
                "Could not determine accumulator dtype from h_init; "
                "expected numpy array or object with .dtype"
            ) from e

        # Validate d_in and d_out if they are device arrays (iterators may not expose
        # dtype reliably here). Additionally, only require equality of dtypes for
        # struct objects; mixed scalar dtypes (e.g. int8 input with int64 output)
        # is acceptable
        if isinstance(h_init, _Struct):
            for arr, name in ((d_in, "input"), (d_out, "output")):
                if isinstance(arr, IteratorBase):
                    continue

                dtype = get_dtype(arr)
                if dtype != accum_dtype:
                    raise TypeError(
                        f"reduce_into dtype mismatch: {name} dtype {dtype} != "
                        f"accumulator dtype {accum_dtype}. "
                        f"Ensure {name} elements and h_init have identical dtype to "
                        "avoid truncation or misinterpretation."
                    )

    op_adapter = make_op_adapter(op)
    return _Reduce(
        d_in,
        d_out,
        op_adapter,
        h_init,
        kwargs.get("determinism", Determinism.RUN_TO_RUN),
        compute_capability=kwargs.get("compute_capability"),
    )


def reduce_into(
    *,
    d_in: DeviceArrayLike | IteratorT,
    d_out: DeviceArrayLike | IteratorT,
    num_items: int,
    op: Operator,
    h_init: np.ndarray | GpuStruct | None = None,
    stream=None,
    **kwargs,
):
    """
    Performs device-wide reduction.

    This function automatically handles temporary storage allocation and execution.

    Example:
        Below, ``reduce_into`` is used to compute the sum of a sequence of integers.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/reduction/sum_reduction.py
            :language: python
            :start-after: # example-begin

    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_out: Device array or iterator to store the result of the reduction
        num_items: Number of items to reduce
        op: Binary operator to apply.
            The signature is ``(T, T) -> T``, where ``T`` is the data type of
            the initial value ``h_init`` (or of the input items when ``h_init``
            is None).
        h_init: Initial value for the reduction, or None for no initial value.
            With None, the result is the reduction of just the input elements
            (equivalent to having the first element of ``d_in`` be the initial
            value). If ``num_items`` is also zero, ``d_out`` is unmodified.
        stream: CUDA stream for the operation (optional)
    """
    reducer = make_reduce_into(d_in=d_in, d_out=d_out, op=op, h_init=h_init, **kwargs)
    tmp_storage_bytes = reducer(
        temp_storage=None,
        d_in=d_in,
        d_out=d_out,
        num_items=num_items,
        op=op,
        h_init=h_init,
        stream=stream,
    )
    tmp_storage = TempStorageBuffer(tmp_storage_bytes, stream)
    reducer(
        temp_storage=tmp_storage,
        d_in=d_in,
        d_out=d_out,
        num_items=num_items,
        op=op,
        h_init=h_init,
        stream=stream,
    )
