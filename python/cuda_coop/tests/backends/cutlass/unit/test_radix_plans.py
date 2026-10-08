# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check radix plan validation, wrapper identity, and failed-call cleanup.

Sort bounds are wrapper arguments even when static. Direction and output
layout change the specialization. Rank wrappers transform signed keys to
ordered bits and may fill a separate prefix payload. If the external
wrapper call fails, the prefix stays unchanged and queued session state
must be restored from its snapshot.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest

cutlass = pytest.importorskip("cutlass")

from cuda.coop._core import (
    LaunchFacts,
    StorageOwnership,
    SynchronizationScope,
    this_block,
    this_warp,
)
from cuda.coop.cutlass import TempStorage, ThreadData, radix_rank_keys
from cuda.coop.cutlass._compiler import _rendering, _state, _storage
from cuda.coop.cutlass._lowering import _radix_sort

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


def _sort(**kwargs):
    options = {
        "group": this_block(),
        "launch": LaunchFacts(exact_block_dim=(8, 4, 2)),
        "key_type": cutlass.Int32,
        "value_type": None,
        "items": 3,
        "scalar": False,
        "begin_bit": 0,
        "end_bit": 32,
        "descending": False,
        "blocked_to_striped": False,
    }
    options.update(kwargs)
    return _radix_sort._CubRadixRequest(_radix_sort._sort_plan(**options))


def _rank(**kwargs):
    options = {
        "group": this_block(),
        "launch": LaunchFacts(exact_block_dim=(8, 4, 2)),
        "key_type": cutlass.Int32,
        "items": 3,
        "scalar": False,
        "begin_bit": 0,
        "end_bit": 4,
        "descending": False,
        "prefix_items": None,
    }
    options.update(kwargs)
    return _radix_sort._CubRadixRequest(_radix_sort._rank_plan(**options))


@pytest.mark.parametrize("dtype", tuple(_radix_sort._SORT_KEYS))
@pytest.mark.parametrize("pairs", (False, True))
def test_sort_plans(dtype, pairs):
    request = _sort(
        key_type=dtype, value_type=cutlass.Uint8 if pairs else None, end_bit=32
    )
    assert request.plan.result.values[0].dtype is dtype
    assert len(request.plan.result.values) == 1 + pairs
    assert request.implementation.struct_name == "CudaCoopBlockRadixSort"


@pytest.mark.parametrize(
    "begin,end", ((-1, 4), (4, 4), (7, 3), (0, 33), (1 << 32, 32))
)
def test_invalid_static_bits(begin, end):
    with pytest.raises(ValueError, match="bit"):
        _sort(begin_bit=begin, end_bit=end)


@pytest.mark.parametrize(
    "dtype", (cutlass.Int32, cutlass.Uint32, cutlass.Int64, cutlass.Uint64)
)
@pytest.mark.parametrize("descending", (False, True))
def test_rank_ordered_bits(dtype, descending):
    """Transform signed keys before extracting their radix digits.

    Only signed key types need the sign-bit flip. Checking each native-width
    mask in generated source exposes a missing or incorrectly sized transform.
    """

    request = _rank(key_type=dtype, descending=descending)
    source = _rendering.render_bundle_source([request])
    assert ("0x80000000u" in source) == (dtype is cutlass.Int32)
    assert ("0x8000000000000000ull" in source) == (dtype is cutlass.Int64)
    assert (
        request.implementation.template_arguments["SMEM_CONFIG"]
        == "cudaSharedMemBankSizeFourByte"
    )


def test_sort_runtime_identity():
    """Reuse one sort wrapper for different bit intervals.

    Bit bounds are provider operands, so they need no new symbol. Direction
    and blocked-to-striped output change the compiled operation and must use
    distinct symbols.
    """

    assert (
        _sort(begin_bit=0, end_bit=4).symbol_name
        == _sort(begin_bit=8, end_bit=12).symbol_name
    )
    assert _sort().symbol_name != _sort(descending=True).symbol_name
    assert _sort().symbol_name != _sort(blocked_to_striped=True).symbol_name


def test_prefix_identity_and_initialization():
    """Give prefix-output requests a separate wrapper and initialized slots.

    The generated wrapper fills its local prefix array with -1 before calling
    CUB, which writes only slots below the bin count. The extra output pointer
    changes the wrapper signature, so the request needs a distinct symbol.
    """

    request = _rank(prefix_items=1)
    assert request.symbol_name != _rank().symbol_name
    assert "prefix[i] = -1" in _rendering.render_bundle_source([request])


@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
@pytest.mark.parametrize("auto_sync", (False, True))
def test_storage_policy(sharing, auto_sync):
    request = _sort(
        temp_storage=TempStorage(
            alignment=128, sharing=sharing, auto_sync=auto_sync
        )
    )
    assert request.plan.temp_storage.ownership is StorageOwnership.CALLER
    assert request.plan.temp_storage.requested_alignment == 128
    assert request.plan.synchronization.storage_reuse_barrier is (
        SynchronizationScope.BLOCK if auto_sync else SynchronizationScope.NONE
    )


@pytest.mark.parametrize("factory", (_sort, _rank))
def test_physical_block_required(factory):
    with pytest.raises(
        (ValueError, NotImplementedError), match="physical block"
    ):
        factory(group=this_warp())


def test_result_dtype_validated():
    request = _rank()
    result = replace(
        request.plan.result,
        values=(replace(request.plan.result.values[0], dtype=cutlass.Uint32),),
    )
    with pytest.raises(ValueError, match="result dtypes"):
        _radix_sort._CubRadixRequest(replace(request.plan, result=result))


def test_prefix_alias_rejected_before_snapshot():
    keys = ThreadData(items_per_thread=1, dtype=cutlass.Int32, values=[2])
    with pytest.raises(ValueError, match="distinct"):
        radix_rank_keys(this_block(), keys, exclusive_digit_prefix=keys)


def test_failed_ffi_preserves_prefix_and_session(monkeypatch):
    """Keep prefix state and input keys unchanged when the wrapper call fails.

    Stubs replace request registration, value and pointer conversion, register
    tensors, and scratch registration. The stub foreign-function interface
    (FFI) call can then raise without an active CuTe kernel trace. The prefix
    starts with no dtype and a sentinel item; both must survive. The session
    restoration spy must receive the snapshot taken before the call.
    """

    request = _rank(prefix_items=1)
    keys = ThreadData(items_per_thread=3, dtype=cutlass.Int32, values=[3, 1, 2])
    prefix = ThreadData(items_per_thread=1, values=[-7])
    saved, restored = object(), []
    monkeypatch.setattr(_state, "snapshot_active_session_state", lambda: saved)
    monkeypatch.setattr(_state, "restore_active_session_state", restored.append)
    monkeypatch.setattr(_state, "register_request", lambda request: None)
    monkeypatch.setattr(_radix_sort, "_typed_item", lambda value, dtype: value)
    monkeypatch.setattr(
        _radix_sort,
        "_make_rmem_tensor",
        lambda *args: SimpleNamespace(iterator=SimpleNamespace(llvm_ptr=0)),
    )
    monkeypatch.setattr(
        _radix_sort,
        "llvm",
        SimpleNamespace(PointerType=SimpleNamespace(get=lambda space: 0)),
    )
    monkeypatch.setattr(
        _storage,
        "register_deferred_temp_storage_event",
        lambda *args, **kwargs: (0, 0, 0),
    )

    def fail(*args):
        raise RuntimeError("injected radix ffi failure")

    monkeypatch.setattr(_radix_sort, "ffi", lambda **kwargs: fail)
    with pytest.raises(RuntimeError, match="injected radix ffi failure"):
        _radix_sort._materialize(
            request, [keys], [(cutlass.Int32, (3, 1, 2))], prefix=prefix
        )
    assert restored == [saved]
    assert prefix.dtype is None
    assert prefix.values("radix_rank_keys") == (-7,)
    assert keys.values("radix_rank_keys") == (3, 1, 2)
