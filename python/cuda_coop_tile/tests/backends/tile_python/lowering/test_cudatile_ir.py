# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

from ....support.paths import PACKAGE_ROOT

SOURCE_ROOT = PACKAGE_ROOT


@pytest.fixture(autouse=True)
def _remove_new_coop_modules(monkeypatch):
    # This dependency makes cleanup run before monkeypatch restores old modules.
    include_path_env = "CUDA_COOP_TILE_PROVIDER_INCLUDE_PATHS"
    monkeypatch.setenv(include_path_env, os.environ.get(include_path_env, ""))
    existing_modules = set(sys.modules)
    yield
    for name in list(sys.modules):
        if name not in existing_modules and (
            name == "cuda.coop" or name.startswith("cuda.coop.")
        ):
            del sys.modules[name]


def _configure_tile_library(monkeypatch) -> Path:
    from cuda.coop.tile._cpp import _get_tile_library_linker_input

    tile_library = _get_tile_library_linker_input()[2].resolve()
    monkeypatch.setenv("CUDA_COOP_TILE_LIBRARY_PATH", str(tile_library))
    return tile_library


def _configure_imports(monkeypatch):
    # Keep the real compiler and its extension paired with their cuda parent.
    importlib.import_module("cuda.tile")

    monkeypatch.syspath_prepend(str(SOURCE_ROOT))
    monkeypatch.setattr(
        sys,
        "meta_path",
        [
            finder
            for finder in sys.meta_path
            if "cuda.coop.tile" not in getattr(finder, "known_source_files", {})
        ],
    )
    for module_name in list(sys.modules):
        if module_name == "cuda.coop" or module_name.startswith("cuda.coop."):
            monkeypatch.delitem(sys.modules, module_name)

    cuda_package = sys.modules["cuda"]
    monkeypatch.delitem(cuda_package.__dict__, "coop", raising=False)
    monkeypatch.setattr(
        cuda_package,
        "__path__",
        [str(SOURCE_ROOT / "cuda"), *cuda_package.__path__],
    )


class _FakeCudaArray:
    def __init__(self, shape, typestr="<i4"):
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": typestr,
            "data": (0x100000, False),
            "version": 3,
        }


def _get_compile_result(
    pyfunc,
    args,
    tile_context,
    *,
    bytecode_version=None,
    return_bytecode=False,
    return_cubin=False,
    sm_arch: str | None = None,
):
    """Compile a kernel with final IR available for linkage assertions."""
    from cuda.tile import _debug
    from cuda.tile._cext import CallingConvention
    from cuda.tile._compile import compile_tile
    from cuda.tile._execution import kernel
    from cuda.tile.compilation._signature import KernelSignature

    dispatcher = kernel(pyfunc)
    calling_convention = (
        CallingConvention.cutile_python_v2()
        if getattr(_debug, "STATIC_SHAPE_STRIDES", False)
        else CallingConvention.cutile_python_v1()
    )
    signature = KernelSignature.from_kernel_args(
        dispatcher,
        args,
        calling_convention,
    )
    result = compile_tile(
        dispatcher._annotated_function,
        (signature,),
        sm_arch or os.environ.get("CUDA_COOP_TILE_E2E_SM_ARCH", "sm_120"),
        dispatcher._compiler_options,
        tile_context,
        bytecode_version=bytecode_version,
        return_final_ir=True,
        return_bytecode=return_bytecode,
        return_cubin=return_cubin,
    )
    assert result.final_ir is not None
    return result


def _get_final_ir(
    pyfunc,
    args,
    tile_context,
    *,
    return_cubin=False,
    sm_arch: str | None = None,
):
    """Build final IR, optionally serializing and linking TileIR."""
    from cuda.tile._bytecode.version import BytecodeVersion

    result = _get_compile_result(
        pyfunc,
        args,
        tile_context,
        bytecode_version=BytecodeVersion.V_13_5,
        return_cubin=return_cubin,
        sm_arch=sm_arch,
    )
    return result.final_ir[0]


def _preview_foreign_calls(func_ir):
    from cuda.tile._ir.type import TokenTy
    from cuda.tile_preview import PreviewForeignCall

    calls = [
        op for op in func_ir.traverse() if isinstance(op, PreviewForeignCall)
    ]
    for op in calls:
        assert op.token is not None
        assert isinstance(op.result_vars[-1].get_type(), TokenTy)
    return calls


def _constant_args(op, names):
    return dict(zip(names, op.constant_params, strict=True))


@pytest.mark.link
def test_cooperative_call_orders_surrounding_memory(monkeypatch):
    # Foreign calls order all memory chains, including arrays that
    # are not operands of the tile-only cooperative primitive.
    _configure_tile_library(monkeypatch)
    _configure_imports(monkeypatch)

    global ct, coop
    ct = importlib.import_module("cuda.tile")
    coop = importlib.import_module("cuda.coop.tile")
    from cuda.tile._bytecode.version import BytecodeVersion
    from cuda.tile._cext import default_tile_context
    from cuda.tile._ir.ops import JoinTokens, TileStore

    def ordered_kernel(x, y):
        values = ct.full((128,), 1, ct.int32)
        ct.store(x, (0,), values)
        ct.store(y, (0,), values)
        scanned = coop.exclusive_sum(values)
        ct.store(x, (0,), scanned)
        ct.store(y, (0,), scanned)

    result = _get_compile_result(
        ordered_kernel,
        (_FakeCudaArray((128,)), _FakeCudaArray((128,))),
        default_tile_context,
        bytecode_version=BytecodeVersion.V_13_5,
        return_bytecode=True,
    )
    func_ir = result.final_ir[0]
    (call,) = _preview_foreign_calls(func_ir)
    assert len(call.result_vars) == 2
    stores = [op for op in func_ir.operations if isinstance(op, TileStore)]
    assert len(stores) == 4
    joins = {
        op.result_var: op.tokens
        for op in func_ir.operations
        if isinstance(op, JoinTokens)
    }
    assert stores[0].result_var in joins[call.token]
    assert stores[1].result_var in joins[call.token]
    for store in stores[2:]:
        assert call.result_vars[-1] in joins[store.token]
    assert result.bytecode


@pytest.mark.link
def test_sum_symbolic_schema_spans_dtypes_and_extents(monkeypatch):
    tile_library_path = _configure_tile_library(monkeypatch)
    _configure_imports(monkeypatch)

    global ct, coop
    ct = importlib.import_module("cuda.tile")
    coop = importlib.import_module("cuda.coop.tile")
    from cuda.tile._cext import default_tile_context

    @ct.kernel
    def mixed_sum_kernel(
        i32_small_in: ct.Array,
        i32_large_in: ct.Array,
        f32_small_in: ct.Array,
        f32_large_in: ct.Array,
        i32_small_out: ct.Array,
        i32_large_out: ct.Array,
        f32_small_out: ct.Array,
        f32_large_out: ct.Array,
    ):
        i32_small = ct.load(i32_small_in, (0,), shape=(128,))
        i32_large = ct.load(i32_large_in, (0,), shape=(256,))
        f32_small = ct.load(f32_small_in, (0,), shape=(128,))
        f32_large = ct.load(f32_large_in, (0,), shape=(256,))
        ct.store(
            i32_small_out,
            (0,),
            coop.reduce(i32_small, binary_op="sum", broadcast=True),
        )
        ct.store(i32_large_out, (0,), coop.sum(i32_large, broadcast=True))
        ct.store(f32_small_out, (0,), coop.sum(f32_small, broadcast=True))
        ct.store(f32_large_out, (0,), coop.sum(f32_large, broadcast=True))

    func_ir = _get_final_ir(
        mixed_sum_kernel._pyfunc,
        (
            _FakeCudaArray((128,), typestr="<i4"),
            _FakeCudaArray((256,), typestr="<i4"),
            _FakeCudaArray((128,), typestr="<f4"),
            _FakeCudaArray((256,), typestr="<f4"),
            _FakeCudaArray((128,), typestr="<i4"),
            _FakeCudaArray((256,), typestr="<i4"),
            _FakeCudaArray((128,), typestr="<f4"),
            _FakeCudaArray((256,), typestr="<f4"),
        ),
        default_tile_context,
    )
    foreign_ops = _preview_foreign_calls(func_ir)
    assert len(foreign_ops) == 4
    assert {op.symbol_name for op in foreign_ops} == {
        "cuda_coop_tile_sum_batched"
    }
    for op in foreign_ops:
        assert len(op.tilelibs) == 1
        assert op.tilelibs[0][1]
        assert op.tilelibs[0][0] == tile_library_path
        assert len(op.inputs) == 2
        assert len(op.result_vars) == 2
        assert op.inputs[1].get_type().shape == (1,)
        assert op.inputs[1].get_type() == op.result_vars[0].get_type()
        assert (
            op.inputs[0].get_type().dtype == op.result_vars[0].get_type().dtype
        )
        assert op.constant_params[-1] == 1
    assert {
        (
            op.constant_params[0],
            op.constant_params[1].name,
        )
        for op in foreign_ops
    } == {
        (128, "int32"),
        (256, "int32"),
        (128, "float32"),
        (256, "float32"),
    }


@pytest.mark.link
@pytest.mark.parametrize(
    "axis,diagnostic",
    [
        (2, "out of range"),
        (-3, "out of range"),
        ((1, -1), "Repeated reduction axis"),
        (1.5, "integer"),
        ((0, "last"), "integers"),
    ],
)
def test_sum_rejects_invalid_axes(monkeypatch, axis, diagnostic):
    _configure_tile_library(monkeypatch)
    _configure_imports(monkeypatch)
    global ct, coop
    ct = importlib.import_module("cuda.tile")
    coop = importlib.import_module("cuda.coop.tile")
    from cuda.tile._cext import default_tile_context
    from cuda.tile._exception import TileTypeError

    @ct.kernel
    def kernel(source: ct.Array, destination: ct.Array):
        values = ct.load(source, (0, 0), (8, 128))
        result = coop.sum(values, axis=axis)
        ct.store(destination, (0,), ct.reshape(result, (-1,)))

    with pytest.raises(TileTypeError, match=diagnostic):
        _get_final_ir(
            kernel._pyfunc,
            (_FakeCudaArray((8, 128)), _FakeCudaArray((8,))),
            default_tile_context,
        )


@pytest.mark.link
def test_radix_sort_keys_static_options_reach_foreign_call_const_args(
    monkeypatch,
):
    tile_library_path = _configure_tile_library(monkeypatch)
    _configure_imports(monkeypatch)

    global ct, coop
    ct = importlib.import_module("cuda.tile")
    coop = importlib.import_module("cuda.coop.tile")
    from cuda.tile._cext import default_tile_context

    @ct.kernel
    def radix_kernel(
        inp: ct.Array,
        begin_bit_in: ct.Array,
        end_bit_in: ct.Array,
        out: ct.Array,
    ):
        keys = ct.load(inp, (0,), shape=(128,))
        begin_bit = ct.load(begin_bit_in, (0,), shape=(1,))
        end_bit = ct.load(end_bit_in, (0,), shape=(1,))
        sorted_keys = coop.radix_sort_keys(
            keys,
            descending=True,
            begin_bit=begin_bit,
            end_bit=end_bit,
            radix_bits=4,
            memoize_outer_scan=True,
            inner_scan_algorithm="warp_scans",
            smem_config="four_byte",
        )
        ct.store(out, (0,), sorted_keys)

    func_ir = _get_final_ir(
        radix_kernel._pyfunc,
        (
            _FakeCudaArray((128,)),
            _FakeCudaArray((1,)),
            _FakeCudaArray((1,)),
            _FakeCudaArray((128,)),
        ),
        default_tile_context,
    )
    foreign_ops = _preview_foreign_calls(func_ir)
    assert len(foreign_ops) == 1

    op = foreign_ops[0]
    assert op.tilelibs[0][0] == tile_library_path
    assert op.symbol_name == "cuda_coop_tile_radix_sort_keys"
    constant_args = _constant_args(
        op,
        (
            "N",
            "T",
            "Descending",
            "RadixBits",
            "MemoizeOuterScan",
            "InnerScanAlgorithm",
            "SmemConfig",
            "DTypeToken",
        ),
    )
    assert constant_args["Descending"] == 1
    assert constant_args["RadixBits"] == 4
    assert constant_args["MemoizeOuterScan"] == 1
    assert constant_args["InnerScanAlgorithm"] == "warp_scans"
    assert constant_args["SmemConfig"] == "four_byte"
    assert constant_args["DTypeToken"] == "i32"

    assert len(op.inputs) == 4
    assert op.inputs[1].get_type().shape == ()
    assert op.inputs[2].get_type().shape == ()


@pytest.mark.link
def test_radix_sort_pairs_static_options_reach_foreign_call_const_args(
    monkeypatch,
):
    tile_library_path = _configure_tile_library(monkeypatch)
    _configure_imports(monkeypatch)

    global ct, coop
    ct = importlib.import_module("cuda.tile")
    coop = importlib.import_module("cuda.coop.tile")
    from cuda.tile._cext import default_tile_context

    @ct.kernel
    def radix_pairs_kernel(
        keys_in: ct.Array,
        vals_in: ct.Array,
        keys_out: ct.Array,
        vals_out: ct.Array,
    ):
        keys = ct.load(keys_in, (0,), shape=(128,))
        values = ct.load(vals_in, (0,), shape=(128,))
        sorted_keys, sorted_values = coop.radix_sort_pairs(
            keys,
            values,
            descending=True,
            begin_bit=0,
            end_bit=4,
            radix_bits=4,
            memoize_outer_scan=False,
            inner_scan_algorithm="raking",
            smem_config="eight_byte",
        )
        ct.store(keys_out, (0,), sorted_keys)
        ct.store(vals_out, (0,), sorted_values)

    func_ir = _get_final_ir(
        radix_pairs_kernel._pyfunc,
        (
            _FakeCudaArray((128,), typestr="<u4"),
            _FakeCudaArray((128,), typestr="<i4"),
            _FakeCudaArray((128,), typestr="<u4"),
            _FakeCudaArray((128,), typestr="<i4"),
        ),
        default_tile_context,
    )
    foreign_ops = _preview_foreign_calls(func_ir)
    assert len(foreign_ops) == 1

    op = foreign_ops[0]
    assert op.tilelibs[0][0] == tile_library_path
    assert op.symbol_name == "cuda_coop_tile_radix_sort_pairs"
    constant_args = _constant_args(
        op,
        (
            "N",
            "KeyT",
            "ValueT",
            "Descending",
            "RadixBits",
            "MemoizeOuterScan",
            "InnerScanAlgorithm",
            "SmemConfig",
            "KeyDTypeToken",
            "ValueDTypeToken",
        ),
    )
    assert constant_args["Descending"] == 1
    assert constant_args["RadixBits"] == 4
    assert constant_args["MemoizeOuterScan"] == 0
    assert constant_args["InnerScanAlgorithm"] == "raking"
    assert constant_args["SmemConfig"] == "eight_byte"
    assert constant_args["KeyDTypeToken"] == "u32"
    assert constant_args["ValueDTypeToken"] == "i32"

    assert len(op.inputs) == 6
    assert op.inputs[2].get_type().shape == ()
    assert op.inputs[3].get_type().shape == ()


@pytest.mark.link
@pytest.mark.parametrize("second_width", [128, 256])
@pytest.mark.parametrize("reverse", [False, True])
def test_registered_exchange_inverse_fold_requires_same_width(
    monkeypatch, second_width, reverse
):
    _configure_tile_library(monkeypatch)
    _configure_imports(monkeypatch)
    global ct, coop
    ct = importlib.import_module("cuda.tile")
    coop = importlib.import_module("cuda.coop.tile")
    from cuda.tile._cext import default_tile_context

    first = (
        coop.exchange_blocked_to_striped
        if reverse
        else coop.exchange_striped_to_blocked
    )
    second = (
        coop.exchange_striped_to_blocked
        if reverse
        else coop.exchange_blocked_to_striped
    )

    @ct.kernel
    def kernel(inp: ct.Array, out: ct.Array):
        values = ct.load(inp, (0,), shape=(512,))
        exchanged = first(values, block_threads=128)
        result = second(exchanged, block_threads=second_width)
        ct.store(out, (0,), result)

    func_ir = _get_final_ir(
        kernel._pyfunc,
        (_FakeCudaArray((512,)), _FakeCudaArray((512,))),
        default_tile_context,
    )
    calls = _preview_foreign_calls(func_ir)
    assert len(calls) == (1 if second_width == 128 else 2)
    assert all(call.symbol_name == "cuda_coop_tile_exchange" for call in calls)
    if second_width != 128:
        assert [call.constant_params[2] for call in calls] == [128, 256]
        assert calls[1].inputs[0] is calls[0].result_vars[0]


@pytest.mark.link
def test_registered_exchange_does_not_fold_another_foreign_callee(monkeypatch):
    tile_library = _configure_tile_library(monkeypatch)
    _configure_imports(monkeypatch)
    global ct, coop, preview
    ct = importlib.import_module("cuda.tile")
    coop = importlib.import_module("cuda.coop.tile")
    preview = importlib.import_module("cuda.tile_preview")
    from cuda.tile._cext import default_tile_context

    library = preview.Tilelib(path=str(tile_library), version="callee-test")

    @ct.kernel
    def kernel(inp: ct.Array, out: ct.Array):
        values = ct.load(inp, (0,), shape=(512,))
        (transformed,) = preview.foreign_call(
            tilelibs=(library,),
            symbol_name="unrelated_permutation",
            constant_params=(512, ct.int32, 128, "blocked_to_striped", "i32"),
            inputs=(values,),
            output_types=((ct.int32, (512,)),),
        )
        result = coop.exchange_blocked_to_striped(
            transformed, block_threads=128
        )
        ct.store(out, (0,), result)

    func_ir = _get_final_ir(
        kernel._pyfunc,
        (_FakeCudaArray((512,)), _FakeCudaArray((512,))),
        default_tile_context,
    )
    calls = _preview_foreign_calls(func_ir)
    assert [call.symbol_name for call in calls] == [
        "unrelated_permutation",
        "cuda_coop_tile_exchange",
    ]
    assert calls[1].inputs[0] is calls[0].result_vars[0]


@pytest.mark.link
@pytest.mark.parametrize("pairs", [False, True])
@pytest.mark.parametrize("valid_items", [128, 129])
def test_merge_sort_checks_constant_valid_item_extent(
    monkeypatch, pairs, valid_items
):
    _configure_tile_library(monkeypatch)
    _configure_imports(monkeypatch)
    global ct, coop
    ct = importlib.import_module("cuda.tile")
    coop = importlib.import_module("cuda.coop.tile")
    from cuda.tile._cext import default_tile_context
    from cuda.tile._exception import TileTypeError

    @ct.kernel
    def kernel(inp: ct.Array, out: ct.Array):
        keys = ct.load(inp, (0,), shape=(128,))
        if pairs:
            result, _ = coop.merge_sort_pairs(
                keys, keys, valid_items=valid_items, oob_default=0
            )
        else:
            result = coop.merge_sort_keys(
                keys, valid_items=valid_items, oob_default=0
            )
        ct.store(out, (0,), result)

    arguments = (_FakeCudaArray((128,)), _FakeCudaArray((128,)))
    if valid_items > 128:
        with pytest.raises(
            TileTypeError, match="valid_items exceeds tile size"
        ):
            _get_final_ir(kernel._pyfunc, arguments, default_tile_context)
    else:
        func_ir = _get_final_ir(kernel._pyfunc, arguments, default_tile_context)
        assert len(_preview_foreign_calls(func_ir)) == 1


@pytest.mark.link
@pytest.mark.parametrize("pairs", [False, True])
@pytest.mark.parametrize("use_methods", [False, True])
def test_merge_sort_rejects_boolean_valid_items(
    monkeypatch, pairs, use_methods
):
    _configure_tile_library(monkeypatch)
    _configure_imports(monkeypatch)
    global ct, coop
    ct = importlib.import_module("cuda.tile")
    coop = importlib.import_module("cuda.coop.tile")
    from cuda.tile._cext import default_tile_context
    from cuda.tile._exception import TileTypeError

    for count in (True, 1):
        valid_items = None if use_methods else count
        oob_default = None if use_methods else 999
        methods = (
            ("valid_items", count, "oob_default", 999) if use_methods else None
        )

        @ct.kernel
        def kernel(inp: ct.Array, out: ct.Array):
            keys = ct.load(inp, (0,), shape=(128,))
            if pairs:
                result, _ = coop.merge_sort_pairs(
                    keys,
                    keys,
                    valid_items=valid_items,
                    oob_default=oob_default,
                    methods=methods,
                )
            else:
                result = coop.merge_sort_keys(
                    keys,
                    valid_items=valid_items,
                    oob_default=oob_default,
                    methods=methods,
                )
            ct.store(out, (0,), result)

        arguments = (_FakeCudaArray((128,)), _FakeCudaArray((128,)))
        if count is True:
            with pytest.raises(
                TileTypeError, match="valid_items must be an int"
            ):
                _get_final_ir(kernel._pyfunc, arguments, default_tile_context)
        else:
            func_ir = _get_final_ir(
                kernel._pyfunc, arguments, default_tile_context
            )
            assert len(_preview_foreign_calls(func_ir)) == 1


@pytest.mark.link
@pytest.mark.parametrize("primitive", ["keys", "pairs", "rank"])
def test_radix_rejects_extent_beyond_cub_rank_capacity(monkeypatch, primitive):
    _configure_tile_library(monkeypatch)
    _configure_imports(monkeypatch)
    global ct, coop
    ct = importlib.import_module("cuda.tile")
    coop = importlib.import_module("cuda.coop.tile")
    from cuda.tile._cext import default_tile_context
    from cuda.tile._exception import TileTypeError

    @ct.kernel
    def kernel(inp: ct.Array, out: ct.Array):
        keys = ct.load(inp, (0,), shape=(65536,))
        if primitive == "keys":
            result = coop.radix_sort_keys(keys)
        elif primitive == "pairs":
            result, _ = coop.radix_sort_pairs(keys, keys)
        else:
            result = coop.radix_rank_keys(keys)
        ct.store(out, (0,), result)

    with pytest.raises(TileTypeError, match="tile size must not exceed 65535"):
        _get_final_ir(
            kernel._pyfunc,
            (_FakeCudaArray((65536,)), _FakeCudaArray((65536,))),
            default_tile_context,
        )


@pytest.mark.link
@pytest.mark.parametrize("pairs", [False, True])
@pytest.mark.parametrize("use_methods", [False, True])
@pytest.mark.parametrize("control", ["begin_bit", "end_bit", "radix_bits"])
def test_radix_sort_rejects_boolean_integer_controls(
    monkeypatch, pairs, use_methods, control
):
    _configure_tile_library(monkeypatch)
    _configure_imports(monkeypatch)
    global ct, coop
    ct = importlib.import_module("cuda.tile")
    coop = importlib.import_module("cuda.coop.tile")
    from cuda.tile._cext import default_tile_context
    from cuda.tile._exception import TileTypeError

    for value in (True, 1):
        begin_bit = (
            value if control == "begin_bit" and not use_methods else None
        )
        end_bit = value if control == "end_bit" and not use_methods else None
        radix_bits = (
            value if control == "radix_bits" and not use_methods else None
        )
        methods = (control, value) if use_methods else None

        @ct.kernel
        def kernel(inp: ct.Array, out: ct.Array):
            keys = ct.load(inp, (0,), shape=(128,))
            if pairs:
                result, _ = coop.radix_sort_pairs(
                    keys,
                    keys,
                    begin_bit=begin_bit,
                    end_bit=end_bit,
                    radix_bits=radix_bits,
                    methods=methods,
                )
            else:
                result = coop.radix_sort_keys(
                    keys,
                    begin_bit=begin_bit,
                    end_bit=end_bit,
                    radix_bits=radix_bits,
                    methods=methods,
                )
            ct.store(out, (0,), result)

        arguments = (_FakeCudaArray((128,)), _FakeCudaArray((128,)))
        if value is True:
            with pytest.raises(
                TileTypeError, match=f"{control} must be an int"
            ):
                _get_final_ir(kernel._pyfunc, arguments, default_tile_context)
        else:
            func_ir = _get_final_ir(
                kernel._pyfunc, arguments, default_tile_context
            )
            assert len(_preview_foreign_calls(func_ir)) == 1


@pytest.mark.link
@pytest.mark.parametrize(
    "name", ["exchange_striped_to_blocked", "exchange_blocked_to_striped"]
)
def test_exchange_dtype_diagnostic_names_the_public_operation(
    monkeypatch, name
):
    _configure_tile_library(monkeypatch)
    _configure_imports(monkeypatch)
    global ct, coop
    ct = importlib.import_module("cuda.tile")
    coop = importlib.import_module("cuda.coop.tile")
    from cuda.tile._cext import default_tile_context
    from cuda.tile._exception import TileTypeError

    operation = getattr(coop, name)

    @ct.kernel
    def kernel(inp: ct.Array, out: ct.Array):
        flags = ct.load(inp, (0,), shape=(128,))
        result = operation(flags, block_threads=128)
        ct.store(out, (0,), result)

    with pytest.raises(
        TileTypeError, match=rf"coop\.{name} does not support dtype bool_"
    ):
        _get_final_ir(
            kernel._pyfunc,
            (_FakeCudaArray((128,), "|b1"), _FakeCudaArray((128,), "|b1")),
            default_tile_context,
        )
