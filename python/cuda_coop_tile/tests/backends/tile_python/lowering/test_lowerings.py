# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib
import importlib.machinery
import os
import sys
import types
from pathlib import Path

import pytest

from ....support.paths import PACKAGE_ROOT

SOURCE_ROOT = PACKAGE_ROOT


@pytest.fixture(autouse=True)
def _restore_cuda_modules():
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "cuda" or name.startswith("cuda.")
    }
    yield
    # Imports made with stub dependencies must not escape into later tests.
    for name in list(sys.modules):
        if name == "cuda" or name.startswith("cuda."):
            del sys.modules[name]
    sys.modules.update(saved)


def _stub_package(name: str, path: Path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = module.__path__
    module.__spec__ = spec
    return module


class _Const:
    def __init__(self, value, ty=None):
        self._value = value
        self.tile_type = ty

    def is_constant(self):
        return True

    def get_constant(self):
        return self._value

    def get_type(self):
        return self.tile_type


class _TileType:
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape
        self.ndim = len(shape)
        numel = 1
        for dim in shape:
            numel *= dim
        self.numel = numel


class _Var:
    def __init__(self, name, tile_type, ctx=None):
        self.name = name
        self.tile_type = tile_type
        self.ctx = ctx or types.SimpleNamespace(producers={})

    def is_constant(self):
        return False

    def get_type(self):
        return self.tile_type


class _DType:
    def __init__(self, name, bitwidth):
        self.name = name
        self.bitwidth = bitwidth

    def __repr__(self):
        return self.name


class _TokenType:
    pass


class _PreviewForeignCall:
    def __init__(
        self,
        *,
        tilelibs,
        symbol_name,
        constant_params,
        inputs,
        token,
    ):
        self.tilelibs = tilelibs
        self.symbol_name = symbol_name
        self.constant_params = constant_params
        self.inputs = inputs
        self.token = token


class _Assign:
    pass


class _TileAnnotation:
    def __class_getitem__(cls, item):
        return item


def _load_lowerings_with_cuda_tile_stubs(monkeypatch, tile_library_path=None):
    include_path_env = "CUDA_COOP_TILE_PROVIDER_INCLUDE_PATHS"
    monkeypatch.setenv(include_path_env, os.environ.get(include_path_env, ""))
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
    if tile_library_path is None:
        monkeypatch.delenv("CUDA_COOP_TILE_LIBRARY_PATH", raising=False)
        monkeypatch.delenv("CUDA_COOP_TILE_LIBRARY_VERSION", raising=False)
    else:
        monkeypatch.setenv(
            "CUDA_COOP_TILE_LIBRARY_PATH", str(tile_library_path)
        )
        monkeypatch.setenv("CUDA_COOP_TILE_LIBRARY_VERSION", "test-tilelib-v1")
    for loaded_module in list(sys.modules):
        if loaded_module == "cuda" or loaded_module.startswith(
            ("cuda.coop", "cuda.tile")
        ):
            monkeypatch.delitem(sys.modules, loaded_module)

    cuda_pkg = _stub_package("cuda", SOURCE_ROOT / "cuda")
    cuda_coop = _stub_package("cuda.coop", SOURCE_ROOT / "cuda" / "coop")
    cuda_coop_tile = _stub_package(
        "cuda.coop.tile", SOURCE_ROOT / "cuda" / "coop" / "tile"
    )

    cuda_tile = types.ModuleType("cuda.tile")
    cuda_tile.__path__ = []
    tile_preview = types.ModuleType("cuda.tile_preview")
    tile_preview.PreviewForeignCall = _PreviewForeignCall
    cuda_pkg.coop = cuda_coop
    cuda_pkg.tile = cuda_tile
    cuda_pkg.tile_preview = tile_preview
    cuda_coop.tile = cuda_coop_tile

    datatype = types.ModuleType("cuda.tile._datatype")
    datatype.DType = _DType
    dtype_bits = {
        "bool_": 1,
        "bfloat16": 16,
        "float16": 16,
        "float32": 32,
        "float64": 64,
        "int8": 8,
        "uint8": 8,
        "int16": 16,
        "uint16": 16,
        "int32": 32,
        "uint32": 32,
        "int64": 64,
        "uint64": 64,
    }
    for name, bitwidth in dtype_bits.items():
        setattr(datatype, name, _DType(name, bitwidth))

    exception = types.ModuleType("cuda.tile._exception")

    class TileTypeError(TypeError):
        pass

    exception.TileTypeError = TileTypeError

    stub = types.ModuleType("cuda.tile._stub")
    stub.Tile = _TileAnnotation

    ir = types.ModuleType("cuda.tile._ir.ir")
    records = []
    ctx = types.SimpleNamespace(producers={})

    class Builder:
        @staticmethod
        def get_current():
            return types.SimpleNamespace(ops=[])

    def add_operation(op_cls, output_type, **kwargs):
        records.append((op_cls, output_type, kwargs))
        if isinstance(output_type, tuple):
            result_vars = tuple(
                _Var(f"out{i}", ty, ctx) for i, ty in enumerate(output_type)
            )
        else:
            result_vars = (_Var(f"out{len(records) - 1}", output_type, ctx),)
        op = op_cls(**kwargs)
        op.result_vars = result_vars
        op.inputs = kwargs.get("inputs")
        op.static_params = kwargs.get("static_params")
        op.tilelibs = kwargs.get("tilelibs")
        op.symbol_name = kwargs.get("symbol_name")
        op.constant_params = kwargs.get("constant_params")
        op.token = kwargs.get("token")
        for var in result_vars:
            ctx.producers[var.name] = op
        if isinstance(output_type, tuple):
            return result_vars
        return result_vars[0]

    ir.Builder = Builder
    ir.add_operation = add_operation
    ir.add_operation_variadic = add_operation

    arithmetic_ops = types.ModuleType("cuda.tile._ir.arithmetic_ops")

    def binary_arithmetic_tensorlike(fn, lhs, rhs):
        del rhs
        return _Var(
            f"{fn}_{getattr(lhs, 'name', 'constant')}", lhs.get_type(), ctx
        )

    arithmetic_ops.binary_arithmetic_tensorlike = binary_arithmetic_tensorlike

    op_impl = types.ModuleType("cuda.tile._ir.op_impl")

    def require_optional_constant_str(var):
        if var.is_constant():
            return var.get_constant()
        raise TileTypeError("value must be a compile-time constant")

    def require_constant_bool(var):
        if not var.is_constant() or type(var.get_constant()) is not bool:
            raise TileTypeError("expected a boolean constant")
        return var.get_constant()

    def require_constant_int_tuple(var, allow_single_int=False):
        if not var.is_constant():
            raise TileTypeError("expected a constant integer tuple")
        value = var.get_constant()
        if allow_single_int and type(value) is int:
            return (value,)
        if not isinstance(value, tuple) or any(
            type(x) is not int for x in value
        ):
            raise TileTypeError("expected a constant integer tuple")
        return value

    def normalize_axis(axis, ndim, var=None):
        original = axis
        if axis < 0:
            axis += ndim
        if not 0 <= axis < ndim:
            raise TileTypeError(
                f"Axis {original} is out of range for rank {ndim}"
            )
        return axis

    op_impl.require_constant_bool = require_constant_bool
    op_impl.require_constant_int_tuple = require_constant_int_tuple
    op_impl.normalize_axis = normalize_axis
    op_impl.require_optional_constant_str = require_optional_constant_str
    op_impl.require_tile_type = lambda var: var.tile_type

    core_ops = types.ModuleType("cuda.tile._ir.core_ops")
    core_ops.Assign = _Assign
    ops = types.ModuleType("cuda.tile._ir.ops")
    ops.build_tuple = lambda outputs: ("tuple", outputs)

    def strictly_typed_const(value, ty):
        return _Const(value, ty)

    def _const(shape, value, dtype):
        return _Const(value, _TileType(dtype, tuple(shape)))

    def astype(var, dtype):
        if var.is_constant():
            return _Const(var.get_constant(), dtype)
        ty = var.get_type()
        if hasattr(ty, "shape"):
            return _Var(f"{var.name}_astype", _TileType(dtype, ty.shape), ctx)
        return _Var(f"{var.name}_astype", dtype, ctx)

    def full(shape, fill_value, dtype):
        if fill_value.is_constant():
            return _Const(
                fill_value.get_constant(), _TileType(dtype, tuple(shape))
            )
        return _Var(
            f"{fill_value.name}_full", _TileType(dtype, tuple(shape)), ctx
        )

    def reshape(var, shape):
        ty = var.get_type()
        return _Var(
            f"{getattr(var, 'name', 'const')}_reshape",
            _TileType(ty.dtype, tuple(shape)),
            ctx,
        )

    ops.astype = astype
    ops._const = _const
    ops.full = full
    ops.reshape = reshape
    ops.broadcast_to = reshape
    ops.permute = lambda var, order: reshape(
        var, tuple(var.get_type().shape[i] for i in order)
    )
    ops.strictly_typed_const = strictly_typed_const

    types_mod = types.ModuleType("cuda.tile._ir.type")
    types_mod.TileTy = _TileType
    types_mod.TokenTy = _TokenType

    pathfinder = types.ModuleType("cuda.pathfinder")
    pathfinder.find_nvidia_header_directory = lambda library: None
    monkeypatch.setitem(sys.modules, "cuda.pathfinder", pathfinder)
    monkeypatch.setitem(sys.modules, "cuda", cuda_pkg)
    monkeypatch.setitem(sys.modules, "cuda.coop", cuda_coop)
    monkeypatch.setitem(sys.modules, "cuda.coop.tile", cuda_coop_tile)
    monkeypatch.setitem(sys.modules, "cuda.tile", cuda_tile)
    monkeypatch.setitem(sys.modules, "cuda.tile_preview", tile_preview)
    monkeypatch.setitem(sys.modules, "cuda.tile._datatype", datatype)
    monkeypatch.setitem(sys.modules, "cuda.tile._exception", exception)
    monkeypatch.setitem(sys.modules, "cuda.tile._stub", stub)
    monkeypatch.setitem(sys.modules, "cuda.tile._ir.ir", ir)
    monkeypatch.setitem(
        sys.modules, "cuda.tile._ir.arithmetic_ops", arithmetic_ops
    )
    monkeypatch.setitem(sys.modules, "cuda.tile._ir.op_impl", op_impl)
    monkeypatch.setitem(sys.modules, "cuda.tile._ir.ops", ops)
    monkeypatch.setitem(sys.modules, "cuda.tile._ir.core_ops", core_ops)
    monkeypatch.setitem(sys.modules, "cuda.tile._ir.type", types_mod)
    lowerings = importlib.import_module("cuda.coop.tile._lowerings")
    return lowerings, datatype, records


def _make_provider_backed_lowerings(monkeypatch, tmp_path):
    tile_library_path = tmp_path / "cuda_coop_tile.tilelib"
    tile_library_path.touch()
    return _load_lowerings_with_cuda_tile_stubs(monkeypatch, tile_library_path)


def _record_static_params(record):
    return record[2]["constant_params"]


_CONSTANT_PARAM_NAMES = {
    "cuda_coop_tile_sum_batched": (
        "N",
        "T",
        "Algorithm",
        "DTypeToken",
        "Batches",
    ),
    "cuda_coop_tile_scan": (
        "N",
        "T",
        "Mode",
        "Algorithm",
        "ScanOp",
        "InitialPresent",
        "PrefixPresent",
        "DTypeToken",
    ),
    "cuda_coop_tile_histogram": (
        "N",
        "Bins",
        "DTypeToken",
    ),
    "cuda_coop_tile_histogram_256": (
        "N",
        "DTypeToken",
    ),
    "cuda_coop_tile_discontinuity_flags": (
        "N",
        "T",
        "Mode",
        "DTypeToken",
    ),
    "cuda_coop_tile_discontinuity_heads_and_tails": (
        "N",
        "T",
        "Mode",
        "DTypeToken",
    ),
    "cuda_coop_tile_adjacent_difference": (
        "N",
        "T",
        "Mode",
        "DTypeToken",
    ),
    "cuda_coop_tile_exchange": (
        "N",
        "T",
        "BlockThreads",
        "Mode",
        "DTypeToken",
    ),
    "cuda_coop_tile_shuffle": (
        "N",
        "T",
        "Mode",
        "DTypeToken",
    ),
    "cuda_coop_tile_radix_rank": (
        "N",
        "T",
        "RadixBits",
        "Descending",
        "MemoizeOuterScan",
        "InnerScanAlgorithm",
        "SmemConfig",
        "DTypeToken",
    ),
    "cuda_coop_tile_merge_sort_keys": (
        "N",
        "T",
        "Descending",
        "PartialTile",
        "DTypeToken",
    ),
    "cuda_coop_tile_merge_sort_pairs": (
        "N",
        "KeyT",
        "ValueT",
        "Descending",
        "PartialTile",
        "KeyDTypeToken",
        "ValueDTypeToken",
    ),
    "cuda_coop_tile_radix_sort_keys": (
        "N",
        "T",
        "Descending",
        "RadixBits",
        "MemoizeOuterScan",
        "InnerScanAlgorithm",
        "SmemConfig",
        "DTypeToken",
    ),
    "cuda_coop_tile_radix_sort_pairs": (
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
    "cuda_coop_tile_topk_keys": (
        "N",
        "T",
        "Descending",
        "FullTile",
        "DTypeToken",
    ),
    "cuda_coop_tile_topk_pairs": (
        "N",
        "KeyT",
        "ValueT",
        "Descending",
        "FullTile",
        "KeyDTypeToken",
        "ValueDTypeToken",
    ),
}


def _record_constant_args(record):
    symbol_name = record[2]["symbol_name"]
    constant_params = _record_static_params(record)
    if symbol_name == "cuda_coop_tile_reduce_batched":
        names = (
            "N",
            "T",
            "Algorithm",
        )
        names += ("BinaryOp", "DTypeToken", "Batches")
    elif symbol_name == "cuda_coop_tile_symbol_test":
        names = tuple(f"Param{index}" for index in range(len(constant_params)))
    else:
        names = _CONSTANT_PARAM_NAMES[symbol_name]
    return dict(zip(names, constant_params, strict=True))


def _assert_provider_call(
    record,
    *,
    callee: str,
    meta: dict[str, str],
    exchange_width: int | None = None,
):
    op_cls, _, kwargs = record
    assert op_cls is _PreviewForeignCall
    assert kwargs["symbol_name"] == callee
    ((tilelib_path, tilelib_version),) = kwargs["tilelibs"]
    assert tilelib_version.startswith("test-tilelib-v1+includes.")
    assert tilelib_path.name == "cuda_coop_tile.tilelib"
    assert kwargs["constant_params"]
    assert kwargs["token"] is None
    constant_args = _record_constant_args(record)
    assert "ItemsPerThread" not in constant_args
    if exchange_width is None:
        assert "BlockThreads" not in constant_args
    else:
        assert constant_args["BlockThreads"] == exchange_width

    aliases = {
        "mode": "Mode",
        "algorithm": "Algorithm",
        "scan_op": "ScanOp",
        "initial_present": "InitialPresent",
        "prefix_present": "PrefixPresent",
        "binary_op": "BinaryOp",
        "descending": "Descending",
        "partial_tile": "PartialTile",
        "radix_bits": "RadixBits",
        "memoize_outer_scan": "MemoizeOuterScan",
        "inner_scan_algorithm": "InnerScanAlgorithm",
        "smem_config": "SmemConfig",
        "full_tile": "FullTile",
        "bins": "Bins",
        "dtype": "T",
        "key_dtype": "KeyT",
        "value_dtype": "ValueT",
    }
    dtype_suffixes = {
        "float16": "f16",
        "float32": "f32",
        "float64": "f64",
        "bfloat16": "bf16",
        "int8": "i8",
        "uint8": "u8",
        "int16": "i16",
        "uint16": "u16",
        "int32": "i32",
        "uint32": "u32",
        "int64": "i64",
        "uint64": "u64",
    }
    for key, value in meta.items():
        if key in {
            "prefix_value",
            "user_mode",
            "distance",
            "begin_bit",
            "end_bit",
        }:
            continue
        param = aliases[key]
        if param not in constant_args:
            assert (key == "bins" and value == "256") or (
                key == "dtype" and value == "u8"
            )
            continue
        actual = constant_args[param]
        if isinstance(actual, _DType):
            actual = dtype_suffixes[actual.name]
        assert str(actual) == value


def _install_test_schema(lowerings, constants, decl_type):
    schema = importlib.import_module("cuda.coop.tile._declarations")
    lowerings._PROVIDER_DECL_SCHEMAS["cuda_coop_tile_symbol_test"] = (
        schema.Declaration(
            constants=tuple(constants),
            operands=(decl_type, decl_type),
            results=(decl_type,),
            access=("in", "out"),
        )
    )


def _emit_symbol_test_call(lowerings, datatype, records, static_prefix):
    x = _Var("x", _TileType(static_prefix[1], (static_prefix[0],)))
    decl_type = lowerings._ProviderDeclTileType("Param1", ("Param0",))
    const_params = []
    for index, value in enumerate(static_prefix):
        kind = "Dim"
        if isinstance(value, str):
            kind = "String"
        elif any(
            value is dtype
            for dtype in (
                datatype.bfloat16,
                datatype.float16,
                datatype.float32,
                datatype.float64,
                datatype.int8,
                datatype.uint8,
                datatype.int16,
                datatype.uint16,
                datatype.int32,
                datatype.uint32,
                datatype.int64,
                datatype.uint64,
            )
        ):
            kind = "DType"
        const_params.append((f"Param{index}", kind))
    _install_test_schema(lowerings, const_params, decl_type)
    lowerings._emit_provider_foreign_call(
        operands=(x,),
        output_types=(x.tile_type,),
        decl_sym_name="cuda_coop_tile_symbol_test",
        static_prefix=static_prefix,
    )
    return records[-1][2]


def test_preview_call_symbol_is_frontend_neutral(monkeypatch, tmp_path):
    lowerings, datatype, records = _make_provider_backed_lowerings(
        monkeypatch, tmp_path
    )

    first = _emit_symbol_test_call(
        lowerings,
        datatype,
        records,
        (128, datatype.float32, 128, 1, "raking"),
    )
    second = _emit_symbol_test_call(
        lowerings,
        datatype,
        records,
        (256, datatype.int32, 64, 4, "warp_scans"),
    )

    assert first["symbol_name"] == "cuda_coop_tile_symbol_test"
    assert second["symbol_name"] == first["symbol_name"]
    assert len(records) == 2
    assert first["tilelibs"] == second["tilelibs"]
    assert first["tilelibs"][0][0].name == "cuda_coop_tile.tilelib"
    assert _record_constant_args(records[0]) == {
        "Param0": 128,
        "Param1": datatype.float32,
        "Param2": 128,
        "Param3": 1,
        "Param4": "raking",
    }
    assert _record_constant_args(records[1]) == {
        "Param0": 256,
        "Param1": datatype.int32,
        "Param2": 64,
        "Param3": 4,
        "Param4": "warp_scans",
    }
    assert [operand.get_type().shape for operand in first["inputs"]] == [
        (128,),
        (128,),
    ]
    assert [operand.get_type().shape for operand in second["inputs"]] == [
        (256,),
        (256,),
    ]
    assert first["inputs"][0].get_type().dtype is datatype.float32
    assert second["inputs"][0].get_type().dtype is datatype.int32


def test_preview_calls_reuse_exact_symbol(monkeypatch, tmp_path):
    lowerings, datatype, records = _make_provider_backed_lowerings(
        monkeypatch, tmp_path
    )
    static_prefix = (128, datatype.float32, 128, 1, "raking")

    first = _emit_symbol_test_call(lowerings, datatype, records, static_prefix)
    second = _emit_symbol_test_call(lowerings, datatype, records, static_prefix)

    assert (
        first["symbol_name"]
        == second["symbol_name"]
        == "cuda_coop_tile_symbol_test"
    )
    assert len(records) == 2


@pytest.mark.parametrize("mismatch", ["dtype", "shape"])
def test_provider_call_rejects_concrete_type_outside_canonical_schema(
    monkeypatch, tmp_path, mismatch
):
    lowerings, datatype, records = _make_provider_backed_lowerings(
        monkeypatch, tmp_path
    )
    x = _Var("x", _TileType(datatype.float32, (128,)))
    decl_type = lowerings._ProviderDeclTileType("T", ("N",))
    _install_test_schema(lowerings, (("N", "Dim"), ("T", "DType")), decl_type)
    expected_dtype = datatype.int32 if mismatch == "dtype" else datatype.float32
    expected_extent = 64 if mismatch == "shape" else 128

    with pytest.raises(
        lowerings.TileTypeError,
        match=r"provider operand 0.*canonical schema and call constants",
    ):
        lowerings._emit_provider_foreign_call(
            operands=(x,),
            output_types=(x.tile_type,),
            decl_sym_name="cuda_coop_tile_symbol_test",
            static_prefix=(expected_extent, expected_dtype),
        )

    assert records == []


def test_provider_call_rejects_non_dtype_parameter_as_schema_element(
    monkeypatch, tmp_path
):
    lowerings, datatype, records = _make_provider_backed_lowerings(
        monkeypatch, tmp_path
    )
    x = _Var("x", _TileType(datatype.float32, (128,)))
    decl_type = lowerings._ProviderDeclTileType("DTypeToken", ("N",))
    _install_test_schema(
        lowerings, (("N", "Dim"), ("DTypeToken", "String")), decl_type
    )

    with pytest.raises(
        lowerings.TileTypeError,
        match=r"'DTypeToken' as an element dtype.*not declared as DType",
    ):
        lowerings._emit_provider_foreign_call(
            operands=(x,),
            output_types=(x.tile_type,),
            decl_sym_name="cuda_coop_tile_symbol_test",
            static_prefix=(128, "f32"),
        )

    assert records == []


def test_provider_path_rejects_missing_explicit_path(monkeypatch, tmp_path):
    tile_library_path = tmp_path / "missing" / "cuda_coop_tile.tilelib"
    lowerings, _, _ = _load_lowerings_with_cuda_tile_stubs(
        monkeypatch, tile_library_path
    )

    with pytest.raises(lowerings.TileTypeError) as excinfo:
        lowerings._coop_tile_linker_input()

    message = str(excinfo.value)
    assert "CUDA_COOP_TILE_LIBRARY_PATH does not exist" in message
    assert str(tile_library_path) in message
    assert "cuda_coop_tile.tilelib" in message


def test_provider_path_rejects_directory_explicit_path(monkeypatch, tmp_path):
    tile_library_path = tmp_path / "tilelib-dir"
    tile_library_path.mkdir()
    lowerings, _, _ = _load_lowerings_with_cuda_tile_stubs(
        monkeypatch, tile_library_path
    )

    with pytest.raises(lowerings.TileTypeError) as excinfo:
        lowerings._coop_tile_linker_input()

    message = str(excinfo.value)
    assert "CUDA_COOP_TILE_LIBRARY_PATH must point to a file" in message
    assert str(tile_library_path) in message
    assert "cuda_coop_tile.tilelib" in message


def test_provider_path_default_error_includes_build_hint(monkeypatch):
    lowerings, _, _ = _load_lowerings_with_cuda_tile_stubs(monkeypatch)
    cpp_tile = importlib.import_module("cuda.coop.tile._cpp")
    monkeypatch.setattr(cpp_tile, "_packaged_tile_library_candidates", tuple)

    with pytest.raises(lowerings.TileTypeError) as excinfo:
        lowerings._coop_tile_linker_input()

    message = str(excinfo.value)
    assert "cuda.coop.tile requires its TileIR library" in message
    assert "Install cuda-coop-tile with its native TileLib" in message
    assert "CUDA_COOP_TILE_LIBRARY_PATH" in message


def test_provider_path_uses_shared_cpp_tilelib_resolver(monkeypatch, tmp_path):
    lowerings, datatype, records = _load_lowerings_with_cuda_tile_stubs(
        monkeypatch
    )
    tile_library = (
        tmp_path / "installed package with spaces" / "cuda_coop_tile.tilelib"
    )
    tile_library.parent.mkdir()
    tile_library.touch()
    monkeypatch.setattr(
        lowerings,
        "_get_tile_library_linker_input",
        lambda: ("cuda-coop-tile", "1.2.3", tile_library),
    )

    name, version, path = lowerings._coop_tile_linker_input()
    assert name == "cuda-coop-tile"
    assert version.startswith("1.2.3+includes.")
    assert path == tile_library
    x = _Var("x", _TileType(datatype.int32, (128,)))
    lowerings.sum_impl(
        x,
        _Const(None),
        _Const(False),
        _Const(None),
        _Const(None),
        _Const(False),
    )
    assert records[0][2]["tilelibs"] == (
        (tile_library, lowerings._coop_tile_linker_input()[1]),
    )


def test_provider_include_paths_prefer_explicit_cccl_root(
    monkeypatch, tmp_path
):
    cccl_root = tmp_path / "cccl"
    for relative in ("thrust", "cub", "libcudacxx/include"):
        (cccl_root / relative).mkdir(parents=True)
    (cccl_root / "cub/cub").mkdir()
    (cccl_root / "cub/cub/version.cuh").write_text("", encoding="utf-8")
    for header in (
        "block/block_adjacent_difference.cuh",
        "block/block_discontinuity.cuh",
        "block/block_exchange.cuh",
        "block/block_histogram.cuh",
        "block/block_merge_sort.cuh",
        "block/block_radix_rank.cuh",
        "block/block_radix_sort.cuh",
        "block/block_reduce.cuh",
        "block/block_run_length_decode.cuh",
        "block/block_scan.cuh",
        "block/block_shuffle.cuh",
        "block/block_topk.cuh",
        "thread/thread_operators.cuh",
    ):
        path = cccl_root / "cub/cub" / header
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    monkeypatch.setenv("CUDA_COOP_CCCL_ROOT", str(cccl_root))

    lowerings, _, _ = _load_lowerings_with_cuda_tile_stubs(monkeypatch)

    assert lowerings._provider_include_paths()[:3] == [
        str(cccl_root / "thrust"),
        str(cccl_root / "cub"),
        str(cccl_root / "libcudacxx" / "include"),
    ]


def test_provider_lowering_exports_encoded_include_paths_once(
    monkeypatch, tmp_path
):
    lowerings, datatype, records = _make_provider_backed_lowerings(
        monkeypatch, tmp_path
    )
    includes = [
        str(tmp_path / "wheel bundle with spaces"),
        str(tmp_path / "toolkit;component"),
    ]
    monkeypatch.delenv(lowerings._PROVIDER_INCLUDE_PATH_ENV, raising=False)
    monkeypatch.setattr(lowerings, "_provider_include_paths", lambda: includes)

    x = _Var("x", _TileType(datatype.float32, (256,)))
    lowerings.sum_impl(
        x,
        _Const(None),
        _Const(False),
        _Const(None),
        _Const(None),
        _Const(False),
    )

    assert records
    assert os.environ[lowerings._PROVIDER_INCLUDE_PATH_ENV] == (
        lowerings._PROVIDER_INCLUDE_PATH_SEPARATOR.join(includes)
    )

    monkeypatch.setenv(lowerings._PROVIDER_INCLUDE_PATH_ENV, "caller-owned")
    monkeypatch.setattr(
        lowerings,
        "_provider_include_paths",
        lambda: pytest.fail("explicit include paths must remain authoritative"),
    )
    lowerings.sum_impl(
        x,
        _Const(None),
        _Const(False),
        _Const(None),
        _Const(None),
        _Const(False),
    )
    assert os.environ[lowerings._PROVIDER_INCLUDE_PATH_ENV] == "caller-owned"


def test_sum_lowering_emits_preview_call_and_metadata(monkeypatch, tmp_path):
    lowerings, datatype, records = _make_provider_backed_lowerings(
        monkeypatch, tmp_path
    )

    x = _Var("x", _TileType(datatype.float32, (256,)))
    result = lowerings.sum_impl(
        x,
        _Const(None),
        _Const(False),
        _Const(None),
        _Const(None),
        _Const(False),
    )

    assert result.tile_type.dtype is datatype.float32
    assert len(records) == 1
    op_cls, output_type, kwargs = records[0]
    assert op_cls is _PreviewForeignCall
    assert output_type[0].dtype is datatype.float32
    assert isinstance(output_type[-1], _TokenType)
    assert kwargs["symbol_name"] == "cuda_coop_tile_sum_batched"
    assert kwargs["tilelibs"] == (
        (
            tmp_path / "cuda_coop_tile.tilelib",
            lowerings._coop_tile_linker_input()[1],
        ),
    )
    assert kwargs["inputs"][0].get_type().shape == x.get_type().shape
    assert len(kwargs["inputs"]) == 2
    assert kwargs["inputs"][1].get_type().shape == (1,)
    assert result.get_type().shape == ()

    static_params = _record_static_params(records[0])
    assert static_params[:3] == (256, datatype.float32, "warp_reductions")
    assert static_params[-2:] == ("f32", 1)
    assert len(static_params) == 5
    assert _record_constant_args(records[0])["Algorithm"] == "warp_reductions"


def test_tuple_output_lowering_builds_tuple_and_metadata(monkeypatch, tmp_path):
    lowerings, datatype, records = _make_provider_backed_lowerings(
        monkeypatch, tmp_path
    )

    x = _Var("x", _TileType(datatype.int32, (128,)))
    result = lowerings.discontinuity_flag_heads_and_tails_impl(x, _Const(None))

    assert result[0] == "tuple"
    assert [var.tile_type.dtype for var in result[1]] == [
        datatype.int32,
        datatype.int32,
    ]
    assert len(records) == 1
    op_cls, output_types, kwargs = records[0]
    assert op_cls is _PreviewForeignCall
    assert (
        kwargs["symbol_name"] == "cuda_coop_tile_discontinuity_heads_and_tails"
    )
    assert kwargs["tilelibs"] == (
        (
            tmp_path / "cuda_coop_tile.tilelib",
            lowerings._coop_tile_linker_input()[1],
        ),
    )
    assert [ty.dtype for ty in output_types[:-1]] == [
        datatype.int32,
        datatype.int32,
    ]
    assert [ty.shape for ty in output_types[:-1]] == [(128,), (128,)]
    assert len(kwargs["inputs"]) == 3

    static_params = _record_static_params(records[0])
    assert static_params[:3] == (128, datatype.int32, "heads_and_tails")
    assert static_params[-1] == "i32"
    assert len(static_params) == 4
    assert _record_constant_args(records[0])["Mode"] == "heads_and_tails"


def test_exchange_inverse_fold_returns_original_operand(monkeypatch, tmp_path):
    lowerings, datatype, records = _make_provider_backed_lowerings(
        monkeypatch, tmp_path
    )

    x = _Var("x", _TileType(datatype.float32, (128,)))
    striped = lowerings.exchange_blocked_to_striped_impl(
        x, _Const(128), _Const(None)
    )
    folded = lowerings.exchange_striped_to_blocked_impl(
        striped, _Const(128), _Const(None)
    )
    y = _Var("y", _TileType(datatype.float32, (128,)))
    blocked = lowerings.exchange_striped_to_blocked_impl(
        y, _Const(128), _Const(None)
    )
    reverse_folded = lowerings.exchange_blocked_to_striped_impl(
        blocked, _Const(128), _Const(None)
    )

    assert folded is x
    assert reverse_folded is y
    assert len(records) == 2
    assert _record_constant_args(records[0])["Mode"] == "striped_to_blocked"
    assert _record_constant_args(records[1])["Mode"] == "blocked_to_striped"
    for _, _, kwargs in records:
        assert kwargs["symbol_name"] == "cuda_coop_tile_exchange"
        assert len(kwargs["inputs"]) == 1


@pytest.mark.parametrize(
    "first_threads, second_threads", [(64, 128), (128, 256)]
)
def test_exchange_inverse_fold_preserves_different_topologies(
    monkeypatch, tmp_path, first_threads, second_threads
):
    lowerings, datatype, records = _make_provider_backed_lowerings(
        monkeypatch, tmp_path
    )
    values = _Var("values", _TileType(datatype.int32, (256,)))
    striped = lowerings.exchange_blocked_to_striped_impl(
        values, _Const(first_threads), _Const(None)
    )
    blocked = lowerings.exchange_striped_to_blocked_impl(
        striped, _Const(second_threads), _Const(None)
    )
    assert blocked is not values
    assert len(records) == 2
    assert [
        _record_constant_args(record)["BlockThreads"] for record in records
    ] == [
        first_threads,
        second_threads,
    ]


def test_exchange_rejects_runtime_topology(monkeypatch, tmp_path):
    lowerings, datatype, _ = _make_provider_backed_lowerings(
        monkeypatch, tmp_path
    )
    values = _Var("values", _TileType(datatype.int32, (256,)))
    threads = _Var("threads", datatype.int32)
    with pytest.raises(
        lowerings.TileTypeError,
        match="block_threads must be a compile-time constant",
    ):
        lowerings.exchange_striped_to_blocked_impl(
            values, threads, _Const(None)
        )


def test_exchange_inverse_fold_follows_identity_assignment(
    monkeypatch, tmp_path
):
    lowerings, datatype, records = _make_provider_backed_lowerings(
        monkeypatch, tmp_path
    )

    x = _Var("x", _TileType(datatype.float32, (128,)))
    striped = lowerings.exchange_blocked_to_striped_impl(
        x, _Const(128), _Const(None)
    )
    assigned = _Var("assigned", striped.tile_type, striped.ctx)
    assign_op = _Assign()
    assign_op.result_vars = (assigned,)
    assign_op.value = striped
    striped.ctx.producers[assigned.name] = assign_op

    folded = lowerings.exchange_striped_to_blocked_impl(
        assigned, _Const(128), _Const(None)
    )

    assert folded is x
    assert len(records) == 1


def test_scan_lowerings_emit_provider_requests(monkeypatch, tmp_path):
    lowerings, datatype, records = _make_provider_backed_lowerings(
        monkeypatch, tmp_path
    )
    x = _Var("x", _TileType(datatype.int32, (128,)))

    lowerings.exclusive_sum_impl(x, _Const(None), _Const(None), _Const(None))
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_scan",
        meta={"mode": "exclusive", "dtype": "i32"},
    )
    constant_args = _record_constant_args(records[-1])
    assert constant_args == {
        "N": 128,
        "T": datatype.int32,
        "Mode": "exclusive",
        "Algorithm": "raking",
        "ScanOp": "sum",
        "InitialPresent": 0,
        "PrefixPresent": 0,
        "DTypeToken": "i32",
    }
    assert [
        operand.get_type().shape for operand in records[-1][2]["inputs"]
    ] == [
        (128,),
        (),
        (),
        (128,),
    ]
    assert records[-1][2]["inputs"][1].get_constant() == 0
    assert records[-1][2]["inputs"][2].get_constant() == 0

    constant_prefix_args = dict(constant_args)
    dynamic_prefix = _Var("dynamic_prefix", datatype.int32)
    lowerings.inclusive_scan_impl(
        x,
        _Const("max"),
        _Const(None),
        dynamic_prefix,
        _Const(None),
        _Const(None),
    )
    assert _record_constant_args(records[-1]) == {
        **constant_prefix_args,
        "Mode": "inclusive",
        "ScanOp": "max",
        "PrefixPresent": 1,
    }
    assert records[-1][2]["inputs"][2].get_type().shape == ()

    dynamic_initial = _Var("dynamic_initial", datatype.int32)
    lowerings.exclusive_scan_impl(
        x,
        _Const("*"),
        dynamic_initial,
        _Const(None),
        _Const(None),
        _Const(None),
    )
    assert _record_constant_args(records[-1]) == {
        **constant_prefix_args,
        "Mode": "exclusive",
        "ScanOp": "mul",
        "InitialPresent": 1,
        "PrefixPresent": 0,
    }
    assert records[-1][2]["inputs"][1].get_type().shape == ()

    lowerings.exclusive_scan_impl(
        x,
        _Const(None),
        _Const(None),
        _Const(None),
        _Const(None),
        _Const(None),
    )
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_scan",
        meta={
            "mode": "exclusive",
            "scan_op": "sum",
            "initial_present": "0",
            "prefix_present": "0",
        },
    )
    constant_args = _record_constant_args(records[-1])
    assert constant_args["ScanOp"] == "sum"
    assert constant_args["InitialPresent"] == 0
    assert constant_args["PrefixPresent"] == 0
    assert records[-1][2]["inputs"][1].get_constant() == 0
    assert records[-1][2]["inputs"][2].get_constant() == 0

    lowerings.inclusive_sum_impl(
        x, _Const(7), _Const("warp_scans"), _Const(None)
    )
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_scan",
        meta={"mode": "inclusive", "prefix_present": "1", "prefix_value": "7"},
    )
    constant_args = _record_constant_args(records[-1])
    assert constant_args["Mode"] == "inclusive"
    assert constant_args["Algorithm"] == "warp_scans"
    assert constant_args["ScanOp"] == "sum"
    assert constant_args["InitialPresent"] == 0
    assert constant_args["PrefixPresent"] == 1
    assert records[-1][2]["inputs"][1].get_constant() == 0
    assert records[-1][2]["inputs"][2].get_constant() == 7

    lowerings.exclusive_scan_impl(
        x,
        _Const("*"),
        _Const(1),
        _Const(None),
        _Const(None),
        _Const(None),
    )
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_scan",
        meta={"mode": "exclusive", "scan_op": "mul", "initial_present": "1"},
    )
    constant_args = _record_constant_args(records[-1])
    assert constant_args["Mode"] == "exclusive"
    assert constant_args["Algorithm"] == "raking"
    assert constant_args["ScanOp"] == "mul"
    assert constant_args["InitialPresent"] == 1
    assert constant_args["PrefixPresent"] == 0
    assert records[-1][2]["inputs"][1].get_constant() == 1
    assert records[-1][2]["inputs"][2].get_constant() == 0

    lowerings.inclusive_scan_impl(
        x,
        _Const("max"),
        _Const(None),
        _Const(0),
        _Const(None),
        _Const(None),
    )
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_scan",
        meta={"mode": "inclusive", "scan_op": "max", "prefix_present": "1"},
    )
    constant_args = _record_constant_args(records[-1])
    assert constant_args["Mode"] == "inclusive"
    assert constant_args["Algorithm"] == "raking"
    assert constant_args["ScanOp"] == "max"
    assert constant_args["InitialPresent"] == 0
    assert constant_args["PrefixPresent"] == 1
    assert records[-1][2]["inputs"][1].get_constant() == 0
    assert records[-1][2]["inputs"][2].get_constant() == 0


def test_reduce_lowerings_emit_provider_requests(monkeypatch, tmp_path):
    lowerings, datatype, records = _make_provider_backed_lowerings(
        monkeypatch, tmp_path
    )
    x = _Var("x", _TileType(datatype.int32, (256,)))

    lowerings.sum_impl(
        x,
        _Const(None),
        _Const(False),
        _Const("raking"),
        _Const(None),
        _Const(False),
    )
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_sum_batched",
        meta={"algorithm": "raking", "dtype": "i32"},
    )
    assert _record_constant_args(records[-1])["Algorithm"] == "raking"

    lowerings.reduce_impl(
        x,
        _Const(None),
        _Const(False),
        _Const("max"),
        _Const(None),
        _Const(None),
        _Const(False),
    )
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_reduce_batched",
        meta={
            "algorithm": "warp_reductions",
            "binary_op": "max",
            "dtype": "i32",
        },
    )
    constant_args = _record_constant_args(records[-1])
    assert constant_args["Algorithm"] == "warp_reductions"
    assert constant_args["BinaryOp"] == "max"


def test_sort_lowerings_emit_provider_requests(monkeypatch, tmp_path):
    lowerings, datatype, records = _make_provider_backed_lowerings(
        monkeypatch, tmp_path
    )
    keys = _Var("keys", _TileType(datatype.int32, (128,)))
    values = _Var("values", _TileType(datatype.float32, (128,)))

    lowerings.merge_sort_keys_impl(
        keys, _Const(True), _Const(127), _Const(0), _Const(None)
    )
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_merge_sort_keys",
        meta={"descending": "1", "partial_tile": "1"},
    )
    merge_key_operands = records[-1][2]["inputs"]
    assert merge_key_operands[0] is keys
    assert merge_key_operands[1].get_constant() == 127
    assert merge_key_operands[2].get_constant() == 0
    constant_args = _record_constant_args(records[-1])
    assert constant_args["Descending"] == 1
    assert constant_args["PartialTile"] == 1

    runtime_valid_items = _Var("runtime_valid_items", datatype.int32)
    runtime_oob_default = _Var("runtime_oob_default", datatype.int32)
    lowerings.merge_sort_keys_impl(
        keys,
        _Const(False),
        runtime_valid_items,
        runtime_oob_default,
        _Const(None),
    )
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_merge_sort_keys",
        meta={"descending": "0", "partial_tile": "1"},
    )
    merge_runtime_operands = records[-1][2]["inputs"]
    assert merge_runtime_operands[0] is keys
    assert merge_runtime_operands[1].tile_type.shape == ()
    assert merge_runtime_operands[2].tile_type.shape == ()

    lowerings.merge_sort_keys_impl(
        keys,
        _Const(False),
        runtime_valid_items,
        _Const(None),
        _Const({"oob_default": 0}),
    )
    merge_split_operands = records[-1][2]["inputs"]
    assert merge_split_operands[1].name.startswith("runtime_valid_items_")
    assert merge_split_operands[2].get_constant() == 0

    lowerings.merge_sort_pairs_impl(
        keys, values, _Const(False), _Const(None), _Const(None), _Const(None)
    )
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_merge_sort_pairs",
        meta={"descending": "0", "key_dtype": "i32", "value_dtype": "f32"},
    )
    constant_args = _record_constant_args(records[-1])
    assert constant_args["Descending"] == 0
    assert constant_args["PartialTile"] == 0

    lowerings.merge_sort_pairs_impl(
        keys,
        values,
        _Const(False),
        _Const(None),
        runtime_oob_default,
        _Const({"valid_items": 127}),
    )
    merge_pair_split_operands = records[-1][2]["inputs"]
    assert merge_pair_split_operands[2].get_constant() == 127
    assert merge_pair_split_operands[3].name.startswith("runtime_oob_default_")

    lowerings.radix_sort_keys_impl(
        keys,
        _Const(0),
        _Const(16),
        _Const(True),
        _Const(4),
        _Const(True),
        _Const("warp_scans"),
        _Const("four_byte"),
        _Const(None),
    )
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_radix_sort_keys",
        meta={"descending": "1", "radix_bits": "4"},
    )
    constant_args = _record_constant_args(records[-1])
    assert constant_args["Descending"] == 1
    assert constant_args["RadixBits"] == 4
    assert constant_args["MemoizeOuterScan"] == 1
    assert constant_args["InnerScanAlgorithm"] == "warp_scans"
    assert constant_args["SmemConfig"] == "four_byte"
    assert constant_args["DTypeToken"] == "i32"
    radix_key_operands = records[-1][2]["inputs"]
    assert radix_key_operands[0] is keys
    assert radix_key_operands[1].get_constant() == 0
    assert radix_key_operands[2].get_constant() == 16
    assert radix_key_operands[1].tile_type.shape == ()
    assert radix_key_operands[2].tile_type.shape == ()

    runtime_begin_bit = _Var("runtime_begin_bit", datatype.int32)
    runtime_end_bit = _Var("runtime_end_bit", datatype.int32)
    lowerings.radix_sort_keys_impl(
        keys,
        runtime_begin_bit,
        runtime_end_bit,
        _Const(False),
        _Const(4),
        _Const(None),
        _Const(None),
        _Const(None),
        _Const(None),
    )
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_radix_sort_keys",
        meta={"descending": "0", "radix_bits": "4"},
    )
    radix_runtime_operands = records[-1][2]["inputs"]
    assert radix_runtime_operands[0] is keys
    assert radix_runtime_operands[1].tile_type.shape == ()
    assert radix_runtime_operands[2].tile_type.shape == ()

    lowerings.radix_sort_pairs_impl(
        keys,
        values,
        _Const(0),
        _Const(16),
        _Const(True),
        _Const(4),
        _Const(False),
        _Const("raking"),
        _Const("eight_byte"),
        _Const(None),
    )
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_radix_sort_pairs",
        meta={
            "descending": "1",
            "radix_bits": "4",
            "key_dtype": "i32",
            "value_dtype": "f32",
        },
    )
    constant_args = _record_constant_args(records[-1])
    assert constant_args["Descending"] == 1
    assert constant_args["RadixBits"] == 4
    assert constant_args["MemoizeOuterScan"] == 0
    assert constant_args["InnerScanAlgorithm"] == "raking"
    assert constant_args["SmemConfig"] == "eight_byte"
    assert constant_args["KeyDTypeToken"] == "i32"
    assert constant_args["ValueDTypeToken"] == "f32"
    radix_pair_operands = records[-1][2]["inputs"]
    assert radix_pair_operands[0:2] == (keys, values)
    assert radix_pair_operands[2].get_constant() == 0
    assert radix_pair_operands[3].get_constant() == 16
    assert radix_pair_operands[2].tile_type.shape == ()
    assert radix_pair_operands[3].tile_type.shape == ()

    lowerings.radix_sort_pairs_impl(
        keys,
        values,
        runtime_begin_bit,
        runtime_end_bit,
        _Const(False),
        _Const(None),
        _Const(None),
        _Const(None),
        _Const(None),
        _Const(None),
    )
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_radix_sort_pairs",
        meta={"descending": "0", "key_dtype": "i32", "value_dtype": "f32"},
    )
    radix_pair_runtime_operands = records[-1][2]["inputs"]
    assert radix_pair_runtime_operands[0:2] == (keys, values)
    assert radix_pair_runtime_operands[2].tile_type.shape == ()
    assert radix_pair_runtime_operands[3].tile_type.shape == ()

    lowerings.topk_max_keys_impl(
        keys,
        _Const(7),
        _Const(None),
        _Const(0),
        _Const(16),
        _Const(None),
    )
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_topk_keys",
        meta={"descending": "1", "full_tile": "1"},
    )
    topk_key_operands = records[-1][2]["inputs"]
    assert topk_key_operands[0] is keys
    assert topk_key_operands[1].get_constant() == 7
    assert topk_key_operands[2].get_constant() == 128
    assert topk_key_operands[3].get_constant() == 0
    assert topk_key_operands[4].get_constant() == 16
    constant_args = _record_constant_args(records[-1])
    assert constant_args["Descending"] == 1
    assert constant_args["FullTile"] == 1

    lowerings.topk_min_pairs_impl(
        keys,
        values,
        _Const(5),
        _Const(64),
        _Const(None),
        _Const(None),
        _Const(None),
    )
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_topk_pairs",
        meta={
            "descending": "0",
            "full_tile": "0",
            "key_dtype": "i32",
            "value_dtype": "f32",
        },
    )
    topk_pair_operands = records[-1][2]["inputs"]
    assert topk_pair_operands[0:2] == (keys, values)
    assert topk_pair_operands[2].get_constant() == 5
    assert topk_pair_operands[3].get_constant() == 64
    constant_args = _record_constant_args(records[-1])
    assert constant_args["Descending"] == 0
    assert constant_args["FullTile"] == 0

    runtime_k = _Var("runtime_k", datatype.int32)
    lowerings.topk_max_keys_impl(
        keys,
        runtime_k,
        _Const(None),
        _Const(0),
        _Const(16),
        _Const(None),
    )
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_topk_keys",
        meta={"descending": "1", "full_tile": "1"},
    )
    runtime_operands = records[-1][2]["inputs"]
    assert runtime_operands[0] is keys
    assert runtime_operands[1].tile_type.shape == ()
    assert runtime_operands[2].get_constant() == 128
    constant_args = _record_constant_args(records[-1])
    assert constant_args["Descending"] == 1
    assert constant_args["FullTile"] == 1

    shape_one_k = _Var("shape_one_k", _TileType(datatype.int32, (1,)))
    with pytest.raises(
        lowerings.TileTypeError,
        match="k must be a scalar or rank-zero tile",
    ):
        lowerings.topk_max_keys_impl(
            keys,
            shape_one_k,
            _Const(None),
            _Const(0),
            _Const(16),
            _Const(None),
        )

    multidimensional_shape_one_k = _Var(
        "multidimensional_shape_one_k", _TileType(datatype.int32, (1, 1))
    )
    with pytest.raises(
        lowerings.TileTypeError,
        match="k must be a scalar or rank-zero tile",
    ):
        lowerings.topk_max_keys_impl(
            keys,
            multidimensional_shape_one_k,
            _Const(None),
            _Const(0),
            _Const(16),
            _Const(None),
        )

    with pytest.raises(lowerings.TileTypeError, match="k must be an int"):
        lowerings.topk_max_keys_impl(
            keys,
            _Const(None),
            _Const(None),
            _Const(None),
            _Const(None),
            _Const(None),
        )
    with pytest.raises(lowerings.TileTypeError, match="k must be <= 128"):
        lowerings.topk_max_keys_impl(
            keys,
            _Const(129),
            _Const(64),
            _Const(None),
            _Const(None),
            _Const(None),
        )
    with pytest.raises(
        lowerings.TileTypeError, match="end_bit must be > begin_bit"
    ):
        lowerings.topk_max_keys_impl(
            keys,
            _Const(7),
            _Const(None),
            _Const(8),
            _Const(8),
            _Const(None),
        )
    with pytest.raises(
        lowerings.TileTypeError, match="begin_bit must be < key bit width"
    ):
        lowerings.topk_max_keys_impl(
            keys,
            _Const(7),
            _Const(None),
            _Const(32),
            _Const(None),
            _Const(None),
        )


def test_warp_merge_sort_lowerings_emit_logical_requests(monkeypatch, tmp_path):
    lowerings, datatype, records = _make_provider_backed_lowerings(
        monkeypatch, tmp_path
    )
    keys = _Var("keys", _TileType(datatype.int32, (32,)))
    values = _Var("values", _TileType(datatype.float32, (32,)))

    lowerings.merge_sort_keys_impl(
        keys, _Const(False), _Const(None), _Const(None), _Const(None)
    )
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_merge_sort_keys",
        meta={"descending": "0", "partial_tile": "0"},
    )

    lowerings.merge_sort_pairs_impl(
        keys,
        values,
        _Const(False),
        _Const(None),
        _Const(None),
        _Const(None),
    )
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_merge_sort_pairs",
        meta={"descending": "0", "partial_tile": "0"},
    )


def test_unary_and_histogram_lowerings_emit_provider_requests(
    monkeypatch, tmp_path
):
    lowerings, datatype, records = _make_provider_backed_lowerings(
        monkeypatch, tmp_path
    )
    x = _Var("x", _TileType(datatype.uint8, (256,)))
    y = _Var("y", _TileType(datatype.int32, (128,)))

    lowerings.histogram_impl(x, _Const(64), _Const(None))
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_histogram",
        meta={"bins": "64", "dtype": "u8"},
    )
    assert _record_constant_args(records[-1])["Bins"] == 64

    lowerings.histogram_256_impl(x, _Const(None))
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_histogram_256",
        meta={"bins": "256", "dtype": "u8"},
    )
    assert _record_static_params(records[-1]) == (256, "u8")

    lowerings.discontinuity_flag_heads_impl(y, _Const(None))
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_discontinuity_flags",
        meta={"mode": "heads", "dtype": "i32"},
    )

    lowerings.adjacent_difference_subtract_right_impl(y, _Const(None))
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_adjacent_difference",
        meta={"mode": "subtract_right", "dtype": "i32"},
    )
    assert _record_constant_args(records[-1])["Mode"] == "subtract_right"

    lowerings.exchange_striped_to_blocked_impl(y, _Const(128), _Const(None))
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_exchange",
        meta={"mode": "blocked_to_striped", "user_mode": "striped_to_blocked"},
        exchange_width=128,
    )
    constant_args = _record_constant_args(records[-1])
    assert constant_args["Mode"] == "blocked_to_striped"
    assert constant_args["DTypeToken"] == "i32"
    assert len(records[-1][2]["inputs"]) == 1
    assert records[-1][2]["inputs"][0] is y

    lowerings.shuffle_rotate_impl(y, _Const(3), _Const(None))
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_shuffle",
        meta={"mode": "rotate", "distance": "3"},
    )
    constant_args = _record_constant_args(records[-1])
    assert constant_args["Mode"] == "rotate"
    assert records[-1][2]["inputs"][1].get_constant() == 3

    constant_shuffle_args = dict(constant_args)
    dynamic_distance = _Var("dynamic_distance", datatype.int32)
    lowerings.shuffle_rotate_impl(y, dynamic_distance, _Const(None))
    assert _record_constant_args(records[-1]) == constant_shuffle_args
    assert records[-1][2]["inputs"][1].get_type().shape == ()

    lowerings.radix_rank_keys_impl(
        y,
        _Const(0),
        _Const(8),
        _Const(8),
        _Const(True),
        _Const(None),
        _Const(None),
        _Const(None),
        _Const(None),
    )
    _assert_provider_call(
        records[-1],
        callee="cuda_coop_tile_radix_rank",
        meta={"begin_bit": "0", "end_bit": "8", "descending": "1"},
    )
    constant_args = _record_constant_args(records[-1])
    assert constant_args["RadixBits"] == 8
    assert constant_args["Descending"] == 1

    constant_rank_args = dict(constant_args)
    dynamic_begin = _Var("dynamic_begin", datatype.int32)
    dynamic_end = _Var("dynamic_end", datatype.int32)
    lowerings.radix_rank_keys_impl(
        y,
        dynamic_begin,
        dynamic_end,
        _Const(8),
        _Const(True),
        _Const(None),
        _Const(None),
        _Const(None),
        _Const(None),
    )
    assert _record_constant_args(records[-1]) == constant_rank_args
    assert records[-1][2]["inputs"][1].get_type().shape == ()
    assert records[-1][2]["inputs"][2].get_type().shape == ()

    lowerings.radix_rank_keys_impl(
        y,
        dynamic_begin,
        _Const(None),
        _Const(8),
        _Const(True),
        _Const(None),
        _Const(None),
        _Const(None),
        _Const(None),
    )
    assert _record_constant_args(records[-1]) == constant_rank_args
    defaulted_dynamic_operands = records[-1][2]["inputs"]
    assert defaulted_dynamic_operands[1] is not dynamic_begin
    assert defaulted_dynamic_operands[1].get_type().shape == ()
    assert defaulted_dynamic_operands[2].name.startswith("min_add_")
    assert defaulted_dynamic_operands[2].get_type().shape == ()


@pytest.mark.parametrize(
    "memoize,algorithm,smem,error",
    [
        (False, "raking", "eight_byte", None),
        (True, "raking_memoize", 4, None),
        (1, None, None, "memoize_outer_scan must be a bool"),
        (None, "invalid", None, "inner_scan_algorithm must be one of"),
        (None, None, 16, "smem_config must be one of"),
    ],
)
def test_radix_rank_keys_policy_controls(
    monkeypatch, tmp_path, memoize, algorithm, smem, error
):
    lowerings, dtype, records = _make_provider_backed_lowerings(
        monkeypatch, tmp_path
    )
    keys = _Var("keys", _TileType(dtype.int32, (128,)))
    args = (
        keys,
        _Const(28),
        _Const(32),
        _Const(4),
        _Const(False),
        _Const(memoize),
        _Const(algorithm),
        _Const(smem),
        _Const(None),
    )
    if error is not None:
        with pytest.raises(lowerings.TileTypeError, match=error):
            lowerings.radix_rank_keys_impl(*args)
        return
    lowerings.radix_rank_keys_impl(*args)
    constants = _record_constant_args(records[-1])
    assert constants["MemoizeOuterScan"] == int(memoize)
    assert constants["InnerScanAlgorithm"] == algorithm
    assert constants["SmemConfig"] == ("four_byte" if smem == 4 else smem)


@pytest.mark.parametrize("with_offsets", [False, True])
def test_run_length_decode_preserves_independent_extents_and_dtypes(
    monkeypatch, tmp_path, with_offsets
):
    lowerings, dtype, records = _make_provider_backed_lowerings(
        monkeypatch, tmp_path
    )
    values = _Var("values", _TileType(dtype.float64, (128,)))
    lengths = _Var("lengths", _TileType(dtype.uint64, (128,)))
    lowering = (
        lowerings.run_length_decode_with_offsets_impl
        if with_offsets
        else lowerings.run_length_decode_impl
    )
    lowering(values, lengths, _Const(256), _Const(1 << 40), _Const(None))
    _, results, call = records[-1]
    assert call["constant_params"] == (
        128,
        256,
        dtype.float64,
        dtype.uint64,
        "f64",
        "u64",
    )
    assert call["inputs"][2].get_type().dtype == dtype.uint64
    assert [(result.dtype, result.shape) for result in results[:-1]] == (
        [(dtype.float64, (256,)), (dtype.uint64, (256,))]
        if with_offsets
        else [(dtype.float64, (256,))]
    )


@pytest.mark.parametrize(
    "control,value,message",
    [
        ("decoded_items", True, "decoded_items must be an int"),
        ("decoded_items", 0, "extents must be positive"),
        ("decoded_items", 2147483648, "fit in int"),
        ("decoded_items", 16, "multiples of 32"),
        ("decoded_items", 32.0, "decoded_items must be an int"),
        ("decoded_window_offset", -1, "nonnegative"),
        ("decoded_window_offset", 1 << 32, "representable"),
        ("decoded_window_offset", 1.5, "must be an integer"),
        ("decoded_window_offset", True, "must be an integer"),
        ("methods", ("algorithm", "raking"), "does not support methods"),
    ],
)
def test_run_length_decode_rejects_invalid_constant_controls(
    monkeypatch, tmp_path, control, value, message
):
    lowerings, dtype, _ = _make_provider_backed_lowerings(monkeypatch, tmp_path)
    values = _Var("values", _TileType(dtype.int32, (128,)))
    lengths = _Var("lengths", _TileType(dtype.uint32, (128,)))
    args = {
        "decoded_items": _Const(256),
        "decoded_window_offset": _Const(0),
        "methods": _Const(None),
    }
    args[control] = _Const(value)
    with pytest.raises(TypeError, match=message):
        lowerings.run_length_decode_impl(values, lengths, **args)


@pytest.mark.parametrize(
    "shape,dtype_name,message",
    [
        ((64,), "uint32", "matching extents"),
        ((8, 16), "uint32", "1D run tiles"),
        ((128,), "int32", "length dtype"),
        ((128,), "float32", "length dtype"),
    ],
)
def test_run_length_decode_rejects_invalid_lengths(
    monkeypatch, tmp_path, shape, dtype_name, message
):
    lowerings, dtype, _ = _make_provider_backed_lowerings(monkeypatch, tmp_path)
    values = _Var("values", _TileType(dtype.int32, (128,)))
    lengths = _Var("lengths", _TileType(getattr(dtype, dtype_name), shape))
    with pytest.raises(TypeError, match=message):
        lowerings.run_length_decode_impl(
            values, lengths, _Const(256), _Const(0), _Const(None)
        )


@pytest.mark.parametrize("bins", [True, 8, 33, 96, 512])
def test_histogram_rejects_unsupported_bin_controls(
    monkeypatch, tmp_path, bins
):
    lowerings, dtype, _ = _make_provider_backed_lowerings(monkeypatch, tmp_path)
    samples = _Var("samples", _TileType(dtype.uint8, (128,)))
    with pytest.raises(TypeError, match="bins must be"):
        lowerings.histogram_impl(samples, _Const(bins), _Const(None))
