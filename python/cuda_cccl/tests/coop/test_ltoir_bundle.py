import types
from collections import OrderedDict

import pytest

from cuda.coop import _types
from cuda.coop._rewrite import CoopNodeRewriter


class _DummyPrimitive:
    def __init__(self, is_child=False):
        self.is_child = is_child


class _DummyAlgo:
    def __init__(self, name_prefix, extra_ltoir):
        self.includes = ["cub/block/block_reduce.cuh"]
        self.type_definitions = [
            types.SimpleNamespace(
                code="struct DummyType { int x; };", lto_irs=[extra_ltoir]
            )
        ]
        self.parameters = [[]]
        self.primitive = _DummyPrimitive(is_child=False)
        self.names = types.SimpleNamespace(
            temp_storage_bytes=f"{name_prefix}_temp_storage_bytes",
            temp_storage_alignment=f"{name_prefix}_temp_storage_alignment",
            algorithm_struct_size=f"{name_prefix}_struct_size",
            algorithm_struct_alignment=f"{name_prefix}_struct_alignment",
        )
        self.source_code = (
            "#include <cuda/std/cstdint>\n"
            "#include <cub/block/block_reduce.cuh>\n"
            "struct DummyType { int x; };\n\n"
            "using dummy_t = int;\n"
        )

    @property
    def temp_storage_bytes(self):
        return self._temp_storage_bytes

    @property
    def temp_storage_alignment(self):
        return self._temp_storage_alignment

    @property
    def algorithm_struct_size(self):
        return self._algorithm_struct_size

    @property
    def algorithm_struct_alignment(self):
        return self._algorithm_struct_alignment


class _DummyDevice:
    compute_capability = (8, 0)


class _DummyObjectCode:
    def __init__(self, name):
        self.name = name

    @classmethod
    def from_ltoir(cls, blob, name):
        return cls(name)


class _DummyLinker:
    def __init__(self, obj, options=None):
        self.obj = obj
        self.options = options

    def link(self, kind):
        return types.SimpleNamespace(code=self._ptx.encode("utf-8"))


class _DummyLinkerOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture()
def _bundle_stubs(monkeypatch):
    def fake_compile(**kwargs):
        return 0, b"fake_lto"

    monkeypatch.setattr(_types.nvrtc, "compile", fake_compile)
    monkeypatch.setattr(_types.cuda, "get_current_device", lambda: _DummyDevice())
    monkeypatch.setattr(_types, "ObjectCode", _DummyObjectCode)
    monkeypatch.setattr(_types, "Linker", _DummyLinker)
    monkeypatch.setattr(_types, "LinkerOptions", _DummyLinkerOptions)
    monkeypatch.setattr(_types, "_get_source_code_rewriter", lambda: None)


def test_prepare_ltoir_bundle_seeds_algo_cache(_bundle_stubs, monkeypatch):
    lto_a = _types.LTOIR(name="extra_a", data=b"a")
    lto_b = _types.LTOIR(name="extra_b", data=b"b")
    algo_a = _DummyAlgo("algo_a", lto_a)
    algo_b = _DummyAlgo("algo_b", lto_b)

    ptx = (
        ".global .align 4 .u32 algo_a_temp_storage_bytes = 64;\n"
        ".global .align 4 .u32 algo_a_temp_storage_alignment = 8;\n"
        ".global .align 4 .u32 algo_a_struct_size = 16;\n"
        ".global .align 4 .u32 algo_a_struct_alignment = 4;\n"
        ".global .align 4 .u32 algo_b_temp_storage_bytes = 96;\n"
        ".global .align 4 .u32 algo_b_temp_storage_alignment = 16;\n"
        ".global .align 4 .u32 algo_b_struct_size = 32;\n"
        ".global .align 4 .u32 algo_b_struct_alignment = 8;\n"
    )

    monkeypatch.setattr(_DummyLinker, "_ptx", ptx, raising=False)

    bundle = _types.prepare_ltoir_bundle([algo_a, algo_b], bundle_name="bundle_test")

    assert bundle is not None
    assert bundle.name == "bundle_test"

    for algo, extra in ((algo_a, lto_a), (algo_b, lto_b)):
        assert "lto_irs" in algo.__dict__
        lto_irs = algo.__dict__["lto_irs"]
        assert lto_irs[0] is bundle
        assert extra in lto_irs

    assert algo_a.temp_storage_bytes == 64
    assert algo_a.temp_storage_alignment == 8
    assert algo_a.algorithm_struct_size == 16
    assert algo_a.algorithm_struct_alignment == 4

    assert algo_b.temp_storage_bytes == 96
    assert algo_b.temp_storage_alignment == 16
    assert algo_b.algorithm_struct_size == 32
    assert algo_b.algorithm_struct_alignment == 8


def _make_dummy_nodes():
    algo_a = types.SimpleNamespace()
    algo_b = types.SimpleNamespace()

    inst_a = types.SimpleNamespace(specialization=algo_a)
    inst_b = types.SimpleNamespace(specialization=algo_b)

    node_a = types.SimpleNamespace(instance=inst_a)
    node_b = types.SimpleNamespace(instance=inst_b)

    return OrderedDict([("a", node_a), ("b", node_b)])


def test_env_var_gates_ltoir_bundle(monkeypatch):
    called = {"count": 0}

    def fake_bundle(algorithms, bundle_name=None):
        called["count"] += 1
        return object()

    monkeypatch.setattr(_types, "prepare_ltoir_bundle", fake_bundle)

    monkeypatch.setenv("NUMBA_CCCL_COOP_BUNDLE_LTOIR", "1")
    rewriter = CoopNodeRewriter(types.SimpleNamespace(typingctx=object()))
    rewriter.nodes = _make_dummy_nodes()
    rewriter.ensure_ltoir_bundle()

    assert called["count"] == 1

    monkeypatch.delenv("NUMBA_CCCL_COOP_BUNDLE_LTOIR", raising=False)
    rewriter2 = CoopNodeRewriter(types.SimpleNamespace(typingctx=object()))
    rewriter2.nodes = _make_dummy_nodes()
    rewriter2.ensure_ltoir_bundle()

    assert called["count"] == 1
