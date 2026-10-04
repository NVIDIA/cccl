# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check scratch-layout metadata alongside compiled provider code.

NVRTC name expressions encode C++ size and alignment in a lowered name.
Synthetic names test decoding without a compiler. Separate compiler stubs
check that cache entries retain the layout facts and LTO IR, the code used
for link-time optimization. One NVRTC program must produce both before it
is destroyed.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass")

from cuda.coop.cutlass._compiler import _bundle, _cache, _layout, _nvrtc
from cuda.coop.cutlass._compiler._types import ScratchLayout, ScratchLayoutProbe

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


def test_probes_deduplicate_expressions_and_ignore_requirement_key_identity():
    probes = [
        ScratchLayoutProbe("first", "sizeof(Storage)", "alignof(Storage)"),
        ScratchLayoutProbe("second", "sizeof(Storage)", "alignof(Storage)"),
    ]
    prepared = _layout._prepare_layout_probes("provider", probes)
    reordered = _layout._prepare_layout_probes("provider", probes[::-1])
    assert prepared.source == reordered.source
    assert prepared.expressions == reordered.expressions
    assert len(prepared.expressions) == 1
    assert (
        prepared.key_to_expression["first"]
        == prepared.key_to_expression["second"]
    )
    assert prepared.source.count("template <unsigned long long") == 1
    assert "sizeof(Storage)" in prepared.expressions[0]


@pytest.mark.parametrize(
    "probes, exception, message",
    [
        ([object()], TypeError, "ScratchLayoutProbe"),
        ([ScratchLayoutProbe([], "16", "4")], TypeError, "hashable"),
        ([ScratchLayoutProbe("x", " ", "4")], ValueError, "non-empty"),
        (
            [
                ScratchLayoutProbe("x", "16", "4"),
                ScratchLayoutProbe("x", "32", "4"),
            ],
            ValueError,
            "conflicting",
        ),
    ],
)
def test_invalid_probes_fail_before_compilation(probes, exception, message):
    with pytest.raises(exception, match=message):
        _layout._prepare_layout_probes("provider", probes)


def test_lowered_name_recovers_exact_storage_layout():
    prepared = _layout._prepare_layout_probes(
        "provider",
        [ScratchLayoutProbe("x", "sizeof(Storage)", "alignof(Storage)")],
    )
    name = f"_Z{len(prepared.symbol)}{prepared.symbol}ILy1040ELy16EE\0".encode()
    result = _layout._decode_layout_probe_name(
        name, symbol=prepared.symbol, expression=prepared.expressions[0]
    )
    assert result == ScratchLayout(1040, 16)
    with pytest.raises(ValueError, match="unexpected lowered"):
        _layout._decode_layout_probe_name(
            name, symbol="other", expression=prepared.expressions[0]
        )


@pytest.mark.parametrize(
    "size, alignment", [(0, 4), (True, 4), (16, False), (16, 3), (17, 4)]
)
def test_invalid_layout(size, alignment):
    with pytest.raises(ValueError, match="Invalid storage layout"):
        _layout._validate_storage_layout(size, alignment, description="Storage")


@pytest.fixture
def compilation(monkeypatch, tmp_path):
    """Isolate the artifact cache and record simulated provider compilations.

    Fixed compiler identities keep cache keys stable. The stub returns layout
    facts for each requested expression with synthetic LTO bytes. Its call
    list distinguishes cache hits from recompilation without invoking NVRTC.
    """

    monkeypatch.setenv(_cache.CACHE_DIR_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(_cache, "_SOURCE_CACHE", {})
    monkeypatch.setattr(_cache, "_MANAGED_BUNDLE_PATHS", set())
    context = _nvrtc.CompileContext(
        ("/headers",),
        "identity",
        "/toolkit",
        (13, 3),
        "/nvrtc",
        "/builtins",
        (13, 3),
        "/nvjitlink",
        (13, 3),
    )
    monkeypatch.setattr(
        _nvrtc, "resolve_compile_context", lambda headers: context
    )
    calls = []

    def compile_source(prepared, options):
        calls.append(prepared)
        layouts = {
            expression: ScratchLayout(1040, 16)
            for expression in prepared.expressions
        }
        return b"test-lto:" + prepared.source.encode(), layouts

    monkeypatch.setattr(_nvrtc, "compile_ltoir_with_layouts", compile_source)
    return calls


def _compile(key="storage", size="sizeof(Storage)"):
    return _bundle.compile_bundle_source_with_layouts(
        "provider",
        arch="compute_120",
        required_headers=(),
        layout_probes=[ScratchLayoutProbe(key, size, "alignof(Storage)")],
    )


def test_layouts_survive_memory_and_disk_hits_with_different_requirement_keys(
    compilation,
):
    """Reuse native layout facts under a different caller requirement key.

    The C++ expression determines the artifact; the caller's key only selects
    where to return its layout. Clearing the memory cache then checks that the
    same facts survive a disk-cache hit. Changing the expression must compile
    a separate artifact.
    """

    initial = _compile()
    assert initial.layouts == {"storage": ScratchLayout(1040, 16)}
    renamed = _compile(key=("different", "key"))
    assert renamed.path == initial.path
    assert renamed.layouts == {("different", "key"): ScratchLayout(1040, 16)}
    _cache._SOURCE_CACHE.clear()
    assert _compile() == initial
    assert len(compilation) == 1
    assert _compile(size="sizeof(OtherStorage)").path != initial.path
    assert len(compilation) == 2


@pytest.mark.parametrize("damage", ("missing", "invalid", "wrong-expression"))
def test_incomplete_layout_cache_is_recompiled(compilation, damage):
    initial = _compile()
    metadata_path = Path(initial.path + ".json")
    metadata = json.loads(metadata_path.read_text())
    if damage == "missing":
        del metadata["layouts"]
    elif damage == "invalid":
        next(iter(metadata["layouts"].values()))["alignment"] = 3
    else:
        metadata["layouts"] = {
            "unrelated": {"size_in_bytes": 1040, "alignment": 16}
        }
    metadata_path.write_text(json.dumps(metadata))
    _cache._SOURCE_CACHE.clear()
    assert _compile() == initial
    assert len(compilation) == 2


def test_failed_probe_does_not_publish_an_artifact(compilation, monkeypatch):
    """Keep unusable LTO out of the cache when its layout query fails.

    A provider that needs scratch is incomplete without size and alignment.
    After the injected failure, restoring compilation must produce a usable
    artifact with its layout metadata.
    """

    compile_source = _nvrtc.compile_ltoir_with_layouts

    def fail(*args):
        raise RuntimeError("layout query failed")

    monkeypatch.setattr(_nvrtc, "compile_ltoir_with_layouts", fail)
    with pytest.raises(RuntimeError, match="layout query failed"):
        _compile()
    assert not list(Path(_cache.configured_cache_dir()).glob("*.ltoir"))
    monkeypatch.setattr(_nvrtc, "compile_ltoir_with_layouts", compile_source)
    assert _compile().layouts["storage"] == ScratchLayout(1040, 16)


class _NVRTC:
    """Record NVRTC calls through one program's compile and cleanup steps.

    The fake creates a program, registers name expressions, compiles, reads
    lowered names, fetches LTO IR, and destroys the program. Methods return
    CUDA-style result tuples and synthetic output. Failure injection at name
    registration or lookup checks cleanup before and after compilation without
    loading NVRTC.
    """

    nvrtcResult = SimpleNamespace(NVRTC_SUCCESS=0)

    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def nvrtcCreateProgram(self, source, *args):
        self.calls.append("create")
        return 0, "program"

    def nvrtcAddNameExpression(self, program, expression):
        self.calls.append("add")
        return (1 if self.fail_at == "add" else 0,)

    def nvrtcCompileProgram(self, *args):
        self.calls.append("compile")
        return (0,)

    def nvrtcGetLoweredName(self, program, expression):
        self.calls.append("lowered")
        symbol = expression.decode().split("<", 1)[0][1:]
        name = f"_Z{len(symbol)}{symbol}ILy96ELy32EE".encode()
        return (1 if self.fail_at == "lowered" else 0), name

    def nvrtcGetLTOIRSize(self, program):
        self.calls.append("lto-size")
        return 0, 4

    def nvrtcGetLTOIR(self, program, blob):
        self.calls.append("lto")
        blob[:] = b"LTO!"
        return (0,)

    def nvrtcDestroyProgram(self, program):
        self.calls.append("destroy")
        return (0,)


def test_one_nvrtc_program_produces_layouts_and_ltoir(monkeypatch):
    runtime = _NVRTC()
    monkeypatch.setattr(_nvrtc, "_load_nvrtc", lambda: runtime)
    prepared = _layout._prepare_layout_probes(
        "provider", [ScratchLayoutProbe("storage", "96", "32")]
    )
    blob, layouts = _nvrtc.compile_ltoir_with_layouts(prepared, ())
    assert blob == b"LTO!"
    assert layouts == {prepared.expressions[0]: ScratchLayout(96, 32)}
    assert runtime.calls == [
        "create",
        "add",
        "compile",
        "lowered",
        "lto-size",
        "lto",
        "destroy",
    ]


@pytest.mark.parametrize("fail_at", ("add", "lowered"))
def test_nvrtc_probe_failure_destroys_program(monkeypatch, fail_at):
    runtime = _NVRTC(fail_at)
    monkeypatch.setattr(_nvrtc, "_load_nvrtc", lambda: runtime)
    prepared = _layout._prepare_layout_probes(
        "provider", [ScratchLayoutProbe("storage", "96", "32")]
    )
    with pytest.raises(RuntimeError, match="storage layout probe"):
        _nvrtc.compile_ltoir_with_layouts(prepared, ())
    assert runtime.calls[-1] == "destroy"
    assert "lto" not in runtime.calls
