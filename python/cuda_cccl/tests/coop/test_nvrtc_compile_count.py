import types

from cuda.coop import _nvrtc, _types


class _DummyNvrtc:
    __file__ = "dummy_nvrtc"

    class nvrtcResult:
        NVRTC_SUCCESS = 0

    @staticmethod
    def nvrtcVersion():
        return (0, 12, 0)

    @staticmethod
    def nvrtcCreateProgram(*args, **kwargs):
        return (0, object())

    @staticmethod
    def nvrtcCompileProgram(*args, **kwargs):
        return (0,)

    @staticmethod
    def nvrtcGetLTOIRSize(*args, **kwargs):
        return (0, 1)

    @staticmethod
    def nvrtcGetLTOIR(*args, **kwargs):
        return (0,)

    @staticmethod
    def nvrtcGetPTXSize(*args, **kwargs):
        return (0, 1)

    @staticmethod
    def nvrtcGetPTX(*args, **kwargs):
        return (0,)

    @staticmethod
    def nvrtcDestroyProgram(*args, **kwargs):
        return (0,)

    @staticmethod
    def nvrtcGetProgramLogSize(*args, **kwargs):
        return (0, 0)

    @staticmethod
    def nvrtcGetProgramLog(*args, **kwargs):
        return (0,)


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


class _DummyPrimitive:
    def __init__(self):
        self.is_child = False


class _DummyAlgo:
    def __init__(self, name_prefix):
        self.includes = []
        self.type_definitions = []
        self.parameters = [[]]
        self.primitive = _DummyPrimitive()
        self.names = types.SimpleNamespace(
            temp_storage_bytes=f"{name_prefix}_temp_storage_bytes",
            temp_storage_alignment=f"{name_prefix}_temp_storage_alignment",
            algorithm_struct_size=f"{name_prefix}_struct_size",
            algorithm_struct_alignment=f"{name_prefix}_struct_alignment",
        )
        self.source_code = "#include <cuda/std/cstdint>\nusing dummy_t = int;\n"

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


def _install_nvrtc_stub(monkeypatch):
    monkeypatch.setattr(_nvrtc, "nvrtc", _DummyNvrtc)
    _nvrtc.compile_impl.cache_clear()
    _nvrtc.reset_compile_counter()
    _nvrtc._set_compile_counter_enabled(True)


def test_nvrtc_compile_counter_counts_cache_misses(monkeypatch):
    _install_nvrtc_stub(monkeypatch)

    _nvrtc.compile(cpp="x", cc=80, rdc=True, code="lto")
    _nvrtc.compile(cpp="x", cc=80, rdc=True, code="lto")

    assert _nvrtc.get_compile_counter() == 1

    _nvrtc.compile(cpp="y", cc=80, rdc=True, code="lto")
    assert _nvrtc.get_compile_counter() == 2


def test_bundle_uses_single_nvrtc_compile(monkeypatch):
    _install_nvrtc_stub(monkeypatch)

    monkeypatch.setattr(_types.cuda, "get_current_device", lambda: _DummyDevice())
    monkeypatch.setattr(_types, "ObjectCode", _DummyObjectCode)
    monkeypatch.setattr(_types, "Linker", _DummyLinker)
    monkeypatch.setattr(_types, "LinkerOptions", _DummyLinkerOptions)
    monkeypatch.setattr(_types, "_get_source_code_rewriter", lambda: None)

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

    algo_a = _DummyAlgo("algo_a")
    algo_b = _DummyAlgo("algo_b")

    _types.prepare_ltoir_bundle([algo_a, algo_b], bundle_name="bundle_count")

    assert _nvrtc.get_compile_counter() == 1


def test_nvrtc_dump_sources(tmp_path, monkeypatch):
    _install_nvrtc_stub(monkeypatch)
    monkeypatch.setenv("NUMBA_CCCL_COOP_NVRTC_DUMP_DIR", str(tmp_path))

    _nvrtc.compile(cpp="x", cc=80, rdc=True, code="lto")

    files = list(tmp_path.iterdir())
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert content == "x"
