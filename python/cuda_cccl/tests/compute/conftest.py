import builtins
from collections.abc import Generator

import numpy as np
import pytest

from cuda.core import Device, Stream

try:
    from cuda.compute._build_info import USING_V2
except ImportError:
    USING_V2 = False

check_ldl_stl_in_sass = False


# Define a pytest fixture that returns random arrays with different dtypes
@pytest.fixture(
    params=[
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ]
)
def input_array(request):
    dtype = np.dtype(request.param)
    sample_size = 1000

    # Generate random values based on the dtype
    if np.issubdtype(dtype, np.integer):
        is_unsigned = dtype.kind == "u"
        # For integer types, use np.random.randint for random integers
        if is_unsigned:
            low_inclusive, high_exclusive = 0, 8
        else:
            low_inclusive, high_exclusive = -5, 6
        array = np.random.randint(
            low=low_inclusive, high=high_exclusive, size=sample_size, dtype=dtype
        )
    elif np.issubdtype(dtype, np.floating):
        # For floating-point types, use np.random.random and cast to the required dtype
        array = np.random.random(sample_size).astype(dtype)
    elif np.issubdtype(dtype, np.complexfloating):
        # For complex types, generate random real and imaginary parts
        packed = np.random.random(2 * sample_size)
        real_part = packed[:sample_size]
        imag_part = packed[sample_size:]
        array = (real_part + 1j * imag_part).astype(dtype)

    return array


# Define a pytest fixture that returns random floating-point arrays only
@pytest.fixture(
    params=[
        np.float32,
        np.float64,
    ]
)
def floating_array(request):
    dtype = np.dtype(request.param)
    sample_size = 1000

    # Generate random floating-point values
    array = np.random.random(sample_size).astype(dtype)
    return array


@pytest.fixture(scope="function")
def cuda_stream() -> Generator[Stream, None, None]:
    device = Device()
    device.set_current()
    stream = device.create_stream()
    try:
        yield stream
    finally:
        stream.close()


@pytest.fixture(scope="function", autouse=True)
def verify_sass(request):
    if request.node.get_closest_marker("no_verify_sass"):
        return

    if not check_ldl_stl_in_sass:
        return

    # Pull monkeypatch dynamically rather than as a fixture parameter so this
    # autouse fixture does not add monkeypatch to every test's static fixture
    # closure. pytest-run-parallel treats monkeypatch as thread-unsafe based on
    # that closure, so a parameter here would serialize the entire free-threaded
    # parallel sweep -- even though this fixture only patches on the opt-in
    # SASS-check path (check_ldl_stl_in_sass, off by default and in CI).
    monkeypatch = request.getfixturevalue("monkeypatch")

    import cuda.compute._cccl_interop

    monkeypatch.setattr(
        cuda.compute._cccl_interop,
        "_check_sass",
        True,
    )


@pytest.fixture
def raise_on_numba_import(monkeypatch):
    """This fixture will raise if a test attempts to import numba"""
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "numba" or name.startswith("numba."):
            raise ModuleNotFoundError(
                "This test is marked 'no_numba' but attempted to import it"
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def pytest_collection_modifyitems(config, items):
    serialization_skip = pytest.mark.skip(
        reason="serialization not supported on v2 (HostJIT) backend"
    )
    # Under the pytest-run-parallel sweep (--parallel-threads > 1) skip the
    # blanket raise_on_numba_import injection: it monkeypatches
    # builtins.__import__, which pytest-run-parallel treats as thread-unsafe and
    # would serialize every no_numba test, neutering the sweep.
    #
    # getoption returns the int 1 from the argparse default but a *string* for
    # CLI-passed values ("2", "auto", "1"), so normalize to str before comparing
    # -- a bare `> 1` would raise TypeError on the "2" string.
    running_parallel = str(config.getoption("parallel_threads", 1)) != "1"
    for item in items:
        if item.get_closest_marker("no_numba") and not running_parallel:
            if "raise_on_numba_import" not in item.fixturenames:
                item.fixturenames.append("raise_on_numba_import")
        if USING_V2 and item.get_closest_marker("serialization"):
            item.add_marker(serialization_skip)
