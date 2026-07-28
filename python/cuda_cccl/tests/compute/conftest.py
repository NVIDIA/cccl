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
    """Runs after pytest collects the tests. Makes a test marked no_numba fail
    if it imports numba, and skips a test marked serialization when running on
    the v2 (HostJIT) backend."""
    serialization_skip = pytest.mark.skip(
        reason="serialization not supported on v2 (HostJIT) backend"
    )

    # Tests marked no_numba must not import numba. We enforce that by attaching
    # the raise_on_numba_import fixture defined above to each one; it raises if
    # numba is imported.
    #
    # We skip attaching it during a real pytest-run-parallel sweep of more than
    # one thread: the fixture uses monkeypatch, which pytest-run-parallel
    # serializes as thread-unsafe, so attaching it to every no_numba test would
    # make the whole sweep run serially and defeat its purpose. A single-threaded
    # run has no sweep to protect, so we attach it and keep the check.
    #
    # config.getoption gives the --parallel-threads value as an int for the
    # default but a str when passed on the command line; normalize to str and
    # count the run as parallel only for an explicit number > 1:
    #
    #   pytest ...                          -> 1       single-threaded, attach
    #   pytest --parallel-threads=1 ...     -> "1"     single-threaded, attach
    #   pytest --parallel-threads=8 ...     -> "8"     parallel, skip (CI sweep)
    #   pytest --parallel-threads=auto ...  -> "auto"  single-threaded, attach
    #
    # "auto" counts as single-threaded on purpose: it can resolve to one CPU, so
    # keeping the check is safer than dropping it on a non-parallel run. isdigit
    # also stops int() from raising on the non-numeric "auto".
    parallel_threads = str(config.getoption("parallel_threads", 1))
    running_parallel = parallel_threads.isdigit() and int(parallel_threads) > 1
    for item in items:
        # no_numba: add raise_on_numba_import unless we skip it for the sweep
        if item.get_closest_marker("no_numba") and not running_parallel:
            if "raise_on_numba_import" not in item.fixturenames:
                item.fixturenames.append("raise_on_numba_import")
        # serialization is unsupported on v2 (HostJIT); skip those tests there
        if USING_V2 and item.get_closest_marker("serialization"):
            item.add_marker(serialization_skip)
