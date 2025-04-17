import pytest


@pytest.fixture(params=[True, False])
def build_only(request):
    return request.param


@pytest.fixture(params=[10_000, 100_000, 1_000_000])
def size(request):
    return request.param


@pytest.fixture
def compile_benchmark(benchmark):
    def run_compile_benchmark(algorithm, function):
        def setup():
            # This function is called once before the benchmark runs
            # to set up the environment.
            algorithm.cache_clear()

        benchmark.pedantic(
            function,
            rounds=3,
            iterations=1,
            setup=setup,
        )

    return run_compile_benchmark
