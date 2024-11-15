CUB Benchmarks
*************************************
.. TODO(bgruber): this guide applies to Thrust as well. We should rename it to "CCCL Benchmarks" and move it out of CUB

CUB comes with a set of `NVBench <https://github.com/NVIDIA/nvbench>`_-based benchmarks for its algorithms,
which can be used to measure the performance of CUB on your system on a variety of workloads.
The integration with NVBench allows to archive and compare benchmark results,
which is useful for continuous performance testing, detecting regressions, tuning, and optimization.
This guide gives an introduction into CUB's benchmarking infrastructure.

Building benchmarks
--------------------------------------------------------------------------------

CUB benchmarks are build as part of the CCCL CMake infrastructure.
Starting from scratch:

.. code-block:: bash

    git clone https://github.com/NVIDIA/cccl.git
    cd cccl
    mkdir build
    cd build
    cmake ..\
        -GNinja\
        -DCCCL_ENABLE_BENCHMARKS=YES\
        -DCCCL_ENABLE_CUB=YES\
        -DCCCL_ENABLE_THRUST=NO\
        -DCCCL_ENABLE_LIBCUDACXX=NO\
        -DCUB_ENABLE_RDC_TESTS=NO\
        -DCMAKE_BUILD_TYPE=Release\
        -DCMAKE_CUDA_ARCHITECTURES=90 # TODO: Set your GPU architecture

You clone the repository, create a build directory and configure the build with CMake.
It's important that you enable benchmarks (`CCCL_ENABLE_BENCHMARKS=ON`),
build in Release mode (`CMAKE_BUILD_TYPE=Release`),
and set the GPU architecture to match your system (`CMAKE_CUDA_ARCHITECTURES=XX`).
This <website `https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/`>_
contains a great table listing the architectures for different brands of GPUs.
.. TODO(bgruber): do we have a public NVIDIA maintained table I can link here instead?
We use Ninja as CMake generator in this guide, but you can use any other generator you prefer.

You can then proceed to build the benchmarks.

You can list the available cmake build targets with, if you intend to only build selected benchmarks:

.. code-block:: bash

    ninja -t targets | grep '\.bench\.'
    cub.bench.adjacent_difference.subtract_left.base: phony
    cub.bench.copy.memcpy.base: phony
    ...
    cub.bench.transform.babelstream3.base: phony
    cub.bench.transform_reduce.sum.base: phony

We also provide a target to build all benchmarks:

.. code-block:: bash

    ninja cub.all.benches


Running a benchmark
--------------------------------------------------------------------------------

After we built a benchmark, we can run it as follows:

.. code-block:: bash

    ./bin/cub.bench.adjacent_difference.subtract_left.base\
        -d 0\
        --stopping-criterion entropy\
        --json base.json\
        --md base.md

In this command, `-d 0` indicates that we want to run on GPU 0 on our system.
Setting `--stopping-criterion entropy` is advisable since it reduces runtime
and increase confidence in the resulting data.
It's not set as default yet, because NVBench is still evaluating it.
By default, NVBench will print the benchmark results to the terminal as Markdown.
`--json base.json` will save the detailed results in a JSON file as well for later use.
`--md base.md` will save the Markdown output to a file as well,
so you can easily view the results later without having to parse the JSON.
More information on what command line options are available can be found in the
`NVBench documentation <https://github.com/NVIDIA/nvbench/blob/main/docs/cli_help.md>`_.

The expected terminal output is something along the following lines (also saved to `base.md`):

.. code-block:: bash

    TODO

If you are only interested in a subset of workloads, you can restrict benchmarking as follows:

.. code-block:: bash

    ./bin/cub.bench.adjacent_difference.subtract_left.base ...\
        -a 'T{ct}=I32'\
        -a 'OffsetT{ct}=I32'\
        -a 'Elements{io}[pow2]=[24,28]'\

The `-a` option allows you to restrict the values for each axis available for the benchmark.
See the `NVBench documentation <https://github.com/NVIDIA/nvbench/blob/main/docs/cli_help_axis.md>`_.
for more information on how to specify the axis values.


Comparing benchmark results
--------------------------------------------------------------------------------

Let's say you have a modification that you'd like to benchmark.
To compare the performance you have to build and run the benchmark as described above for the unmodified code,
saving the results to a JSON file, e.g. `base.json`.
Then, you apply your code changes (e.g., switch to a different branch, git stash pop, apply a patch file, etc.),
rebuild and rerun the benchmark, saving the results to a different JSON file, e.g. `new.json`.

You can now compare the two result JSON files using, assuming you are still in your build directory:

.. code-block:: bash

    PYTHONPATH=./_deps/nvbench-src/scripts ./_deps/nvbench-src/scripts/nvbench_compare.py base.json new.json

The `PYTHONPATH` environment variable may not be necessary in all cases.
The script will print a Markdown report, showing the runtime differences between each variant of the two benchmark run:

.. code-block:: bash

    TODO

In addition to showing the absolute and relative runtime difference,
NVBench reports the noise of the measurements,
which corresponds to the relative standard deviation.
It then reports with statistical significance in the `Status` column
how the runtime changed from the base to the new version.


Running all benchmarks
--------------------------------------------------------------------------------

This file contains instructions on how to run all CUB benchmarks using CUB tuning infrastructure.

.. code-block:: bash

    pip install --user fpzip pandas scipy
    ../benchmarks/scripts/run.py


Expected output for the command above is:


.. code-block:: bash

    ../benchmarks/scripts/run.py
    &&&& RUNNING bench
    ctk:  12.2.140
    cub:  812ba98d1
    &&&& PERF cub_bench_adjacent_difference_subtract_left_base_T_ct__I32___OffsetT_ct__I32___Elements_io__pow2__16 4.095999884157209e-06 -sec
    &&&& PERF cub_bench_adjacent_difference_subtract_left_base_T_ct__I32___OffsetT_ct__I32___Elements_io__pow2__20 1.2288000107218977e-05 -sec
    &&&& PERF cub_bench_adjacent_difference_subtract_left_base_T_ct__I32___OffsetT_ct__I32___Elements_io__pow2__24 0.00016998399223666638 -sec
    &&&& PERF cub_bench_adjacent_difference_subtract_left_base_T_ct__I32___OffsetT_ct__I32___Elements_io__pow2__28 0.002673664130270481 -sec
    ...


It's also possible to benchmark a subset of algorithms and workloads:

.. code-block:: bash

    ../benchmarks/scripts/run.py -R '.*scan.exclusive.sum.*' -a 'Elements{io}[pow2]=[24,28]' -a 'T{ct}=I32'
    &&&& RUNNING bench
    ctk:  12.2.140
    cub:  812ba98d1
    &&&& PERF cub_bench_scan_exclusive_sum_base_T_ct__I32___OffsetT_ct__I32___Elements_io__pow2__24 0.00016899200272746384 -sec
    &&&& PERF cub_bench_scan_exclusive_sum_base_T_ct__I32___OffsetT_ct__I32___Elements_io__pow2__28 0.002696000039577484 -sec
    &&&& PASSED bench
