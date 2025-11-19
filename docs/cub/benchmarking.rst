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
    cmake .. --preset=cub-benchmark

You clone the repository, create a build directory and configure the build with CMake.
The preset `cub-benchmark` takes care of everything.

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


.. _cub-benchmarking-running:

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
`NVBench documentation <https://github.com/NVIDIA/nvbench/blob/main/docs/cli_help.md>`__.

The expected terminal output is something along the following lines (also saved to `base.md`),
shortened for brevity:

.. code-block:: bash

    # Log
    Run:  [1/8] base [Device=0 T{ct}=I32 OffsetT{ct}=I32 Elements{io}=2^16]
    Pass: Cold: 0.004571ms GPU, 0.009322ms CPU, 0.00s total GPU, 0.01s total wall, 334x
    Run:  [2/8] base [Device=0 T{ct}=I32 OffsetT{ct}=I32 Elements{io}=2^20]
    Pass: Cold: 0.015161ms GPU, 0.023367ms CPU, 0.01s total GPU, 0.02s total wall, 430x
    ...
    # Benchmark Results
    | T{ct} | OffsetT{ct} |   Elements{io}   | Samples |  CPU Time  |  Noise  |  GPU Time  | Noise  | Elem/s  | GlobalMem BW | BWUtil |
    |-------|-------------|------------------|---------|------------|---------|------------|--------|---------|--------------|--------|
    |   I32 |         I32 |     2^16 = 65536 |    334x |   9.322 us | 104.44% |   4.571 us | 10.87% | 14.337G | 114.696 GB/s | 14.93% |
    |   I32 |         I32 |   2^20 = 1048576 |    430x |  23.367 us | 327.68% |  15.161 us |  3.47% | 69.161G | 553.285 GB/s | 72.03% |
    ...

If you are only interested in a subset of workloads, you can restrict benchmarking as follows:

.. code-block:: bash

    ./bin/cub.bench.adjacent_difference.subtract_left.base ...\
        -a 'T{ct}=I32'\
        -a 'OffsetT{ct}=I32'\
        -a 'Elements{io}[pow2]=[24,28]'\

The `-a` option allows you to restrict the values for each axis available for the benchmark.
See the `NVBench documentation <https://github.com/NVIDIA/nvbench/blob/main/docs/cli_help_axis.md>`__.
for more information on how to specify the axis values.
If the specified axis does not exist, the benchmark will terminate with an error.


.. _cub-benchmarking-comparing:

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
The script will print a Markdown report showing the runtime differences between each variant of the two benchmark run.
This could look like this, again shortened for brevity:

.. code-block:: bash

    |  T{ct}  |  OffsetT{ct}  |  Elements{io}  |   Ref Time |   Ref Noise |   Cmp Time |   Cmp Noise |       Diff |   %Diff |  Status  |
    |---------|---------------|----------------|------------|-------------|------------|-------------|------------|---------|----------|
    |   I32   |      I32      |      2^16      |   4.571 us |      10.87% |   4.096 us |       0.00% |  -0.475 us | -10.39% |   FAIL   |
    |   I32   |      I32      |      2^20      |  15.161 us |       3.47% |  15.143 us |       3.55% |  -0.018 us |  -0.12% |   PASS   |
    ...

In addition to showing the absolute and relative runtime difference,
NVBench reports the noise of the measurements,
which corresponds to the relative standard deviation.
It then reports with statistical significance in the `Status` column
how the runtime changed from the base to the new version.


Running all benchmarks directly from the command line
--------------------------------------------------------------------------------

To get a full snapshot of CUB's performance, you can run all benchmarks and save the results.
For example, inside a build directory you can run:

.. code-block:: bash

    ninja cub.all.benches
    benchmarks=$(ls bin | grep cub.bench); n=$(echo $benchmarks | wc -w); i=1; \
    for b in $benchmarks; do \
      echo "=== Running $b ($i/$n) ==="; \
      ./bin/$b -d 0 --stopping-criterion entropy --json $b.json --md $b.md; \
      ((i++)); \
    done

This will generate one JSON and one Markdown file for each benchmark.
You can archive those files for later comparison or analysis.


Running all benchmarks via tuning scripts (alternative)
--------------------------------------------------------------------------------

The benchmark suite can also be run using the :ref:`tuning <cub-tuning>` infrastructure.
The tuning infrastructure handles building benchmarks itself, because it records the build times.
Therefore, it's critical that you run it in a clean build directory without any build artifacts.
Running cmake is enough. Alternatively, you can also clean your build directory.
Furthermore, the tuning scripts require some additional python dependencies, which you have to install:

.. code-block:: bash

    ninja clean
    pip install --user fpzip pandas scipy

To select the appropriate CUDA GPU, first identify the GPU ID by running `nvidia-smi`, then set the
desired GPU using `export CUDA_VISIBLE_DEVICES=x <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-environment-variables>`_,
where `x` is the ID of the GPU you want to use (e.g., `1`).
This ensures your application uses only the specified GPU.
We can then run the full benchmark suite from the build directory with:

.. code-block:: bash

    export CUDA_VISIBLE_DEVICES=0 # or any other GPU ID
    PYTHONPATH=../benchmarks/scripts ../benchmarks/scripts/run.py

You can expect the output to look like this:

.. code-block:: bash

    &&&& RUNNING bench
    ctk:  12.2.140
    cub:  812ba98d1
    &&&& PERF cub_bench_adjacent_difference_subtract_left_base_T_ct__I32___OffsetT_ct__I32___Elements_io__pow2__16 4.095999884157209e-06 -sec
    &&&& PERF cub_bench_adjacent_difference_subtract_left_base_T_ct__I32___OffsetT_ct__I32___Elements_io__pow2__20 1.2288000107218977e-05 -sec
    &&&& PERF cub_bench_adjacent_difference_subtract_left_base_T_ct__I32___OffsetT_ct__I32___Elements_io__pow2__24 0.00016998399223666638 -sec
    &&&& PERF cub_bench_adjacent_difference_subtract_left_base_T_ct__I32___OffsetT_ct__I32___Elements_io__pow2__28 0.002673664130270481 -sec
    ...

The tuning infrastructure will build and execute all benchmarks and their variants one after each other,
reporting the time in seconds it took to execute the benchmarked region.

It's also possible to benchmark a subset of algorithms and workloads, by running in a build directory:

.. code-block:: bash

    export CUDA_VISIBLE_DEVICES=0 # or any other GPU ID
    PYTHONPATH=../benchmarks/scripts ../benchmarks/scripts/run.py -R '.*scan.exclusive.sum.*' -a 'Elements{io}[pow2]=[24,28]' -a 'T{ct}=I32'
    &&&& RUNNING bench
     ctk:  12.6.77
    cccl:  v2.7.0-rc0-265-g32aa6aa5a
    &&&& PERF cub_bench_scan_exclusive_sum_base_T_ct__I32___OffsetT_ct__U32___Elements_io__pow2__28 0.003194367978721857 -sec
    &&&& PERF cub_bench_scan_exclusive_sum_base_T_ct__I32___OffsetT_ct__U64___Elements_io__pow2__28 0.00319383991882205 -sec
    &&&& PASSED bench


The `-R` option allows you to specify a regular expression for selecting benchmarks.
The `-a` restricts the values for an axis across all benchmarks
See the `NVBench documentation <https://github.com/NVIDIA/nvbench/blob/main/docs/cli_help_axis.md>`__.
for more information on how to specify the axis values.
Contrary to running a benchmark directly,
the tuning infrastructure will just ignore an axis value if a benchmark does not support,
run the benchmark regardless, and continue.

The tuning infrastructure stores results in an SQLite database called :code:`cccl_meta_bench.db` in the build directory.
This database persists across tuning runs.
If you interrupt the benchmark script and then launch it again, only missing benchmark variants will be run.


Comparing results of multiple tuning databases
--------------------------------------------------------------------------------

Benchmark results captured in different tuning databases can be compared as well:

.. code-block:: bash

    <cccl_git_root>/benchmarks/scripts/compare.py -o cccl_meta_bench1.db cccl_meta_bench2.db

This will print a Markdown report showing the runtime differences and noise for each variant.

Furthermore, you can plot the results, which requires additional python packages:

.. code-block:: bash

    pip install fpzip pandas matplotlib seaborn tabulate PyQt5 colorama

You can plot one or more tuning databases as a bar chart or a box plot (add `--box`):

.. code-block:: bash

    <cccl_git_root>/benchmarks/scripts/sol.py cccl_meta_bench.db ...

This is useful to display the current performance of CUB as captured in a single tuning database,
or visually compare the performance of CUB across different tuning databases
(from different points in time, on different GPUs, etc.).


Dumping benchmark results from a tuning database
--------------------------------------------------------------------------------

The resulting database contains all samples, which can be extracted into JSON files:

.. code-block:: bash

    <cccl_git_root>/benchmarks/scripts/analyze.py -o ./cccl_meta_bench.db

This will create a JSON file for each benchmark variant next to the database.
For example:

.. code-block:: bash

    cat cub_bench_scan_exclusive_sum_base_T_ct__I32___OffsetT_ct__U32___Elements_io__pow2__28.json
    [
      {
        "variant": "base ()",
        "elapsed": 2.6299014091,
        "center": 0.003194368,
        "bw": 0.8754671386,
        "samples": [
          0.003152896,
          0.0031549439,
          ...
        ],
        "Elements{io}[pow2]": "28",
        "base_samples": [
          0.003152896,
          0.0031549439,
          ...
        ],
        "speedup": 1
      }
    ]


Profiling benchmarks with Nsight Compute
--------------------------------------------------------------------------------

If you want to see profiling metrics on source code level,
you have to recompile your benchmarks with the `-lineinfo` option.
With cmake, you can just add `-DCMAKE_CUDA_FLAGS=-lineinfo` when invoking cmake in the `build` directory:

.. code-block:: bash

    cmake .. --preset=cub-benchmark -DCMAKE_CUDA_FLAGS=-lineinfo

To profile the kernels, use the `ncu` command.
A typical invocation, if you work on a remote cluster, could look like this:

.. code-block:: bash

    ncu --set full --import-source yes -o base.ncu-rep -f ./bin/thrust.bench.transform.basic.base -d 0 --profile

The option `--set full` instructs `ncu` to collect all metrics.
This requires rerunning some kernels and takes more time.
`--import-source yes` imports the source code into the report file,
so you can see metrics not only in SASS but also in your source code,
even if you copy the resulting report away from the source code.
`-o base.ncu-rep` specifies the output file and `-f` overwrites the output file if it already exists.
`--profile` tells NVBench to run only one iteration, which speeds up profiling.

For inspecting the profiling report, we recommend using the GUI of Nsight Compute.
If you run on a remote machine, you may want to copy the report `base.ncu-rep` back to your local workstation,
before viewing the report using `ncu-ui`:

.. code-block:: bash

    scp <remote hostname>:<cccl repo directory>/build/base.ncu-rep .
    ncu-ui base.ncu-rep

The version of `ncu-ui` needs to be at least as high as the version of `ncu` used to create the report.

Authoring benchmarks
--------------------------------------------------------------------------------

CUB's benchmarks serve a dual purpose.
They are used to measure and compare the performance of CUB and to tune CUB's algorithms.
More information on how to create new benchmarks is provided in the :ref:`CUB tuning guide <cub-tuning>`.
