.. _cub-tuning:

CUB Tuning Infrastructure
================================================================================

Device-scope algorithms in CUB have many knobs that do not affect the algorithms' correctness but can significantly impact performance. For instance, the number of threads per block and items per thread can be tuned to maximize performance for a given device and data type.
This document describes CUB's tuning Infrastructure, a set of tools facilitating the process of
selecting optimal tuning parameters for a given device and data type.

Definitions
--------------------------------------------------------------------------------

We omit the word "tuning" but assume it in the definitions for all terms below,
so those terms may mean something else in a more generic context.

Algorithms are tuned for different workloads, which are sub spaces of all benchmark versions defined by NVBench via a benchmark's axes.
For instance, radix sort can be tuned for different key types, different number of keys, and different distributions of keys.
We partition the space spanned by a benchmark's axes into two categories:

* **Compile-time (ct) Workload** - a workload that can be recognized at compile time. For instance, the combination of key type and offset type is a compile-time workload for radix sort. A compile-time workload is a point in the space spanned by the Cartesian product of compile-time type axes of NVBench.

* **Runtime (rt) Workload** - a workload that can be recognized only at runtime. For instance, the number of keys along with their distribution is a runtime workload for radix sort. A runtime workload is a point in the space spanned by the Cartesian product of non-compile-time type axes of NVBench.

The tuning infrastructure can optimize algorithms only for specific compile-time workloads,
aggregating results across all runtime workloads:
It searches through a space of parameters to find the combination for a given compile-time workload with the highest score:

* **Parameter** - a parameter that can be tuned to maximize performance for a given device and data type. For instance, the number of threads per block and items per thread are tuning parameters.

* **Parameter Space** - the set of all possible values for a given tuning parameter. Parameter Space is specific to algorithm. For instance, the parameter space for the number of threads per block is :math:`\{32, 64, 96, 128, \dots, 1024\}` for radix sort, but :math:`\{32, 64, 128, 256, 512\}` for merge sort.

* **Parameter Point** - a concrete value of a tuning parameter. For instance, the parameter point for the number of threads per block is :math:`threads\_per\_block=128`.

* **Search Space** - Cartesian product of parameter spaces. For instance, search space for an algorithm with tunable items per thread and threads per block might look like :math:`\{(ipt \times tpb) | ipt \in \{1, \dots, 25\} \text{and} tpb \in \{32, 64, 96, 128, \dots, 1024\}\}`.

* **Variant** - a point in the corresponding search space.

* **Base** - the variant that CUB uses by default.

* **Score** - a single number representing the performance for a given compile-time workload across all runtime workloads. For instance, a weighted-sum of speedups of a given variant compared to its base for all runtime workloads is a score.

* **Search** - a process consisting of covering all variants for all compile-time workloads to find a variant with maximal score.


Authoring Benchmarks
--------------------------------------------------------------------------------

CUB benchmarks are split into multiple files based on the algorithm they are testing
and potentially further into compile-time flavors that are tuned for individually
(e.g.: sorting only keys vs. key-value pairs, or reducing using sum vs. using min).
The name of the directory represents the name of the algorithm.
The filename corresponds on the flavor.
For instance, the benchmark :code:`benchmarks/bench/radix_sort/keys.cu` tests the radix sort implementation sorting only keys.0
The file name is going to be transformed into :code:`cub.bench.radix_sort.keys...`,
which is the benchmark name reported by the infrastructure.

Benchmarks are based on NVBench.
You start writing a benchmark by including :code:`nvbench_helper.cuh`. It contains all
necessary includes and definitions.

.. code:: c++

  #include <nvbench_helper.cuh>

The next step is to define a search space. The search space is represented by a number of C++ comments.
The format consists of the :code:`%RANGE%` keyword, a parameter macro, a short parameter name, and a range.
The range is represented by three numbers: :code:`start:end:step`.
Start and end are included.
For instance, the following code defines a search space for two parameters, the number of threads per block and items per thread.

.. code:: c++

  // %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
  // %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32

Next, you need to define a benchmark function. The function accepts :code:`nvbench::state &state` and
a :code:`nvbench::type_list`. For more details on the benchmark signature, take a look at the
`NVBench documentation <https://github.com/NVIDIA/nvbench>`_.

.. code:: c++

  template <typename T, typename OffsetT>
  void algname(nvbench::state &state, nvbench::type_list<T, OffsetT>)
  {

Tuning relies on CUB's device algorithms to expose a dispatch layer which can be parameterized by a policy hub.
CUB usually provides a default policy hub, but when tuning we want to overwrite it, so we have to specialize the dispatch layer.
The tuning infrastructure will use the :code:`TUNE_BASE` macro to distinguish between compiling the base version (i.e. baseline) of a benchmark
and compiling a variant for a given set of tuning parameters.
When base is used, no policy is specified, so that the default one CUB provides is used.
If :code:`TUNE_BASE` is not defined, we specify a custom policy
using the parameter macros defined in the :code:`%RANGE%` comments which specify the search space.

.. code:: c++

  #if TUNE_BASE
    using dispatch_t = cub::DispatchReduce<T, OffsetT>; // uses default policy hub
  #else
    template <typename AccumT, typename OffsetT>
    struct policy_hub_t {
      struct MaxPolicy : cub::ChainedPolicy<300, policy_t, policy_t> {
        static constexpr int threads_per_block  = TUNE_THREADS_PER_BLOCK;
        static constexpr int items_per_thread   = TUNE_ITEMS_PER_THREAD;
        ...
      };
    };

    using dispatch_t = cub::DispatchReduce<T, OffsetT, policy_hub_t<accum_t, offset_t>>;
  #endif

The custom policy hub used for tuning should only expose a single :code:`MaxPolicy` so CUB will use it.
It must contain all parameters from the search space.

The :code:`state` passed into the benchmark function allows access to runtime workload axes,
for example the number of elements to process.
When creating containers for the input avoid to initialize data yourself.
Instead, use the :code:`gen` function,
which will fill the input vector with random data on GPU with no compile-time overhead.

.. code:: c++

    const auto elements = static_cast<std::size_t>(state.get_int64("Elements{io}"));
    thrust::device_vector<T> in(elements);
    thrust::device_vector<T> out(1);

    gen(seed_t{}, in);

In addition to benchmark runtime, NVBench can also report information on the achieved memory bandwidth.
For this, you can optionally provide information on the memory reads and writes of the algorithm to the :code:`state`:

.. code:: c++

    state.add_element_count(elements);
    state.add_global_memory_reads<T>(elements, "Size");
    state.add_global_memory_writes<T>(1);

Most CUB algorithms need to be called twice:

1. once to query the amount of temporary storage needed,
2. once to run the actual algorithm.

We perform the first call now and allocate temporary storage:

.. code:: c++

    std::size_t temp_size;
    dispatch_t::Dispatch(nullptr,
                         temp_size,
                         d_in,
                         d_out,
                         static_cast<offset_t>(elements),
                         0 /* stream */);

    thrust::device_vector<char> temp(temp_size);
    auto *temp_storage = thrust::raw_pointer_cast(temp.data());

Finally, we can execute the timed region of the benchmark,
which contains the second call to a CUB algorithm and performs the actual work we want to benchmark:

.. code:: c++

    state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch &launch) {
      dispatch_t::Dispatch(temp_storage,
                           temp_size,
                           d_in,
                           d_out,
                           static_cast<offset_t>(elements),
                           launch.get_stream());
    });
  }

This concludes defining the benchmark function.
Now we need to tell NVBench about it:

.. code:: c++

  NVBENCH_BENCH_TYPES(algname, NVBENCH_TYPE_AXES(all_types, offset_types))
    .set_name("base")
    .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
    .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));

:code:`NVBENCH_BENCH_TYPES` registers the benchmark as one with multiple compile-time workloads,
which are defined by the Cartesian product of the type lists in :code:`NVBENCH_TYPE_AXES`.
:code:`set_name(...)` sets the name of the benchmark.
Only alphabetical characters, numbers and underscores are allowed in the benchmark name.

Furthermore, compile-time axes should be suffixed with :code:`{ct}`. The runtime axes might be optionally annotated
as :code:`{io}` which stands for importance-ordered. This will tell the tuning infrastructure that
the later values on the axis are more important. If the axis is not annotated, each value will be
treated as equally important.

When you define a type axis annotated with :code:`{ct}`, you should consider optimizing
the build time. Many variants are going to be build, but the search is considering one compile-time
use case at a time. This means that if you have many types to tune for, you'll end up having
many template specializations that you don't need. To avoid this, for each compile time axis, the tuning framework will predefine
a `TUNE_AxisName` macro with the type that's currently being tuned. For instance, if you
have the type axes :code:`T{ct}` and :code:`OffsetT` (as shown above), you can use the following
pattern to narrow down the types you compile for:

.. code:: c++

  #ifdef TUNE_T
  using all_types = nvbench::type_list<TUNE_T>;
  #else
  using all_types = nvbench::type_list<char, short, int, long, ...>;
  #endif

  #ifdef TUNE_OffsetT
  using offset_types = nvbench::type_list<TUNE_OffsetT>;
  #else
  using offset_types = nvbench::type_list<int32_t, int64_t>;
  #endif


This logic is already implemented if you use any of the following predefined type lists:

.. list-table:: Predefined type lists
   :header-rows: 1

   * - Axis name
     - C++ identifier
     - Included types
   * - :code:`T{ct}`
     - :code:`integral_types`
     - :code:`int8_t, int16_t, int32_t, int64_t`
   * - :code:`T{ct}`
     - :code:`fundamental_types`
     - :code:`integral_types` and :code:`int128_t, float, double`
   * - :code:`T{ct}`
     - :code:`all_types`
     - :code:`fundamental_types` and :code:`complex`
   * - :code:`OffsetT{ct}`
     - :code:`offset_types`
     - :code:`int32_t, int64_t`


But you are free to define your own axis names and use the logic above for them (see the sort pairs example).

A single benchmark file can define multiple benchmarks (multiple benchmark functions registered with :code:`NVBENCH_BENCH_TYPES`).
All benchmarks in a single file must share the same compile-time axes.
The tuning infrastructure will run all benchmarks in a single file together for the same compile-time workload
and compute a common score across all benchmarks and runtime workloads.
This is useful to tune an algorithm for multiple runtime use cases at once,
that we don't intend to provide separate tuning policies for.


Search Process
--------------------------------------------------------------------------------

To get started with tuning, you need to configure CMake.
You can use the following command:

.. code:: bash

  $ mkdir build
  $ cd build
  $ cmake .. --preset=cub-tune -DCMAKE_CUDA_ARCHITECTURES=90 # TODO: Set your GPU architecture

You can then run the tuning search for a specific algorithm and compile-time workload:

.. code:: bash

  $ ../benchmarks/scripts/search.py -R '.*merge_sort.*pairs' -a 'KeyT{ct}=I128' -a 'Elements{io}[pow2]=28'
  cub.bench.merge_sort.pairs.trp_0.ld_1.ipt_13.tpb_6 0.6805093269929858
  cub.bench.merge_sort.pairs.trp_0.ld_1.ipt_11.tpb_10 1.0774560502969677
  ...

This will tune merge sort for key-value pairs, for the key type :code:`int128_t` on :code:`2^28` elements.
The :code:`-R` and :code:`-a` options are optional. If not specified, all benchmarks are going to be tuned.
The :code:`-R` option can select multiple benchmarks using a regular expression.
For the axis option :code:`-a`, you can also specify a range of values like :code:`-a 'KeyT{ct}=[I32,I64]'`.
Any axis values not supported by a selected benchmark will be ignored.
The first variant :code:`cub.bench.merge_sort.pairs.trp_0.ld_1.ipt_13.tpb_6` has a score <1 and is thus generally slower than baseline,
whereas the second variant :code:`cub.bench.merge_sort.pairs.trp_0.ld_1.ipt_11.tpb_10` has a score of >1 and is thus an improvement over the baseline.

Notice there is currently a limitation in :code:`search.py`
which will only execute runs for the first axis value for each axis
(independently of whether the axis is specified on the command line or not).
Please see `this issue <https://github.com/NVIDIA/cccl/issues/2267>`_ for more information.

The tuning framework will handle building the benchmarks (base and variants) by itself.
It will keep track of the build time for base and variants.
Sometimes, a tuning variant may lead the compiler to hang or take exceptionally long to compile.
To keep the tuning process going, if the build time of a variant exceeds a threshold, the build is cancelled.
The same applies to benchmarks running for too long.

To get quick feedback on what benchmarks are selected and how big the search space is,
you can add the :code:`-l` option:

.. code:: bash

  $ ../benchmarks/scripts/search.py -R '.*merge_sort.*pairs' -a 'KeyT{ct}=I128' -a 'Elements{io}[pow2]=28' -l
  ctk:  12.6.85
  cccl:  v2.7.0
  ### Benchmarks
    * `cub.bench.merge_sort.pairs`: 540 variants:
      * `trp`: (0, 2, 1)
      * `ld`: (0, 3, 1)
      * `ipt`: (7, 25, 1)
      * `tpb`: (6, 11, 1)

It will list all selected benchmarks as well as the total number of variants (the magnitude of the search space)
as a result of the Cartesian product of all its tuning parameter spaces.


Analyzing the results
--------------------------------------------------------------------------------

The result of the search is stored in the :code:`build/cccl_meta_bench.db` file. To analyze the
result you can use the :code:`analyze.py` script.
The :code:`--coverage` flag will show the amount of variants that were covered per compile-time workload:

.. code:: bash

  $ ../benchmarks/scripts/analyze.py --coverage
    cub.bench.radix_sort.keys[T{ct}=I8, OffsetT{ct}=I32] coverage: 167 / 522 (31.9923%)
    cub.bench.radix_sort.keys[T{ct}=I8, OffsetT{ct}=I64] coverage: 152 / 522 (29.1188%)

The :code:`--top N` flag will list the best :code:`N` variants for each compile-time workload:

.. code:: bash

  $ ../benchmarks/scripts/analyze.py --top=5
    cub.bench.radix_sort.keys[T{ct}=I8, OffsetT{ct}=I32]:
              variant     score      mins     means      maxs
    97  ipt_19.tpb_512  1.141015  1.039052  1.243448  1.679558
    84  ipt_18.tpb_512  1.136463  1.030434  1.245825  1.668038
    68  ipt_17.tpb_512  1.132696  1.020470  1.250665  1.688889
    41  ipt_15.tpb_576  1.124077  1.011560  1.245011  1.722379
    52  ipt_16.tpb_512  1.121044  0.995238  1.252378  1.717514
    cub.bench.radix_sort.keys[T{ct}=I8, OffsetT{ct}=I64]:
              variant     score      mins     means      maxs
    71  ipt_19.tpb_512  1.250941  1.155738  1.321665  1.647868
    86  ipt_20.tpb_512  1.250840  1.128940  1.308591  1.612382
    55  ipt_17.tpb_512  1.244399  1.152033  1.327424  1.692091
    98  ipt_21.tpb_448  1.231045  1.152798  1.298332  1.621110
    85  ipt_20.tpb_480  1.229382  1.135447  1.294937  1.631225

The name of the variant contains the short parameter names and values used for the variant.
For each variant, a score is reported. The base has a score of 1.0, so each score higher than 1.0 is an improvement over the base.
However, because a single variant contains multiple runtime workloads, also the minimum, mean, maximum score is reported.
If all those three values are larger than 1.0, the variant is strictly better than the base.
If only the mean or max are larger than 1.0, the variant may perform better in most runtime workloads, but regress in others.
This information can be used to change the existing tuning policies in CUB.

..
    TODO(bgruber): the following is outdated:

.. code:: bash

  $ ../benchmarks/scripts/analyze.py --variant='ipt_(18|19).tpb_512'

The last command plots distribution of the elapsed times for the specified variants.
