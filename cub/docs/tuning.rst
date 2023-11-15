CUB Tuning Infrastructure
================================================================================

Device-scope algorithms in CUB have many knobs that do not affect the algorithms' correctness but can significantly impact performance. For instance, the number of threads per block and items per thread can be tuned to maximize performance for a given device and data type. 
This document describes CUB Tuning Infrastructure, a set of tools facilitating the process of 
selecting optimal tuning parameters for a given device and data type.

Definitions
--------------------------------------------------------------------------------

Terms might be ambiguous in a generic context. Below, we omit the word "tuning" but assume it in all definitions. 
Algorithms are tuned for different workloads. For instance, radix sort can be tuned for different key types, different number of keys, and different distribution of keys. We separate tuning parameters into two categories: 

* **Compile-time (ct) Workload** - a workload that can be recognized at compile time. For instance, the combination of key type and offset type is a compile-time workload for radix sort.

* **Runtime (rt) Workload** - a workload that can be recognized only at runtime. For instance, the number of keys along with their distribution is a runtime workload for radix sort.

* **Parameter** - a parameter that can be tuned to maximize performance for a given device and data type. For instance, the number of threads per block and items per thread are tuning parameters.

* **Parameter Space** - the set of all possible values for a given tuning parameter. Parameter Space is specific to algorithm. For instance, the parameter space for the number of threads per block is :math:`\{32, 64, 96, 128, \dots, 1024\}` for radix sort, but :math:`\{32, 64, 128, 256, 512\}` for merge sort.

* **Parameter Point** - a concrete value of a tuning parameter. For instance, the parameter point for the number of threads per block is :math:`threads\_per\_block=128`.

* **Search Space** - Cartesian product of parameter spaces. For instance, search space for an algorithm with tunable items per thread and threads per block might look like :math:`\{(ipt \times tpb) | ipt \in \{1, \dots, 25\} \text{and} tpb \in \{32, 64, 96, 128, \dots, 1024\}\}`.

* **Variant** - a point from corresponding search space.

* **Base** - a variant that CUB uses by default.

* **Score** - a single number representing the performance for a given compile-time workload and all runtime workloads. For instance, a weighted-sum of speedups of a given variant compared to its base for all runtime workloads is a score.

* **Search** - a process consisting of covering all variants for all compile-time workloads to find a variant with maximal score. 


Contributing Benchmarks
--------------------------------------------------------------------------------

There are a few constraints on benchmarks. First of all, all benchmarks in a single
file should share type axes. Only alphabetical characters, numbers and underscore are allowed in the 
benchmark name.  The name of the file is represets the name of the algorithm. 
For instance, the :code:`benchmarks/bench/radix_sort/keys.cu` file name is going to be transformed 
into :code:`cub.bench.radix_sort.keys` that's further used in the infrastructure.

You start writing a benchmark by including :code:`nvbench_helper.cuh` file. It contains all
necessary includes and definitions.

.. code:: c++

  #include <nvbench_helper.cuh>

The next step is to define a search space. The search space is represented by a number of comments. 
The format consists of :code:`%RANGE%` keyword, a parameter name, and a range. The range is
represented by three numbers: start, end, and step. For instance, the following code defines a search
space for the number of threads per block and items per thread.

.. code:: c++

  // %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
  // %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32

Next, you need to define a benchmark function. The function accepts :code:`nvbench::state &state` and
a :code:`nvbench::type_list`. For more details on the benchmark signature, take a look at the 
nvbench docs.

.. code:: c++

  template <typename T, typename OffsetT>
  void algname(nvbench::state &state, nvbench::type_list<T, OffsetT>)
  {

Now we have to specialize the dispatch layer. The tuning infrastructure will use `TUNE_BASE` macro 
to distinguish between the base and the variant. When base is used, do not specify the policy, so 
that the default one is used. If the macro is not defined, specify custom policy using macro 
names defined at the search space specification step.

.. code:: c++

  #if TUNE_BASE
    using dispatch_t = cub::DispatchReduce<T, OffsetT>;
  #else
    using policy_t = policy_hub_t<accum_t, offset_t>;
    using dispatch_t = cub::DispatchReduce<T, OffsetT, policy_t>;
  #endif

If possible, do not initialize the input data in the benchmark function. Instead, use the
:code:`gen` function. This function will fill the input vector with random data on GPU with no 
compile-time overhead. 

.. code:: c++

    const auto elements = static_cast<std::size_t>(state.get_int64("Elements{io}"));
    thrust::device_vector<T> in(elements);
    thrust::device_vector<T> out(1);

    gen(seed_t{}, in);

You can optionally add memory usage to the state: 

.. code:: c++

    state.add_element_count(elements);
    state.add_global_memory_reads<T>(elements, "Size");
    state.add_global_memory_writes<T>(1);

Now we are ready to allocate temporary storage:

.. code:: c++

    std::size_t temp_size;
    dispatch_t::Dispatch(nullptr,
                         temp_size,
                         d_in,
                         d_out,
                         static_cast<offset_t>(elements),
                         0 /* stream */);

    thrust::device_vector<nvbench::uint8_t> temp(temp_size);
    auto *temp_storage = thrust::raw_pointer_cast(temp.data());

Finally, we can run the algorithm:

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

Having the benchmark function, we can tell nvbench about it. A few things to note here. First of all,
compile-time axes should be annotated as :code:`{ct}`. The runtime axes might be optionally annotated
as :code:`{io}` which stands for importance-ordered. This will tell the tuning infrastructure that 
the later values on the axis are more important. If the axis is not annotated, each value will be
treated as equally important.

.. code:: c++

  NVBENCH_BENCH_TYPES(algname, NVBENCH_TYPE_AXES(all_types, offset_types))
    .set_name("base")
    .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
    .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));


When you define a type axis that's annotated as :code:`{ct}`, you might want to consider optimizing 
the build time. Many variants are going to be build, but the search is considering one compile-time 
use case at a time. This means, that if you have many types to tune for, you'll end up having 
many specializations that you don't need. To avoid this, for each compile time axis, you can 
expect a `TUNE_AxisName` macro with the type that's currently being tuned. For instance, if you
have a type axes :code:`T{ct}` and :code:`OffsetT` (as shown above), you can use the following
construct:

.. code:: c++

  #ifdef TUNE_T
  using types = nvbench::type_list<TUNE_T>;
  #else
  using types = all_types;
  #endif

  #ifdef TUNE_OffsetT
  using offset_types = nvbench::type_list<TUNE_OffsetT>;
  #else
  using offset_types = nvbench::type_list<int32_t, int64_t>;
  #endif


This logic is automatically applied to :code:`all_types`, :code:`offset_types`, and 
:code:`fundamental_types` lists when you use matching names for the axes. You can define 
your own axis names and use the logic above for them (see sort pairs example).


Search Process
--------------------------------------------------------------------------------

To get started with tuning / benchmarking, you need to configure CMake. The following options are
available: 

* :code:`CUB_ENABLE_BENCHMARKS` - enable bases (default: OFF).
* :code:`CUB_ENABLE_TUNING` - enable variants (default: OFF).

Having configured CMake, you can start the search process. Note that the search has to be started 
from the build directory.

.. code:: bash

  $ cd build
  $ cmake -DThrust_DIR=path-to-thrust/thrust/cmake -DCUB_ENABLE_TUNING=YES -DCUB_ENABLE_BENCHMARKS=YES -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="90" ..
  $ ../benchmarks/scripts/search.py -a "T{ct}=[I8,I16]" -R ".*algname.*"

Both :code:`-a` and :code:`-R` options are optional. The first one is used to specify types to tune 
for. The second one is used to specify benchmarks to be tuned. If not specified, all benchmarks are 
going to be tuned.

The result of the search is stored in the :code:`build/cccl_meta_bench.db` file. To analyze the 
result you can use the :code:`analyze.py` script:

.. code:: bash 

  $ ../benchmarks/scripts/analyze.py --coverage                     
    cub.bench.radix_sort.keys[T{ct}=I8, OffsetT{ct}=I32] coverage: 167 / 522 (31.9923%)
    cub.bench.radix_sort.keys[T{ct}=I8, OffsetT{ct}=I64] coverage: 152 / 522 (29.1188%)
  
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
  
  $ ../benchmarks/scripts/analyze.py --variant='ipt_(18|19).tpb_512'

The last command plots distribution of the elapsed times for the specified variants.
