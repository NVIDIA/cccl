CUB Developer Overview
##########################


This living document serves as a guide to the design of the internal structure of CUB.

CUB provides layered algorithms that correspond to the thread/warp/block/device hierarchy of threads in CUDA.
There are distinct algorithms for each layer and higher-level layers build on top of those below.

For example, CUB has four flavors of ``reduce``,
one for each layer: ``ThreadReduce, WarpReduce, BlockReduce``, and ``DeviceReduce``.
Each is unique in how it is invoked,
how many threads participate,
and on which thread(s) the result is valid.

These layers naturally build on each other.
For example, :cpp:struct:`cub::WarpReduce` uses :cpp:func:`cub::ThreadReduce`,
:cpp:struct:`cub::BlockReduce` uses :cpp:struct:`cub::WarpReduce`, etc.

:cpp:func:`cub::ThreadReduce`

   - A normal function invoked and executed sequentially by a single thread that returns a valid result on that thread
   - Single thread functions are usually an implementation detail and not exposed in CUB's public API

:cpp:struct:`cub::WarpReduce` and :cpp:struct:`cub::BlockReduce`

   - A "cooperative" function where threads concurrently invoke the same function to execute parallel work
   - The function's return value is well-defined only on the "first" thread (lowest thread index)

:cpp:struct:`cub::DeviceReduce`

   - A normal function invoked by a single thread that spawns additional threads to execute parallel work
   - Result is stored in the pointer provided to the function
   - Function returns a ``cudaError_t`` error code
   - Function does not synchronize the host with the device


The table below provides a summary of these functions:

.. list-table::
    :class: table-no-stripes
    :header-rows: 1

    * - layer
      - coop invocation
      - parallel execution
      - max threads
      - valid result in
    * - :cpp:func:`cub::ThreadReduce`
      - :math:`-`
      - :math:`-`
      - :math:`1`
      - invoking thread
    * - :cpp:struct:`cub::WarpReduce`
      - :math:`+`
      - :math:`+`
      - :math:`32`
      - main thread
    * - :cpp:struct:`cub::BlockReduce`
      - :math:`+`
      - :math:`+`
      - :math:`1024`
      - main thread
    * - :cpp:struct:`cub::DeviceReduce`
      - :math:`-`
      - :math:`+`
      - :math:`\infty`
      - global memory

The details of how each of these layers are implemented is described below.

Common Patterns
************************************

While CUB's algorithms are unique at each layer,
there are commonalities among all of them:

    - Algorithm interfaces are provided as *types* (classes)\ [1]_
    - Algorithms need temporary storage
    - Algorithms dispatch to specialized implementations depending on compile-time and runtime information
    - Cooperative algorithms require the number of threads at compile time (template parameter)

Invoking any CUB algorithm follows the same general pattern:

    #. Select the class for the desired algorithm
    #. Query the temporary storage requirements
    #. Allocate the temporary storage
    #. Pass the temporary storage to the algorithm
    #. Invoke it via the appropriate member function

An example of :cpp:struct:`cub::BlockReduce` demonstrates these patterns in practice:

.. code-block:: c++

    __global__ void kernel(int* per_block_results)
    {
      // (1) Select the desired class
      // `cub::BlockReduce` is a class template that must be instantiated for the
      // input data type and the number of threads. Internally the class is
      // specialized depending on the data type, number of threads, and hardware
      // architecture. Type aliases are often used for convenience:
      using BlockReduce = cub::BlockReduce<int, 128>;
      // (2) Query the temporary storage
      // The type and amount of temporary storage depends on the selected instantiation
      using TempStorage = typename BlockReduce::TempStorage;
      // (3) Allocate the temporary storage
      __shared__ TempStorage temp_storage;
      // (4) Pass the temporary storage
      // Temporary storage is passed to the constructor of the `BlockReduce` class
      BlockReduce block_reduce{temp_storage};
      // (5) Invoke the algorithm
      // The `Sum()` member function performs the sum reduction of `thread_data` across all 128 threads
      int thread_data[4] = {1, 2, 3, 4};
      int block_result = block_reduce.Sum(thread_data);

      per_block_results[blockIdx.x] = block_result;
    }

.. [1] Algorithm interfaces are provided as classes because it provides encapsulation for things like temporary storage requirements and enables partial template specialization for customizing an algorithm for specific data types or number of threads.

Thread-level
************************************

In contrast to algorithms at the warp/block/device layer,
single threaded functionality like ``cub::ThreadReduce``
is typically implemented as a sequential function and rarely exposed to the user.

.. code-block:: c++

    template <
        int         LENGTH,
        typename    T,
        typename    ReductionOp,
        typename    PrefixT,
        typename    AccumT = detail::accumulator_t<ReductionOp, PrefixT, T>>
    __device__ __forceinline__ AccumT ThreadReduce(
        T           (&input)[LENGTH],
        ReductionOp reduction_op,
        PrefixT     prefix)
    {
        return ...;
    }

Warp-level
************************************

CUB warp-level algorithms are specialized for execution by threads in the same CUDA warp.
These algorithms may only be invoked by ``1 <= n <= 32`` *consecutive* threads in the same warp.

Overview
====================================

Warp-level functionality is provided by types (classes) to provide encapsulation and enable partial template specialization.

For example, :cpp:struct:`cub::WarpReduce` is a class template:

.. code-block:: c++

    template <typename T,
              int      LOGICAL_WARP_THREADS = 32>
    class WarpReduce {
      // ...
      // (1)   define `_TempStorage` type
      // ...
      _TempStorage &temp_storage;
    public:

      // (2)   wrap `_TempStorage` in uninitialized memory
      struct TempStorage : Uninitialized<_TempStorage> {};

      __device__ __forceinline__ WarpReduce(TempStorage &temp_storage)
      // (3)   reinterpret cast
        : temp_storage(temp_storage.Alias())
      {}

      // (4)   actual algorithms
      __device__ __forceinline__ T Sum(T input);
    };

In CUDA, the hardware warp size is 32 threads.
However, CUB enables warp-level algorithms on "logical" warps of ``1 <= n <= 32`` threads.
The size of the logical warp is required at compile time via the ``LOGICAL_WARP_THREADS`` non-type template parameter.
This value is defaulted to the hardware warp size of ``32``.
There is a vital difference in the behavior of warp-level algorithms that depends on the value of ``LOGICAL_WARP_THREADS``:

- If ``LOGICAL_WARP_THREADS`` is a power of two - warp is partitioned into *sub*-warps,
  each reducing its data independently from other *sub*-warps.
  The terminology used in CUB: ``32`` threads are called hardware warp.
  Groups with less than ``32`` threads are called *logical* or *virtual* warp since it doesn't correspond directly to any hardware unit.
- If ``LOGICAL_WARP_THREADS`` is **not** a power of two - there's no partitioning.
  That is, only the first logical warp executes algorithm.

.. TODO: Add diagram showing non-power of two logical warps.

Temporary storage usage
====================================

Warp-level algorithms require temporary storage for scratch space and inter-thread communication.
The temporary storage needed for a given instantiation of an algorithm is known at compile time
and is exposed through the ``TempStorage`` member type definition.
It is the caller's responsibility to create this temporary storage and provide it to the constructor of the algorithm type.
It is possible to reuse the same temporary storage for different algorithm invocations,
but it is unsafe to do so without first synchronizing to ensure the first invocation is complete.

.. TODO: Add more explanation of the `TempStorage` type and the `Uninitialized` wrapper.
.. TODO: Explain if `TempStorage` is required to be shared memory or not.


.. code-block:: c++

    using WarpReduce = cub::WarpReduce<int>;

    // Allocate WarpReduce shared memory for four warps
    __shared__ WarpReduce::TempStorage temp_storage[4];

    // Get this thread's warp id
    int warp_id = threadIdx.x / 32;
    int aggregate_1 = WarpReduce(temp_storage[warp_id]).Sum(thread_data_1);
    // illegal, has to add `__syncwarp()` between the two
    int aggregate_2 = WarpReduce(temp_storage[warp_id]).Sum(thread_data_2);
    // illegal, has to add `__syncwarp()` between the two
    foo(temp_storage[warp_id]);


Specialization
====================================

The goal of CUB is to provide users with algorithms that abstract the complexities of achieving speed-of-light performance across a variety of use cases and hardware.
It is a CUB developer's job to abstract this complexity from the user by providing a uniform interface that statically dispatches to the optimal code path.
This is usually accomplished via customizing the implementation based on compile time information like the logical warp size, the data type, and the target architecture.
For example, :cpp:struct:`cub::WarpReduce` dispatches to two different implementations based on if the logical warp size is a power of two (described above):

.. code-block:: c++

    using InternalWarpReduce = cuda::std::conditional_t<
      IS_POW_OF_TWO,
      detail::WarpReduceShfl<T, LOGICAL_WARP_THREADS>,  // shuffle-based implementation
      detail::WarpReduceSmem<T, LOGICAL_WARP_THREADS>>; // smem-based implementation

Specializations provide different shared memory requirements,
so the actual ``_TempStorage`` type is defined as:

.. code-block:: c++

    using _TempStorage = typename InternalWarpReduce::TempStorage;

and algorithm implementation look like:

.. code-block:: c++

    __device__ __forceinline__ T Sum(T input, int valid_items) {
      return InternalWarpReduce(temp_storage)
          .Reduce(input, valid_items, ::cuda::std::plus<>{});
    }



``__CUDA_ARCH__`` cannot be used because it is conflicting with the PTX dispatch refactoring and limited NVHPC support.
Due to  this limitation, we can't specialize on the PTX version.
``NV_IF_TARGET`` shall be used by specializations instead:

.. code-block:: c++

    template <typename T, int LOGICAL_WARP_THREADS>
    struct WarpReduceShfl
    {


    template <typename ReductionOp>
    __device__ __forceinline__ T ReduceImpl(T input, int valid_items,
                                            ReductionOp reduction_op)
    {
      // ... base case (SM < 80) ...
    }

    template <class U = T>
    __device__ __forceinline__
      typename std::enable_if<std::is_same_v<int, U> ||
                              std::is_same_v<unsigned int, U>,
                              T>::type
        ReduceImpl(T input,
                  int,               // valid_items
                  ::cuda::std::plus<>) // reduction_op
    {
      T output = input;

      NV_IF_TARGET(NV_PROVIDES_SM_80,
                  (output = __reduce_add_sync(member_mask, input);),
                  (output = ReduceImpl<::cuda::std::plus<>>(
                        input, LOGICAL_WARP_THREADS, ::cuda::std::plus<>{});));

      return output;
    }


    };

Specializations are stored in the ``cub/warp/specializations`` directory.

Block-scope
************************************

Overview
====================================

Block-scope algorithms are provided by structures as well:

.. code-block:: c++

    template <typename T,
              int BLOCK_DIM_X,
              BlockReduceAlgorithm ALGORITHM = BLOCK_REDUCE_WARP_REDUCTIONS,
              int BLOCK_DIM_Y = 1,
              int BLOCK_DIM_Z = 1>
    class BlockReduce {
    public:
      struct TempStorage : Uninitialized<_TempStorage> {};

      // (1) new constructor
      __device__ __forceinline__ BlockReduce()
          : temp_storage(PrivateStorage()),
            linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z)) {}

      __device__ __forceinline__ BlockReduce(TempStorage &temp_storage)
          : temp_storage(temp_storage.Alias()),
            linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z)) {}
    };

While warp-scope algorithms only provide a single constructor that requires the user to provide temporary storage,
block-scope algorithms provide two constructors:

    #. The default constructor that allocates the required shared memory internally.
    #. The constructor that requires the user to provide temporary storage as argument.

In the case of the default constructor,
the block-level algorithm uses the ``PrivateStorage()`` member function to allocate the required shared memory.
This ensures that shared memory required by the algorithm is only allocated when the default constructor is actually called in user code.
If the default constructor is never called,
then the algorithm will not allocate superfluous shared memory.

.. code-block:: c++

    __device__ __forceinline__ _TempStorage& PrivateStorage()
    {
      __shared__ _TempStorage private_storage;
      return private_storage;
    }

The ``__shared__`` memory has static semantic, so it's safe to return a reference here.

Specialization
====================================

Block-scope facilities usually expose algorithm selection to the user.
The algorithm is represented by the enumeration part of the API.
For the reduction case,
``BlockReduceAlgorithm`` is provided.
Specializations are stored in the ``cub/block/specializations`` directory.

Temporary storage usage
====================================

For block-scope algorithms,
it's unsafe to use temporary storage without synchronization:

.. code-block:: c++

    using BlockReduce = cub::BlockReduce<int, 128> ;

    __shared__ BlockReduce::TempStorage temp_storage;

    int aggregate_1 = BlockReduce(temp_storage).Sum(thread_data_1);
    // illegal, has to add `__syncthreads` between the two
    int aggregate_2 = BlockReduce(temp_storage).Sum(thread_data_2);
    // illegal, has to add `__syncthreads` between the two
    foo(temp_storage);


Device-scope
************************************

Overview
====================================

Device-scope functionality is provided by classes called ``DeviceAlgorithm``,
where ``Algorithm`` is the implemented algorithm.
These classes then contain static member functions providing corresponding API entry points.

.. code-block:: c++

    struct DeviceAlgorithm {
      template <typename ...>
      CUB_RUNTIME_FUNCTION static cudaError_t Algorithm(
          void *d_temp_storage, size_t &temp_storage_bytes, ..., cudaStream_t stream = 0) {
        // optional: minimal argument checking or setup to call dispatch layer
        return DispatchAlgorithm<...>::Dispatch(d_temp_storage, temp_storage_bytes, ..., stream);
      }
    };

For example, device-level reduce will look like `cub::DeviceReduce::Sum`.
Device-scope facilities always return ``cudaError_t`` and accept ``stream`` as the last parameter (NULL stream by default)
and the first two parameters are always ``void *d_temp_storage, size_t &temp_storage_bytes``.
The implementation may consist of some minimal argument checking, but should forward as soon as possible to the dispatch layer.
Device-scope algorithms are implemented in files located in `cub/device/device_***.cuh`.

In general, the use of a CUB algorithm consists of two phases:

  1. Temporary storage size is calculated and returned in ``size_t &temp_storage_bytes``.
  2. ``temp_storage_bytes`` of memory is expected to be allocated and ``d_temp_storage`` is expected to be the pointer to this memory.

The following example illustrates this pattern:

.. code-block:: c++

    // First call: Determine temporary device storage requirements
    std::size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);

    // Allocate temporary storage
    void *d_temp_storage = nullptr;
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Second call: Perform algorithm
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);

.. warning::
    Even if the algorithm doesn't need temporary storage as scratch space,
    we still require one byte of memory to be allocated.


Dispatch layer
====================================

A dispatch layer exists for each device-scope algorithms (e.g., `DispatchReduce`),
and is located in `cub/device/dispatch`.
Only device-scope algorithms have a dispatch layer.

The dispatch layer follows a certain architecture.
The high-level control flow is represented by the code below.
A more precise description is given later.

.. code-block:: c++

    // Device-scope API
    cudaError_t cub::DeviceAlgorithm::Algorithm(d_temp_storage, temp_storage_bytes, ...) {
      return DispatchAlgorithm::Dispatch(d_temp_storage, temp_storage_bytes, ...); // calls (1)
    }

    // Dispatch entry point
    static cudaError_t DispatchAlgorithm::Dispatch(...) { // (1)
      DispatchAlgorithm closure{...};
      // MaxPolicy - tail of linked list containing architecture-specific tunings
      return MaxPolicy::Invoke(get_device_ptx_version(), closure); // calls (2)
    }

    // Chained policy - linked list of tunings
    template <int PolicyPtxVersion, typename Policy, typename PrevPolicy>
    struct ChainedPolicy {
      using ActivePolicy = conditional_t<CUB_PTX_ARCH < PolicyPtxVersion, // (5)
                                        typename PrevPolicy::ActivePolicy, Policy>;

      static cudaError_t Invoke(int device_ptx_version, auto dispatch_closure) { // (2)
        if (device_ptx_version < PolicyPtxVersion) {
          PrevPolicy::Invoke(device_ptx_version, dispatch_closure); // calls (2) of next policy
        }
        dispatch_closure.Invoke<Policy>(); // eventually calls (3)
      }
    };

    // Dispatch object - a closure over all algorithm parameters
    template <typename Policy>
    cudaError_t DispatchAlgorithm::Invoke() { // (3)
        // host-side implementation of algorithm, calls kernels
        kernel<MaxPolicy><<<grid_size, Policy::AlgorithmPolicy::BLOCK_THREADS>>>(...); // calls (4)
    }

    template <typename ChainedPolicy>
    __launch_bounds__(ChainedPolicy::ActivePolicy::AlgorithmPolicy::BLOCK_THREADS) CUB_DETAIL_KERNEL_ATTRIBUTES
    void kernel(...) { // (4)
      using policy = ChainedPolicy::ActivePolicy; // selects policy of active device compilation pass (5)
      using agent = AgentAlgorithm<policy>; // instantiates (6)
      agent a{...};
      a.Process(); // calls (7)
    }

    template <typename Policy>
    struct AlgorithmAgent {  // (6)
      void Process() { ... } // (7)
    };

Let's look at each of the building blocks closer.

The dispatch entry point is typically represented by a static member function called ``DispatchAlgorithm::Dispatch``
that constructs an object of type ``DispatchAlgorithm``, filling it with all arguments to run the algorithm,
and passes it to the ``ChainedPolicy::Invoke`` function:

.. code-block:: c++

    template <..., // algorithm specific compile-time parameters
              typename PolicyHub>
    struct DispatchAlgorithm {
      CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static
      cudaError_t Dispatch(void *d_temp_storage, size_t &temp_storage_bytes, ..., cudaStream stream) {
        if (/* no items to process */) {
          if (d_temp_storage == nullptr) {
            temp_storage_bytes = 1;
          }
          return cudaSuccess;
        }

        int ptx_version   = 0;
        const cudaError_t error = CubDebug(PtxVersion(ptx_version));
        if (cudaSuccess != error)
        {
          return error;
        }
        DispatchAlgorithm dispatch(..., stream);
        return CubDebug(PolicyHub::MaxPolicy::Invoke(ptx_version, dispatch));
      }
    };

For many legacy algorithms, the dispatch layer is publicly accessible and used directly by users,
since it often exposes additional performance knobs or configuration,
like choosing the index type or policies to use.
Exposing the dispatch layer also allowed users to tune algorithms for their use cases.
In the newly added algorithms, the dispatch layer should not be exposed publicly anymore.

The ``ChainedPolicy`` has two purposes.
During ``Invoke``, it converts the runtime PTX version of the current device
to the nearest lower-or-equal compile-time policy available:

.. code-block:: c++

    template <int PolicyPtxVersion, typename Policy, typename PrevPolicy>
    struct ChainedPolicy {
      using ActivePolicy = conditional_t<CUB_PTX_ARCH < PolicyPtxVersion,
                                        typename PrevPolicy::ActivePolicy, Policy>;

      template <typename Functor>
      CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE
      static cudaError_t Invoke(int device_ptx_version, Functor dispatch_closure) {
        if (device_ptx_version < PolicyPtxVersion) {
          PrevPolicy::Invoke(device_ptx_version, dispatch_closure);
        }
        dispatch_closure.Invoke<Policy>();
      }
    };

The dispatch object's ``Invoke`` function is then called with the best policy for the device's PTX version:

.. code-block:: c++

    template <..., typename PolicyHub = detail::algorithm::policy_hub>
    struct DispatchAlgorithm {
      template <typename ActivePolicy>
      CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE
      cudaError_t Invoke() {
        // host-side implementation of algorithm, calls kernels
        using MaxPolicy = typename DispatchSegmentedReduce::MaxPolicy;
        kernel<MaxPolicy /*(2)*/><<<grid_size, ActivePolicy::AlgorithmPolicy::BLOCK_THREADS /*(1)*/>>>(...); // calls (4)
      }
    };

This is where all the host-side work happens and kernels are eventually launched using the supplied policies.
Note how the kernel is instantiated on ``MaxPolicy`` (2) while the kernel launch configuration uses ``ActivePolicy`` (1).
This is an important optimization to reduce compilation-time:

.. code-block:: c++

    template <typename ChainedPolicy /* ... */ >
    __launch_bounds__(ChainedPolicy::ActivePolicy::AlgorithmPolicy::BLOCK_THREADS) __CUB_DETAIL_KERNEL_ATTRIBUTES
    void kernel(...) {
      using policy = ChainedPolicy::ActivePolicy::AlgorithmPolicy;
      using agent = AgentAlgorithm<policy>;

      __shared__ typename agent::TempStorage temp_storage; // allocate static shared memory for agent

      agent a{temp_storage, ...};
      a.Process();
    }

The kernel gets compiled for each PTX version (``N`` many) that was provided to the compiler.
During each device pass,
``ChainedPolicy`` compares ``CUB_PTX_ARCH`` against the template parameter ``PolicyPtxVersion``
to select an ``ActivePolicy`` type.
During the host pass,
``Invoke`` is compiled for each architecture in the tuning list (``M`` many).
If we used ``ActivePolicy`` instead of ``MaxPolicy`` as a kernel template parameter,
we would compile ``O(M*N)`` kernels instead of ``O(N)``.

The kernels in the dispatch layer shouldn't contain a lot of code.
Usually, the functionality is extracted into the agent layer.
All the kernel does is derive the proper policy type,
unwrap the policy to initialize the agent and call one of its ``Consume`` / ``Process`` functions.
Agents hold kernel bodies and are frequently reused by unrelated device-scope algorithms.

To gain a better understanding of why we use ``MaxPolicy`` instead of
``ActivePolicy`` to instantiate the kernel, consider the following example which
minimally reproduces the CUB dispatch layer.

.. code-block:: c++

    #include <cuda/std/type_traits>
    #include <cstdio>

    /// In device code, CUB_PTX_ARCH expands to the PTX version for which we are
    /// compiling. In host code, CUB_PTX_ARCH's value is implementation defined.
    #ifndef CUB_PTX_ARCH
    #  if !defined(__CUDA_ARCH__)
    #    define CUB_PTX_ARCH 0
    #  else
    #    define CUB_PTX_ARCH __CUDA_ARCH__
    #  endif
    #endif

    template <int PTXVersion, typename Policy, typename PrevPolicy>
    struct ChainedPolicy {
      static constexpr int cc = PTXVersion;
      using ActivePolicy      = ::cuda::std::conditional_t<CUB_PTX_ARCH<PTXVersion, PrevPolicy, Policy>;
      using PrevPolicyT       = PrevPolicy;
      using PolicyT           = Policy;
    };

    template <int PTXVersion, typename Policy>
    struct ChainedPolicy<PTXVersion, Policy, Policy> {
      static constexpr int cc = PTXVersion;
      using ActivePolicy      = Policy;
      using PrevPolicyT       = Policy;
      using PolicyT           = Policy;
    };

    struct sm60 : ChainedPolicy<600, sm60, sm60> { static constexpr int BLOCK_THREADS = 256; };
    struct sm70 : ChainedPolicy<700, sm70, sm60> { static constexpr int BLOCK_THREADS = 512; };
    struct sm80 : ChainedPolicy<800, sm80, sm70> { static constexpr int BLOCK_THREADS = 768; };
    struct sm90 : ChainedPolicy<900, sm90, sm80> { static constexpr int BLOCK_THREADS = 1024; };

    using MaxPolicy = sm90;

    // Kernel instantiated with ActivePolicy
    template <typename ActivePolicy>
    __launch_bounds__(ActivePolicy::BLOCK_THREADS) __global__ void bad_kernel() {
      if (threadIdx.x == 0) {
        std::printf("launch bounds %d; block threads %d\n", ActivePolicy::BLOCK_THREADS, blockDim.x);
      }
    }

    // Kernel instantiated with MaxPolicy
    template <typename MaxPolicy>
    __launch_bounds__(MaxPolicy::ActivePolicy::BLOCK_THREADS) __global__ void good_kernel() {
      if (threadIdx.x == 0) {
        std::printf("launch bounds %d; block threads %d\n", MaxPolicy::ActivePolicy::BLOCK_THREADS, blockDim.x);
      }
    }

    // Function to dispatch kernel with the correct ActivePolicy
    template <typename T>
    void invoke_with_best_policy_bad(int runtime_cc) {
      if (runtime_cc < T::cc) {
        invoke_with_best_policy_bad<typename T::PrevPolicyT>(runtime_cc);
      } else {
        bad_kernel<typename T::PolicyT><<<1, T::PolicyT::BLOCK_THREADS>>>();
      }
    }

    // Function to dispatch kernel with the correct MaxPolicy
    template <typename T>
    void invoke_with_best_policy_good(int runtime_cc) {
      if (runtime_cc < T::cc) {
        invoke_with_best_policy_good<typename T::PrevPolicyT>(runtime_cc);
      } else {
        good_kernel<MaxPolicy><<<1, T::PolicyT::BLOCK_THREADS>>>();
      }
    }

    void call_kernel() {
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, 0);
      int runtime_cc = deviceProp.major * 100 + deviceProp.minor * 10;
      std::printf("runtime cc %d\n", runtime_cc);

      invoke_with_best_policy_bad<MaxPolicy>(runtime_cc);
      invoke_with_best_policy_good<MaxPolicy>(runtime_cc);
    }

    int main() {
      call_kernel();
      cudaDeviceSynchronize();
    }

In this example, we define four execution policies for four GPU architectures,
``sm60``, ``sm70``, ``sm80``, and ``sm90``, with each having a different value
for ``BLOCK_THREADS``. Additionally, we define two kernels, ``bad_kernel`` and
``good_kernel`` to illustrate the effect of instantiating with the
``ActivePolicy`` and ``MaxPolicy`` respectively, as well a function to invoke
each of the kernels by selecting the policy for the appropriate architecture.
Finally, we call these two functions by specifying the current device's
architecture, which we obtain at run-time from host code. We cannot use the
``__CUDA_ARCH__`` macro here since it is not defined in host code, so it must be
extracted at run-time.

Compiling this file results in four template instantiations of ``bad_kernel``,
one for each architecture's policy, but only one template instantiation of
``good_kernel`` using ``MaxPolicy``. Assume we compiled the above file using
``nvcc policies.cu -std=c++20 -gencode arch=compute_80,code=sm_80 -o policies``,
we can inspect the generated binary to see these template instantiations using
``cuobjdump --dump-elf-symbols policies``. The relevant output is shown below.

.. code-block:: text

    symbols:
    STT_OBJECT       STB_LOCAL  STV_DEFAULT    $str
    STT_FUNC         STB_GLOBAL STO_ENTRY      _Z11good_kernelI4sm90Evv
    STT_FUNC         STB_GLOBAL STV_DEFAULT  U vprintf
    STT_FUNC         STB_GLOBAL STO_ENTRY      _Z10bad_kernelI4sm90Evv
    STT_FUNC         STB_GLOBAL STO_ENTRY      _Z10bad_kernelI4sm80Evv
    STT_FUNC         STB_GLOBAL STO_ENTRY      _Z10bad_kernelI4sm70Evv
    STT_FUNC         STB_GLOBAL STO_ENTRY      _Z10bad_kernelI4sm60Evv

We can see that there are four symbols containing ``bad_kernel`` in their name,
which correspond to the four execution policies, but only one symbol for
``good_kernel``. This is due to the functions that invoke these kernels. For
``bad_kernel``, the template instantiation we call depends on the ``T::PolicyT``
(the ``ActivePolicy``), which depends on a run-time argument, so the compiler
must instantiate ``bad_kernel`` for every policy. For ``good_kernel``, we always
instantiate the kernel with ``MaxPolicy``. This results in only one template
instantiation compared to four template instantiations per fat binary, which
saves on compile time and binary size.

To show that both kernels' invocation is equivalent, we can run the code using
``./policies``. We obtain the following output.

.. code-block:: text

    runtime cc 890
    launch bounds 768; block threads 768
    launch bounds 768; block threads 768

Our GPU's architecture is ``sm89``, so the nearest policy less than or equal to
that, ``sm80`` in this case, is selected at run-time. Both kernels are invoked
with the same launch bounds and number of threads per block.

.. _cub-developer-policies:

Policies
====================================

Policies describe the configuration of agents wrt. to their execution.
They do not change functional behavior, but usually affect how work is mapped to the hardware by defining certain compile-time parameters (items per thread, block size, etc.).

An agent policy could look like this:

.. code-block:: c++

    template <int _BLOCK_THREADS,
              int _ITEMS_PER_THREAD,
              BlockLoadAlgorithm _LOAD_ALGORITHM,
              CacheLoadModifier _LOAD_MODIFIER>
    struct AgentAlgorithmPolicy {
      static constexpr int BLOCK_THREADS    = _BLOCK_THREADS;
      static constexpr int ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
      static constexpr int ITEMS_PER_TILE   = BLOCK_THREADS * ITEMS_PER_THREAD;
      static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM   = _LOAD_ALGORITHM;
      static constexpr cub::CacheLoadModifier LOAD_MODIFIER     = _LOAD_MODIFIER;
    };

It's typically a collection of configuration values for the kernel launch configuration,
work distribution setting, load and store algorithms to use, as well as load instruction cache modifiers.
A CUB algorithm can have multiple agents and thus use multiple agent policies.

Since the device code of CUB algorithms is compiled for each PTX version, a different agent policy may be used.
Therefore, all agent policies of a CUB algorithm, called a policy, may be replicated for several minimum PTX versions.
A chained collection of such policies finally forms a policy hub:

.. code-block:: c++

    template <typename... TuningRelevantParams /* ... */>
    struct policy_hub {
      // TuningRelevantParams... could be used for decision making, like element types used, iterator category, etc.

      // for SM50
      struct Policy500 : ChainedPolicy<500, Policy500, Policy500> {
        using AlgorithmPolicy = AgentAlgorithmPolicy<256, 20, BLOCK_LOAD_DIRECT, LOAD_LDG>;
        // ... additional policies may exist, often one per agent
      };

      // for SM60
      struct Policy600 : ChainedPolicy<600, Policy600, Policy500> {
        using AlgorithmPolicy = AgentAlgorithmPolicy<256, 16, BLOCK_LOAD_DIRECT, LOAD_LDG>;
      };

      using MaxPolicy = Policy600; // alias where policy selection is started by ChainedPolicy
    };

The policy hub is a class template, possibly parameterized by tuning-relevant compile-time parameters,
containing a list of policies, one per minimum PTX version (i.e., SM architecture) they target.
These policies are chained by inheriting from ``ChainedPolicy``
and passing the minimum PTX version where they should be used,
as well as their own policy type and the next lower policy type.
An alias ``MaxPolicy`` serves as entry point into the chain of tuning policies.


Tunings
====================================

Because the values to parameterize an agent may vary a lot for different compile-time parameters,
the selection of values can be further delegated to tunings.
Often, such tunings are found by experimentation or heuristic search.
See also :ref:`cub-tuning`.

Tunings are usually organized as a class template, one per PTX version,
with a template specialization for each combination of the compile-time parameters,
for which better values for an agent policy are known.
An example set of tunings could look like this:

.. code-block:: c++

    template <int ValueSize, bool IsPlus>
    struct sm60_tuning { // default tuning
        static constexpr int threads = 128;
        static constexpr int items = 16;
    };

    template <>
    struct sm60_tuning<4, true> { // tuning for summing 4-byte values
        static constexpr int threads = 256;
        static constexpr int items = 20;
    };

    template <int ValueSize>
    struct sm60_tuning<ValueSize, true> { // tuning for summing values of other sizes
        static constexpr int threads = 128;
        static constexpr int items = 12;
    };

    ...

    template <typename ValueType, typename Operation>
    struct policy_hub {
      struct Policy600 : ChainedPolicy<600, Policy600, Policy500> {

        using tuning = sm60_tuning<sizeof(ValueType), is_same_v<Operation, plus>>;
        using AlgorithmPolicy = AgentAlgorithmPolicy<tuning::threads, tuning::items, BLOCK_LOAD_DIRECT, LOAD_LDG>;
      };
    };

Here, ``sm60_tuning`` provides defaults for the tuning values ``threads`` and ``items``.
``sm60_tuning`` is instantiated with the size of the value type and with a boolean indicating whether the operation is a sum.
Template specializations of ``sm60_tuning`` then provide different tuning values for summing value types of 4-byte size,
and for summing any other value types.
Notice how partial template specializations are used to pattern match the compile-time parameters.
Independent of which template specializations (or the base template) of the tuning is chose,
the agent policy is then parameterized by the nested ``threads`` and ``items`` values from this tuning.

The logic to select tunings varies, and different mechanisms are used for different algorithms.
Some algorithms provide a generic default policy if no tuning is available,
others implement a fallback logic to select the previous PTX version's agent policy,
if no tuning is available for the current PTX version.
In general, tunings are not exhaustive and usually only apply for specific combinations of parameter values and a single PTX version,
falling back to generic policies when no tuning matches.
Tunings for CUB algorithms reside in ``cub/device/dispatch/tuning/tuning_<algorithm>.cuh``.


Temporary storage usage
====================================

It's safe to reuse storage in the stream order:

.. code-block:: c++

    cub::DeviceReduce::Sum(nullptr, storage_bytes, d_in, d_out, num_items, stream_1);
    // allocate temp storage
    cub::DeviceReduce::Sum(d_storage, storage_bytes, d_in, d_out, num_items, stream_1);
    // fine not to synchronize stream
    cub::DeviceReduce::Sum(d_storage, storage_bytes, d_in, d_out, num_items, stream_1);
    // illegal, should call cudaStreamSynchronize(stream)
    cub::DeviceReduce::Sum(d_storage, storage_bytes, d_in, d_out, num_items, stream_2);

Temporary storage management
====================================

Often times temporary storage for device-scope algorithms has a complex structure.
To simplify temporary storage management and make it safer,
we introduced ``cub::detail::temporary_storage::layout``:

.. code-block:: c++

    cub::detail::temporary_storage::layout<2> storage_layout;

    auto slot_1 = storage_layout.get_slot(0);
    auto slot_2 = storage_layout.get_slot(1);

    auto allocation_1 = slot_1->create_alias<int>();
    auto allocation_2 = slot_1->create_alias<double>(42);
    auto allocation_3 = slot_2->create_alias<char>(12);

    if (condition)
    {
      allocation_1.grow(num_items);
    }

    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = storage_layout.get_size();
      return;
    }

    storage_layout.map_to_buffer(d_temp_storage, temp_storage_bytes);

    // different slots, safe to use simultaneously
    use(allocation_1.get(), allocation_3.get(), stream);
    // `allocation_2` alias `allocation_1`, safe to use in stream order
    use(allocation_2.get(), stream);


Symbols visibility
====================================

Using CUB/Thrust in shared libraries is a known source of issues.
For a while, the solution to these issues consisted of wrapping CUB/Thrust namespaces with
the ``THRUST_CUB_WRAPPED_NAMESPACE`` macro so that different shared libraries have different symbols.
This solution has poor discoverability,
since issues present themselves in forms of segmentation faults, hangs, wrong results, etc.
To eliminate the symbol visibility issues on our end, we follow the following rules:

    #. Hiding symbols accepting kernel pointers:
       it's important that an API accepting kernel pointers (e.g. ``triple_chevron``) always resides in the same
       library as the code taking this pointers.

    #. Hiding all kernels:
       it's important that kernels always reside in the same library as the API using these kernels.

    #. Incorporating GPU architectures into symbol names:
       it's important that kernels compiled for a given GPU architecture are always used by the host
       API compiled for that architecture.

To satisfy (1), the visibility of ``thrust::cuda_cub::detail::triple_chevron`` is hidden.

To satisfy (2), instead of annotating kernels as ``__global__`` we annotate them as
``CUB_DETAIL_KERNEL_ATTRIBUTES``. Apart from annotating a kernel as global function, the macro also
contains an attribute to set the visibility to hidden.

To satisfy (3), CUB symbols are placed inside an inline namespace containing the set of
GPU architectures for which the TU is being compiled.


NVTX
************************************

The `NVIDIA Tools Extension SDK (NVTX) <https://nvidia.github.io/NVTX/>`_ is a cross-platform API
for annotating source code to provide contextual information to developer tools.
All device-scope algorithms in CUB are annotated with NVTX ranges,
allowing their start and stop to be visualized in profilers
like `NVIDIA Nsight Systems <https://developer.nvidia.com/nsight-systems>`_.
Only the public APIs available in the ``<cub/device/device_xxx.cuh>`` headers are annotated,
excluding direct calls to the dispatch layer.
NVTX annotations can be disabled by defining ``NVTX_DISABLE`` during compilation.
When CUB device algorithms are called on a stream subject to
`graph capture <https://developer.nvidia.com/blog/cuda-graphs/>`_,
the NVTX range is reported for the duration of capture (where no execution happens),
and not when a captured graph is executed later (the actual execution).
