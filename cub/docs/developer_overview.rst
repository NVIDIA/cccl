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
For example, :cpp:struct:`WarpReduce <cub::WarpReduce>` uses :cpp:func:`ThreadReduce <cub::internal::ThreadReduce>`, 
:cpp:struct:`BlockReduce <cub::BlockReduce>` uses :cpp:struct:`WarpReduce <cub::WarpReduce>`, etc.

:cpp:func:`ThreadReduce <cub::internal::ThreadReduce>` 

   - A normal function invoked and executed sequentially by a single thread that returns a valid result on that thread
   - Single thread functions are usually an implementation detail and not exposed in CUB's public API

:cpp:struct:`WarpReduce <cub::WarpReduce>` and :cpp:struct:`BlockReduce <cub::BlockReduce>`

   - A "cooperative" function where threads concurrently invoke the same function to execute parallel work
   - The function's return value is well-defined only on the "first" thread (lowest thread index)

:cpp:struct:`DeviceReduce <cub::DeviceReduce>`

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
    * - :cpp:func:`ThreadReduce <cub::internal::ThreadReduce>`  
      - :math:`-`        
      - :math:`-`           
      - :math:`1`       
      - invoking thread 
    * - :cpp:struct:`WarpReduce <cub::WarpReduce>`              
      - :math:`+`        
      - :math:`+`           
      - :math:`32`      
      - main thread     
    * - :cpp:struct:`BlockReduce <cub::BlockReduce>`            
      - :math:`+`        
      - :math:`+`           
      - :math:`1024`    
      - main thread     
    * - :cpp:struct:`DeviceReduce <cub::DeviceReduce>`          
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
              int LOGICAL_WARP_THREADS = 32,
              int LEGACY_PTX_ARCH = 0>
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

It's important to note that ``LEGACY_PTX_ARCH`` has been recently deprecated. 
This parameter used to affect specialization selection (see below). 
It was conflicting with the PTX dispatch refactoring and limited NVHPC support.

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

    using InternalWarpReduce = cub::detail::conditional_t<
      IS_POW_OF_TWO,
      WarpReduceShfl<T, LOGICAL_WARP_THREADS>,  // shuffle-based implementation 
      WarpReduceSmem<T, LOGICAL_WARP_THREADS>>; // smem-based implementation 

Specializations provide different shared memory requirements, 
so the actual ``_TempStorage`` type is defined as:

.. code-block:: c++

    typedef typename InternalWarpReduce::TempStorage _TempStorage;

and algorithm implementation look like:

.. code-block:: c++

    __device__ __forceinline__ T Sum(T input, int valid_items) {
      return InternalWarpReduce(temp_storage)
          .Reduce(input, valid_items, cub::Sum());
    }

Due to ``LEGACY_PTX_ARCH`` issues described above, 
we can't specialize on the PTX version. 
``NV_IF_TARGET`` shall be used by specializations instead:

.. code-block:: c++

    template <typename T, int LOGICAL_WARP_THREADS, int LEGACY_PTX_ARCH = 0>
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
      typename std::enable_if<std::is_same<int, U>::value ||
                              std::is_same<unsigned int, U>::value,
                              T>::type
        ReduceImpl(T input, 
                  int,      // valid_items
                  cub::Sum) // reduction_op
    {
      T output = input;

      NV_IF_TARGET(NV_PROVIDES_SM_80,
                  (output = __reduce_add_sync(member_mask, input);),
                  (output = ReduceImpl<cub::Sum>(
                        input, LOGICAL_WARP_THREADS, cub::Sum{});));

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
              int BLOCK_DIM_Z = 1, 
              int LEGACY_PTX_ARCH = 0>
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

Device-scope functionality is provided by static member functions:

.. code-block:: c++

    struct DeviceReduce
    {


    template <typename InputIteratorT, typename OutputIteratorT>
    CUB_RUNTIME_FUNCTION static 
    cudaError_t Sum(
        void *d_temp_storage, size_t &temp_storage_bytes, InputIteratorT d_in,
        OutputIteratorT d_out, int num_items, cudaStream_t stream = 0) 
    {
      using OffsetT = int;
      using OutputT =
          cub::detail::non_void_value_t<OutputIteratorT,
                                        cub::detail::value_t<InputIteratorT>>;

      return DispatchReduce<InputIteratorT, OutputIteratorT, OffsetT,
                            cub::Sum>::Dispatch(d_temp_storage, temp_storage_bytes,
                                                d_in, d_out, num_items, cub::Sum(),
                                                OutputT(), stream);
    }


    };

Device-scope facilities always return ``cudaError_t`` and accept ``stream`` as the last parameter (main stream by default). 
The first two parameters are always ``void *d_temp_storage, size_t &temp_storage_bytes``. 
The algorithm invocation consists of two phases. 
During the first phase, 
temporary storage size is calculated and returned in ``size_t &temp_storage_bytes``. 
During the second phase, 
``temp_storage_bytes`` of memory is expected to be allocated and ``d_temp_storage`` is expected to be the pointer to this memory.

.. code-block:: c++

    // Determine temporary device storage requirements
    void *d_temp_storage {};
    std::size_t temp_storage_bytes {};

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);

.. warning::
    Even if the algorithm doesn't need temporary storage as a scratch space, 
    we still require one byte of memory to be allocated.


Dispatch layer
====================================

The dispatch layer is specific to the device scope (`DispatchReduce`) and located in `cub/device/dispatch`. 
High-level control flow can be represented by the code below. 
A more precise description is given later.

.. code-block:: c++

    // Device-scope API
    cudaError_t cub::Device::Algorithm(args) {
      return DispatchAlgorithm::Dispatch(args); // (1)
    }

    // Dispatch entry point
    static cudaError_t DispatchAlgorithm::Dispatch(args) { // (1)
      DispatchAlgorithm invokable(args);
      // MaxPolicy - tail of linked list contaning architecture-specific tunings
      return MaxPolicy::Invoke(get_runtime_ptx_version(), invokable); // (2)
    }

    // Chained policy - linked list of tunings 
    cudaError_t ChainedPolicy::Invoke(ptx_version, invokable) { // (2)
      if (ptx_version < ChainedPolicy::PTX_VERSION) {
        ChainedPolicy::PrevPolicy::Invoke(ptx_version, invokable); // (2)
      } 
      invokable.Invoke<ChainedPolicy::PTX_VERSION>(); // (3)
    }

    // Dispatch object - parameters closure
    template <class Policy>
    cudaError_t DispatchAlgorithm::Invoke() { // (3)
        kernel<MaxPolicy><<<grid_size, Policy::BLOCK_THREADS>>>(args); // (4)
    }

    template <class ChainedPolicy>
    void __global__ __launch_bounds__(ChainedPolicy::ActivePolicy::BLOCK_THREADS)
    kernel(args) { // (4)
        using policy = ChainedPolicy::ActivePolicy; // (5)
        using agent = AgentAlgorithm<policy>; // (6)
        agent a(args);
        a.Process();
    }

    template <int PTX_VERSION, typename PolicyT, typename PrevPolicyT>
    struct ChainedPolicy {
      using ActivePolicy = conditional_t<(CUB_PTX_ARCH < PTX_VERSION), // (5)
                                        PrevPolicyT::ActivePolicy,
                                        PolicyT>;
    };

    template <class Policy>
    struct AlgorithmAgent { // (6)
      void Process();
    };

The code above represents control flow. 
Let's look at each of the building blocks closer. 

The dispatch entry point is typically represented by a static member function that constructs an object of ``DispatchReduce`` and passes it to ``ChainedPolicy`` ``Invoke`` method:

.. code-block:: c++

    template <typename InputIteratorT, 
              // ... 
              typename SelectedPolicy>
    struct DispatchReduce : SelectedPolicy 
    {

      // 
      CUB_RUNTIME_FUNCTION __forceinline__ static 
      cudaError_t Dispatch(
          void *d_temp_storage, size_t &temp_storage_bytes, 
          InputIteratorT d_in,
          /* ... */) 
      {
        typedef typename DispatchSegmentedReduce::MaxPolicy MaxPolicyT;

        if (num_segments <= 0) return cudaSuccess;

        cudaError error = cudaSuccess;
        do {
          // Get PTX version
          int ptx_version = 0;
          error = CubDebug(PtxVersion(ptx_version));
          if (cudaSuccess != error) 
          {
            break;
          }

          // Create dispatch functor
          DispatchSegmentedReduce dispatch(
              d_temp_storage, temp_storage_bytes, d_in, /* ... */);

          // Dispatch to chained policy
          MaxPolicyT::Invoke(ptx_version, dispatch);
        } while (0);

        return error;
      }

    };

For many algorithms, the dispatch layer is part of the API. 
The first reason for this to be the case is ``size_t`` support. 
Our API uses ``int`` as a type for ``num_items``. 
Users rely on the dispatch layer directly to workaround this. 
Exposing the dispatch layer also allows users to tune algorithms for their use cases. 
In the newly added functionality, the dispatch layer should not be exposed.

The ``ChainedPolicy`` converts the runtime PTX version to the closest compile-time one:

.. code-block:: c++

    template <int PTX_VERSION, 
              typename PolicyT, 
              typename PrevPolicyT>
    struct ChainedPolicy
    {
      using ActivePolicy =
        cub::detail::conditional_t<(CUB_PTX_ARCH < PTX_VERSION),
                                  typename PrevPolicyT::ActivePolicy,
                                  PolicyT>;

      template <typename FunctorT>
      CUB_RUNTIME_FUNCTION __forceinline__
      static cudaError_t Invoke(int ptx_version, FunctorT& op)
      {
          if (ptx_version < PTX_VERSION) {
              return PrevPolicyT::Invoke(ptx_version, op);
          }
          return op.template Invoke<PolicyT>();
      }
    };

The dispatch object's ``Invoke`` method is then called with proper policy:

.. code-block:: c++

    template <typename InputIteratorT, 
              // ... 
              typename SelectedPolicy = DefaultTuning>
    struct DispatchReduce : SelectedPolicy 
    {
        
    template <typename ActivePolicyT>
    CUB_RUNTIME_FUNCTION __forceinline__
    cudaError_t Invoke()
    {
      using MaxPolicyT = typename DispatchSegmentedReduce::MaxPolicy;

      return InvokePasses<ActivePolicyT /* (1) */>(
        DeviceReduceKernel<MaxPolicyT /* (2) */, InputIteratorT /* ... */>);
    }
        
    };

This is where the actual work happens. 
Note how the kernel is used against ``MaxPolicyT`` (2) while the kernel invocation part uses ``ActivePolicyT`` (1). 
This is an important part:

.. code-block:: c++

    template <typename ChainedPolicyT /* ... */ >
    __launch_bounds__(int(ChainedPolicyT::ActivePolicy::ReducePolicy::BLOCK_THREADS)) 
    __global__ void DeviceReduceKernel(InputIteratorT d_in /* ... */) 
    {
      // Thread block type for reducing input tiles
      using AgentReduceT =
          AgentReduce<typename ChainedPolicyT::ActivePolicy::ReducePolicy,
                      InputIteratorT, OutputIteratorT, OffsetT, ReductionOpT>;

      // Shared memory storage
      __shared__ typename AgentReduceT::TempStorage temp_storage;

      // Consume input tiles
      OutputT block_aggregate =
          AgentReduceT(temp_storage, d_in, reduction_op).ConsumeTiles(even_share);

      // Output result
      if (threadIdx.x == 0) {
        d_out[blockIdx.x] = block_aggregate;
      }
    }

The kernel gets compiled for each PTX version that was provided to the compiler. 
During the device pass, 
``ChainedPolicy`` compares ``CUDA_ARCH`` against the template parameter to select ``ActivePolicy`` type alias. 
During the host pass, 
``Invoke`` is compiled for each architecture in the tuning list. 
If we use ``ActivePolicy`` instead of ``MaxPolicy`` as a kernel template parameter, 
we will compile ``O(N^2)`` kernels instead of ``O(N)``. 

Finally, the tuning looks like:

.. code-block:: c++

    template <typename InputT /* ... */>
    struct DeviceReducePolicy 
    {
      /// SM35
      struct Policy350 : ChainedPolicy<350, Policy350, Policy300> {
        typedef AgentReducePolicy<256, 20, InputT, 4, BLOCK_REDUCE_WARP_REDUCTIONS,
                                  LOAD_LDG>
            ReducePolicy;
      };

      /// SM60
      struct Policy600 : ChainedPolicy<600, Policy600, Policy350> {
        typedef AgentReducePolicy<256, 16, InputT, 4, BLOCK_REDUCE_WARP_REDUCTIONS,
                                  LOAD_LDG>
            ReducePolicy;
      };

      /// MaxPolicy
      typedef Policy600 MaxPolicy;
    };

The kernels in the dispatch layer shouldn't contain a lot of code. 
Usually, the functionality is extracted into the agent layer. 
All the kernel does is derive the proper policy type, 
unwrap the policy to initialize the agent and call one of its ``Consume`` / ``Process`` methods. 
Agents are frequently reused by unrelated device-scope algorithms. 

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

    #. Hiding symbols accpeting kernel pointers: 
       it's important that API accepting kernel pointers (e.g. ``triple_chevron``) always reside in the same 
       library as the code taking this pointers.  

    #. Hiding all kernels: 
       it's important that kernels always reside in the same library as the API using these kernels.  

    #. Incorporating GPU architectures into symbol names: 
       it's important that kernels compiled for a given GPU architecture are always used by the host 
       API compiled for that architecture.  

To satisfy (1), ``thrust::cuda_cub::launcher::triple_chevron`` visibility is hidden.  

To satisfy (2), instead of annotating kernels as ``__global__`` we annotate them as 
``CUB_DETAIL_KERNEL_ATTRIBUTES``. Apart from annotating a kernel as global function, the macro 
contains hidden visibility attribute.

To satisfy (3), CUB symbols are placed inside an inline namespace containing the set of 
GPU architectures for which the TU is being compiled.  
