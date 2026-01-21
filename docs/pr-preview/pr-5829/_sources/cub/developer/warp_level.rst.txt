.. _cub-developer-guide-warp-level:

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
