.. _cub-developer-guide-block-scope:

Block-scope
************

Overview
=========

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
