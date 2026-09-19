.. _cub-developer-guide-thread-level:

Thread-level
*************

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
