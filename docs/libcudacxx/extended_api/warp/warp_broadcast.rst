.. _libcudacxx-extended-api-warp-warp-broadcast:

``cuda::device::warp_broadcast``
==============================================

Defined in ``<cuda/warp>`` header.

.. code:: cuda

    namespace cuda::device {

    template <typename T>
    [[nodiscard]] __device__ T
    warp_broadcast(const T& value,
                   cuda::std::uint32_t src_lane,
                   lane_mask lane_mask = lane_mask::all())

    } // namespace cuda::device

Broadcasts a ``value`` from ``src_lane`` to all lanes whose lane bit is set in ``lane_mask``.

**Parameters**

- ``value``: The value to broadcast.
- ``src_lane``: The source lane.
- ``lane_mask``: The lane mask of threads participating in the broadcast.

**Return value**

Returns ``T`` of the broadcasted value.

**Constrains**

- ``T`` must be trivially copyable.
- ``T`` must be default constructible.

**Preconditions**

- ``src_lane`` must be less than ``32``.
- ``src_lane``'s lane bit must be set in the ``lane_mask``.
- The calling thread have its lane bit set in the ``lane_mask``.

**Performance considerations**

- The function calls the PTX instruction ``shfl.sync`` :math:`ceil\left(\frac{sizeof(value)}{4}\right)` times.

**References**

- `PTX shfl.sync instruction <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync>`_

Example
-------

.. code:: cuda

    #include <cuda/warp>
    #include <cuda/ptx>

    __global__ void warp_broadcast_kernel()
    {
        const auto lane = cuda::ptx::get_sreg_laneid();

        // Set the secret value only on lane 0.
        unsigned secret;
        if (lane == 0)
        {
            secret = 0xDEAD'BEEF;
        }

        // Broadcast the secret and verify all threads got the same value.
        secret = cuda::device::warp_broadcast(secret, 0);
        assert(secret == 0xDEAD'BEEF);

        // Set the odd_secret value only on lane 1.
        unsigned odd_secret = 0;
        if (lane == 1)
        {
            odd_secret = 0xA'BAD'CAFE;
        }

        // Broadcast the odd_secret only among odd lanes.
        if (lane % 2 == 1)
        {
            odd_secret = cuda::device::warp_broadcast(odd_secret, 1, cuda::device::lane_mask{0xaaaa'aaaau});
        }

        // Verify that only the odd lanes got the result.
        assert(odd_secret == ((lane % 2 == 1) ? 0xA'BAD'CAFE : 0));
    }

    int main()
    {
        warp_broadcast_kernel<<<1, 32>>>();
        assert(cudaDeviceSynchronize() == cudaSuccess);
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/cv6hY5nPK>`_
