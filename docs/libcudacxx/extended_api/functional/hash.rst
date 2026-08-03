.. _libcudacxx-extended-api-functional-hash:

``cuda::hash``
================

Defined in the header ``<cuda/functional>``.

.. code:: cuda

    enum class hash_algorithm {
        xxhash_32,
        xxhash_64,
        murmurhash3_32,
        murmurhash3_x86_128,
        murmurhash3_x64_128
    };

    template <typename Key, hash_algorithm Algorithm = hash_algorithm::xxhash_32>
    class hash;

``cuda::hash`` provides host/device implementations of xxHash and MurmurHash3.
``Key`` must be trivially copyable. Each specialization accepts an optional seed
and hashes either one key or a contiguous ``cuda::std::span`` of keys.

The supported algorithms and result types are:

.. list-table::
   :header-rows: 1

   * - Algorithm
     - Result type
   * - ``hash_algorithm::xxhash_32``
     - ``cuda::std::uint32_t``
   * - ``hash_algorithm::xxhash_64``
     - ``cuda::std::uint64_t``
   * - ``hash_algorithm::murmurhash3_32``
     - ``cuda::std::uint32_t``
   * - ``hash_algorithm::murmurhash3_x86_128``
     - ``__uint128_t``
   * - ``hash_algorithm::murmurhash3_x64_128``
     - ``__uint128_t``

The 128-bit algorithms are available only when the compiler supports 128-bit
integer types.

Example
-------

.. code:: cuda

    #include <cuda/functional>
    #include <cuda/std/cstdint>

    __global__ void hash_keys(const int* keys, cuda::std::uint64_t* hashes)
    {
        const auto index = blockIdx.x * blockDim.x + threadIdx.x;
        hashes[index] = cuda::hash<int, cuda::hash_algorithm::xxhash_64>{}(keys[index]);
    }
