.. _libcudacxx-extended-api-functional-hash:

``cuda::hash``
================

Defined in the header ``<cuda/functional>``.

.. code:: cuda

    enum class hash_algorithm {
        xxhash_32,
        xxhash_64,
        murmurhash3_32,
        murmurhash3_x86_128, // requires compiler support for __int128
        murmurhash3_x64_128, // requires compiler support for __int128
    };

    template <typename Key, hash_algorithm Algorithm = hash_algorithm::xxhash_64>
    class hash;

    template <typename Key>
    class hash<Key, hash_algorithm::xxhash_32> {
    public:
        __host__ __device__ constexpr explicit hash(cuda::std::uint32_t seed = 0);
        [[nodiscard]] __host__ __device__ constexpr cuda::std::uint32_t operator()(const Key& key) const noexcept;

        template <cuda::std::size_t Extent>
        [[nodiscard]] __host__ __device__ cuda::std::uint32_t operator()(cuda::std::span<Key, Extent> keys) const noexcept;
    };

    template <typename Key>
    class hash<Key, hash_algorithm::xxhash_64> {
    public:
        __host__ __device__ constexpr explicit hash(cuda::std::uint64_t seed = 0);
        [[nodiscard]] __host__ __device__ cuda::std::uint64_t operator()(const Key& key) const noexcept;

        template <cuda::std::size_t Extent>
        [[nodiscard]] __host__ __device__ cuda::std::uint64_t operator()(cuda::std::span<Key, Extent> keys) const noexcept;
    };

    template <typename Key>
    class hash<Key, hash_algorithm::murmurhash3_32> {
    public:
        __host__ __device__ constexpr explicit hash(cuda::std::uint32_t seed = 0);
        [[nodiscard]] __host__ __device__ constexpr cuda::std::uint32_t operator()(const Key& key) const noexcept;

        template <cuda::std::size_t Extent>
        [[nodiscard]] __host__ __device__ cuda::std::uint32_t operator()(cuda::std::span<Key, Extent> keys) const noexcept;
    };

    #if _CCCL_HAS_INT128()

    template <typename Key>
    class hash<Key, hash_algorithm::murmurhash3_x86_128> {
    public:
        __host__ __device__ constexpr explicit hash(cuda::std::uint32_t seed = 0);
        [[nodiscard]] __host__ __device__ constexpr __uint128_t operator()(const Key& key) const noexcept;

        template <cuda::std::size_t Extent>
        [[nodiscard]] __host__ __device__ __uint128_t operator()(cuda::std::span<Key, Extent> keys) const noexcept;
    };

    template <typename Key>
    class hash<Key, hash_algorithm::murmurhash3_x64_128> {
    public:
        __host__ __device__ constexpr explicit hash(cuda::std::uint64_t seed = 0);
        [[nodiscard]] __host__ __device__ constexpr __uint128_t operator()(const Key& key) const noexcept;

        template <cuda::std::size_t Extent>
        [[nodiscard]] __host__ __device__ __uint128_t operator()(cuda::std::span<Key, Extent> keys) const noexcept;
    };

    #else // _CCCL_HAS_INT128()

    template <typename Key>
    class hash<Key, hash_algorithm::murmurhash3_x86_128> {
        static_assert(cuda::std::__always_false_v<Key>,
                      "cuda::hash with hash_algorithm::murmurhash3_x86_128 requires compiler support for __int128");
    };

    template <typename Key>
    class hash<Key, hash_algorithm::murmurhash3_x64_128> {
        static_assert(cuda::std::__always_false_v<Key>,
                      "cuda::hash with hash_algorithm::murmurhash3_x64_128 requires compiler support for __int128");
    };

    #endif // _CCCL_HAS_INT128()

``cuda::hash`` provides host/device implementations of xxHash and MurmurHash3.
The hash is computed from the raw object representation of a key.
Consequently, equal objects whose complete object representations differ,
including padding bytes, are not guaranteed to produce equal hash values.
xxHash and MurmurHash3 are non-cryptographic hash functions and must not be used
for security-sensitive hashing.

Each specialization accepts an optional seed and hashes either one key or the
concatenated object representations in a contiguous mutable ``cuda::std::span``
of keys. Spans with static or dynamic extent are accepted.

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

The 128-bit algorithm enumerators are always available. Instantiating either
128-bit ``cuda::hash`` specialization without compiler support for ``__int128``
triggers a static assertion.

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
