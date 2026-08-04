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

    template <typename Key>
    class hash<Key, hash_algorithm::xxhash_32> {
    public:
        __host__ __device__ constexpr hash(cuda::std::uint32_t seed = 0) noexcept;
        [[nodiscard]] __host__ __device__ constexpr cuda::std::uint32_t operator()(const Key& key) const noexcept;

        template <typename SpanKey, cuda::std::size_t Extent>
        [[nodiscard]] __host__ __device__ cuda::std::uint32_t operator()(cuda::std::span<SpanKey, Extent> keys) const noexcept;
    };

    template <typename Key>
    class hash<Key, hash_algorithm::xxhash_64> {
    public:
        __host__ __device__ constexpr hash(cuda::std::uint64_t seed = 0) noexcept;
        [[nodiscard]] __host__ __device__ cuda::std::uint64_t operator()(const Key& key) const noexcept;

        template <typename SpanKey, cuda::std::size_t Extent>
        [[nodiscard]] __host__ __device__ cuda::std::uint64_t operator()(cuda::std::span<SpanKey, Extent> keys) const noexcept;
    };

    template <typename Key>
    class hash<Key, hash_algorithm::murmurhash3_32> {
    public:
        __host__ __device__ constexpr hash(cuda::std::uint32_t seed = 0) noexcept;
        [[nodiscard]] __host__ __device__ constexpr cuda::std::uint32_t operator()(const Key& key) const noexcept;

        template <typename SpanKey, cuda::std::size_t Extent>
        [[nodiscard]] __host__ __device__ cuda::std::uint32_t operator()(cuda::std::span<SpanKey, Extent> keys) const noexcept;
    };

    template <typename Key>
    class hash<Key, hash_algorithm::murmurhash3_x86_128> {
    public:
        __host__ __device__ constexpr hash(cuda::std::uint32_t seed = 0) noexcept;
        [[nodiscard]] __host__ __device__ constexpr cuda::std::array<cuda::std::uint32_t, 4> operator()(const Key& key) const noexcept;

        template <typename SpanKey, cuda::std::size_t Extent>
        [[nodiscard]] __host__ __device__ cuda::std::array<cuda::std::uint32_t, 4> operator()(cuda::std::span<SpanKey, Extent> keys) const noexcept;
    };

    template <typename Key>
    class hash<Key, hash_algorithm::murmurhash3_x64_128> {
    public:
        __host__ __device__ constexpr hash(cuda::std::uint64_t seed = 0) noexcept;
        [[nodiscard]] __host__ __device__ constexpr cuda::std::array<cuda::std::uint64_t, 2> operator()(const Key& key) const noexcept;

        template <typename SpanKey, cuda::std::size_t Extent>
        [[nodiscard]] __host__ __device__ cuda::std::array<cuda::std::uint64_t, 2> operator()(cuda::std::span<SpanKey, Extent> keys) const noexcept;
    };

``cuda::hash`` provides host/device implementations of xxHash and MurmurHash3.
``Key`` must be trivially copyable; this requirement is enforced when the class is
instantiated. The hash is computed from the raw object representation of a key.

Each specialization accepts an optional seed and hashes either one key or the
concatenated object representations in a contiguous ``cuda::std::span`` of keys.
Both mutable and const spans, with static or dynamic extent, are accepted.
The span overload participates in overload resolution only when
``cuda::std::remove_const_t<SpanKey>`` and ``cuda::std::remove_const_t<Key>``
are the same type.

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
     - ``cuda::std::array<cuda::std::uint32_t, 4>``
   * - ``hash_algorithm::murmurhash3_x64_128``
     - ``cuda::std::array<cuda::std::uint64_t, 2>``

The arrays store words in algorithm output order: ``{h1, h2, h3, h4}`` for
``murmurhash3_x86_128`` and ``{h1, h2}`` for ``murmurhash3_x64_128``.

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
