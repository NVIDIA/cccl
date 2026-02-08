#include <thrust/detail/config.h>

#include <thrust/mr/disjoint_pool.h>
#include <thrust/mr/disjoint_sync_pool.h>
#include <thrust/mr/new.h>

#include <unittest/unittest.h>

struct alloc_id
{
  std::size_t id;
  std::size_t size;
  std::size_t alignment;
  std::size_t offset{};

  _CCCL_HOST_DEVICE bool operator==(const alloc_id& other) const
  {
    return id == other.id && size == other.size && alignment == other.alignment;
  }

  alloc_id operator+(std::size_t size_) const
  {
    alloc_id ret;
    ret.id        = id;
    ret.size      = size_;
    ret.alignment = alignment;
    ret.offset    = size_;
    return ret;
  }
};

THRUST_NAMESPACE_BEGIN
namespace detail
{
template <>
struct pointer_traits<alloc_id>
{
  template <typename>
  struct rebind
  {
    using other = alloc_id;
  };

  // implemented for the purposes of alignment test in disjoint pool's do_deallocate
  static void* get(const alloc_id& id)
  {
    return reinterpret_cast<void*>(id.alignment);
  }
};
} // end namespace detail

THRUST_NAMESPACE_END

class dummy_resource final : public thrust::mr::memory_resource<alloc_id>
{
public:
  dummy_resource() = default;

  ~dummy_resource()
  {
    ASSERT_EQUAL(id_to_allocate, 0u);
    ASSERT_EQUAL(id_to_deallocate, 0u);
    ASSERT_EQUAL(used_bytes, 0u);
    ASSERT_EQUAL(allocation_ids.size(), 0u);
  }

  void assert_empty_and_reset()
  {
    ASSERT_EQUAL(used_bytes, 0u);
    ASSERT_EQUAL(allocation_ids.size(), 0u);
    free_bytes       = 1ull << 63;
    id_to_allocate   = 0;
    id_to_deallocate = 0;
  }

  virtual alloc_id do_allocate(std::size_t bytes, std::size_t alignment) override
  {
    if (bytes > free_bytes)
    {
      throw thrust::system::detail::bad_alloc("Dummy allocation failed: insufficient free bytes.");
    }

    ASSERT_NOT_EQUAL(id_to_allocate, 0u);
    // Ensure that the allocation ID is unique
    ASSERT_EQUAL_QUIET(find(allocation_ids.begin(), allocation_ids.end(), id_to_allocate), allocation_ids.end());

    free_bytes -= bytes;
    used_bytes += bytes;
    allocation_ids.push_back(id_to_allocate);

    alloc_id ret;
    ret.id        = id_to_allocate;
    ret.size      = bytes;
    ret.alignment = alignment;

    id_to_allocate = 0;

    return ret;
  }

  virtual void do_deallocate(alloc_id p, std::size_t bytes, std::size_t alignment) override
  {
    ASSERT_EQUAL(p.size, bytes);
    ASSERT_EQUAL(p.alignment, alignment);
    ASSERT_LEQUAL(bytes, used_bytes);
    // Check that the id has been previously allocated
    ASSERT_NOT_EQUAL_QUIET(find(allocation_ids.begin(), allocation_ids.end(), p.id), allocation_ids.end());

    free_bytes += bytes;
    used_bytes -= bytes;
    allocation_ids.erase(find(allocation_ids.begin(), allocation_ids.end(), p.id));

    if (id_to_deallocate != 0)
    {
      ASSERT_EQUAL(p.id, id_to_deallocate);
      id_to_deallocate = 0;
    }
  }

  std::size_t free_bytes{1ull << 63};
  std::size_t used_bytes{0};
  std::vector<std::size_t> allocation_ids;

  std::size_t id_to_allocate{};
  std::size_t id_to_deallocate{};
};

template <template <typename, typename> class PoolTemplate>
void TestDisjointPool()
{
  dummy_resource upstream;
  thrust::mr::new_delete_resource bookkeeper;

  using Pool = PoolTemplate<dummy_resource, thrust::mr::new_delete_resource>;

  thrust::mr::pool_options opts = Pool::get_default_options();
  opts.cache_oversized          = false;

  // avoid having the destructor run when an assertion failure is raised
  // (the destructor will try to release, which in turn calls do_deallocate,
  // which may fail with an assertion failure exception...)
  Pool* pool = new Pool(&upstream, &bookkeeper, opts);

  upstream.id_to_allocate = 1;

  // first allocation
  alloc_id a1 = pool->do_allocate(12, THRUST_MR_DEFAULT_ALIGNMENT);
  ASSERT_EQUAL(a1.id, 1u);

  // due to chunking, the above allocation should be enough for the next one too
  alloc_id a2 = pool->do_allocate(16, THRUST_MR_DEFAULT_ALIGNMENT);
  ASSERT_EQUAL(a2.id, 1u);

  // deallocating and allocating back should give the same resource back
  pool->do_deallocate(a1, 12, THRUST_MR_DEFAULT_ALIGNMENT);
  alloc_id a3 = pool->do_allocate(12, THRUST_MR_DEFAULT_ALIGNMENT);
  ASSERT_EQUAL(a1.id, a3.id);
  ASSERT_EQUAL(a1.size, a3.size);
  ASSERT_EQUAL(a1.alignment, a3.alignment);
  ASSERT_EQUAL(a1.offset, a3.offset);

  // allocating over-aligned memory should give non-cached results
  upstream.id_to_allocate = 2;
  alloc_id a4             = pool->do_allocate(32, THRUST_MR_DEFAULT_ALIGNMENT * 2);
  ASSERT_EQUAL(a4.id, 2u);
  ASSERT_EQUAL(a4.size, 32u);
  ASSERT_EQUAL(a4.alignment, (std::size_t) THRUST_MR_DEFAULT_ALIGNMENT * 2);

  // and deallocating it should return it back to upstream
  upstream.id_to_deallocate = 2;
  pool->do_deallocate(a4, 32u, THRUST_MR_DEFAULT_ALIGNMENT * 2);
  ASSERT_EQUAL(upstream.id_to_deallocate, 0u);

  // release actually returns properly sized memory to upstream
  upstream.id_to_deallocate = 1;
  pool->release();
  ASSERT_EQUAL(upstream.id_to_deallocate, 0u);

  // and does the same for oversized/overaligned memory
  upstream.id_to_allocate = 3;
  alloc_id a5             = pool->do_allocate(1024, THRUST_MR_DEFAULT_ALIGNMENT * 2);
  ASSERT_EQUAL(upstream.id_to_allocate, 0u);
  ASSERT_EQUAL(a5.id, 3u);

  upstream.id_to_deallocate = 3;
  pool->release();
  ASSERT_EQUAL(upstream.id_to_deallocate, 0u);

  // and after that, the formerly cached memory isn't used anymore,
  // so new memory from upstream is returned back
  upstream.id_to_allocate = 4;
  alloc_id a6             = pool->do_allocate(16, THRUST_MR_DEFAULT_ALIGNMENT);
  ASSERT_EQUAL(upstream.id_to_allocate, 0u);
  ASSERT_EQUAL(a6.id, 4u);

  // destruction also returns memory
  upstream.id_to_deallocate = 4;

  // actually destroy the pool; reasons why RAII is not used outlined at the beginning
  // of this function
  delete pool;
  ASSERT_EQUAL(upstream.id_to_deallocate, 0u);
}

void TestDisjointUnsynchronizedPool()
{
  TestDisjointPool<thrust::mr::disjoint_unsynchronized_pool_resource>();
}
DECLARE_UNITTEST(TestDisjointUnsynchronizedPool);

void TestDisjointSynchronizedPool()
{
  TestDisjointPool<thrust::mr::disjoint_synchronized_pool_resource>();
}
DECLARE_UNITTEST(TestDisjointSynchronizedPool);

template <template <typename, typename> class PoolTemplate>
void TestDisjointPoolCachingOversized()
{
  dummy_resource upstream;
  thrust::mr::new_delete_resource bookkeeper;

  using Pool = PoolTemplate<dummy_resource, thrust::mr::new_delete_resource>;

  thrust::mr::pool_options opts = Pool::get_default_options();
  opts.cache_oversized          = true;
  opts.largest_block_size       = 1024;

  Pool pool(&upstream, &bookkeeper, opts);

  upstream.id_to_allocate = 1;
  alloc_id a1             = pool.do_allocate(2048, 32);
  ASSERT_EQUAL(a1.id, 1u);

  upstream.id_to_allocate = 2;
  alloc_id a2             = pool.do_allocate(64, 32);
  ASSERT_EQUAL(a2.id, 2u);

  pool.do_deallocate(a2, 64, 32);
  pool.do_deallocate(a1, 2048, 32);

  // make sure a good fit is used from the cache
  alloc_id a3 = pool.do_allocate(32, 32);
  ASSERT_EQUAL(a3.id, 2u);

  alloc_id a4 = pool.do_allocate(1024, 32);
  ASSERT_EQUAL(a4.id, 1u);

  pool.do_deallocate(a4, 1024, 32);

  // make sure that a new block is allocated when there's nothing cached with
  // the required alignment
  upstream.id_to_allocate = 3;
  alloc_id a5             = pool.do_allocate(32, 64);
  ASSERT_EQUAL(a5.id, 3u);

  pool.release();

  // make sure that release actually clears caches
  upstream.id_to_allocate = 4;
  alloc_id a6             = pool.do_allocate(32, 64);
  ASSERT_EQUAL(a6.id, 4u);

  upstream.id_to_allocate = 5;
  alloc_id a7             = pool.do_allocate(2048, 1024);
  ASSERT_EQUAL(a7.id, 5u);

  pool.do_deallocate(a7, 2048, 1024);

  // make sure that the 'ridiculousness' factor for size (options.cached_size_cutoff_factor)
  // is respected
  upstream.id_to_allocate = 6;
  alloc_id a8             = pool.do_allocate(24, 1024);
  ASSERT_EQUAL(a8.id, 6u);

  // make sure that the 'ridiculousness' factor for alignment (options.cached_alignment_cutoff_factor)
  // is respected
  upstream.id_to_allocate = 7;
  alloc_id a9             = pool.do_allocate(2048, 32);
  ASSERT_EQUAL(a9.id, 7u);
}

void TestDisjointUnsynchronizedPoolCachingOversized()
{
  TestDisjointPoolCachingOversized<thrust::mr::disjoint_unsynchronized_pool_resource>();
}
DECLARE_UNITTEST(TestDisjointUnsynchronizedPoolCachingOversized);

void TestDisjointSynchronizedPoolCachingOversized()
{
  TestDisjointPoolCachingOversized<thrust::mr::disjoint_synchronized_pool_resource>();
}
DECLARE_UNITTEST(TestDisjointSynchronizedPoolCachingOversized);

template <template <typename, typename> class PoolTemplate>
void TestDisjointGlobalPool()
{
  using Pool = PoolTemplate<thrust::mr::new_delete_resource, thrust::mr::new_delete_resource>;

  ASSERT_EQUAL(thrust::mr::get_global_resource<Pool>() != nullptr, true);
}

void TestUnsynchronizedDisjointGlobalPool()
{
  TestDisjointGlobalPool<thrust::mr::disjoint_unsynchronized_pool_resource>();
}
DECLARE_UNITTEST(TestUnsynchronizedDisjointGlobalPool);

void TestSynchronizedDisjointGlobalPool()
{
  TestDisjointGlobalPool<thrust::mr::disjoint_synchronized_pool_resource>();
}
DECLARE_UNITTEST(TestSynchronizedDisjointGlobalPool);

template <template <typename, typename> class PoolTemplate>
void TestDisjointPoolSqueeze()
{
  dummy_resource upstream;
  thrust::mr::new_delete_resource bookkeeper;

  using Pool = PoolTemplate<dummy_resource, thrust::mr::new_delete_resource>;

  thrust::mr::pool_options opts = Pool::get_default_options();
  opts.cache_oversized          = true;

  const std::size_t not_enough_bytes      = 3u; // free bytes that should trigger OOM
  const std::size_t small_block           = opts.min_bytes_per_chunk / 8u - 3u;
  const std::size_t medium_block          = opts.min_bytes_per_chunk + 3u;
  const std::size_t large_block           = opts.min_bytes_per_chunk * 8u - 3u;
  const std::size_t extra_large_block     = opts.largest_block_size - 3u;
  const std::size_t oversized_block       = opts.largest_block_size + 1u;
  const std::size_t many_chunks_of_blocks = 2048;

  // avoid having the destructor run when an assertion failure is raised
  // (the destructor will try to release, which in turn calls do_deallocate,
  // which may fail with an assertion failure exception...)
  Pool* pool = new Pool(&upstream, &bookkeeper, opts);

  // Test that OOM throws bad_alloc
  {
    upstream.free_bytes = not_enough_bytes;
    ASSERT_THROWS([[maybe_unused]] auto _ = pool->do_allocate(small_block), thrust::system::detail::bad_alloc);
    ASSERT_EQUAL(upstream.free_bytes, not_enough_bytes);
    upstream.assert_empty_and_reset();
  }

  {
    // Allocate several blocks from different pools + oversized:
    upstream.id_to_allocate = 1u;
    alloc_id a1             = pool->do_allocate(small_block);
    ASSERT_EQUAL(a1.id, 1u);
    ASSERT_EQUAL(upstream.id_to_allocate, 0u);

    upstream.id_to_allocate = 2u;
    alloc_id a2             = pool->do_allocate(large_block);
    ASSERT_EQUAL(a2.id, 2u);
    ASSERT_EQUAL(upstream.id_to_allocate, 0u);

    upstream.id_to_allocate = 3u;
    alloc_id a3             = pool->do_allocate(oversized_block);
    ASSERT_EQUAL(a3.id, 3u);
    ASSERT_EQUAL(upstream.id_to_allocate, 0u);

    // Simulate OOM, ensure that the allocations are still in place:
    std::size_t old_free_bytes = upstream.free_bytes;
    upstream.free_bytes        = not_enough_bytes;
    ASSERT_THROWS([[maybe_unused]] auto _ = pool->do_allocate(medium_block), thrust::system::detail::bad_alloc);
    ASSERT_THROWS([[maybe_unused]] auto _ = pool->do_allocate(oversized_block), thrust::system::detail::bad_alloc);
    ASSERT_EQUAL(upstream.free_bytes, not_enough_bytes);
    ASSERT_EQUAL(upstream.allocation_ids.size(), 3u);
    ASSERT_EQUAL(upstream.allocation_ids[0], 1u);
    ASSERT_EQUAL(upstream.allocation_ids[1], 2u);
    ASSERT_EQUAL(upstream.allocation_ids[2], 3u);
    upstream.free_bytes = old_free_bytes;

    // Allocate enough blocks to create a few more chunks and then
    // immediately deallocate them to generate a few unused chunk
    // allocations:
    std::vector<alloc_id> small_alloc_ids;
    std::vector<alloc_id> medium_alloc_ids;
    std::vector<alloc_id> large_alloc_ids;
    std::vector<alloc_id> oversized_alloc_ids;
    small_alloc_ids.reserve(many_chunks_of_blocks);
    medium_alloc_ids.reserve(many_chunks_of_blocks);
    large_alloc_ids.reserve(many_chunks_of_blocks);
    oversized_alloc_ids.reserve(many_chunks_of_blocks);
    for (std::size_t i = 0; i < many_chunks_of_blocks; ++i)
    {
      upstream.id_to_allocate = 100000u + i;
      small_alloc_ids.push_back(pool->do_allocate(small_block));
      upstream.id_to_allocate = 200000u + i;
      medium_alloc_ids.push_back(pool->do_allocate(medium_block));
      upstream.id_to_allocate = 300000u + i;
      large_alloc_ids.push_back(pool->do_allocate(large_block));
      upstream.id_to_allocate = 400000u + i;
      oversized_alloc_ids.push_back(pool->do_allocate(oversized_block));
    }
    for (const auto& alloc_id : small_alloc_ids)
    {
      pool->do_deallocate(alloc_id, small_block, alloc_id.alignment);
    }
    for (const auto& alloc_id : medium_alloc_ids)
    {
      pool->do_deallocate(alloc_id, medium_block, alloc_id.alignment);
    }
    for (const auto& alloc_id : large_alloc_ids)
    {
      pool->do_deallocate(alloc_id, large_block, alloc_id.alignment);
    }
    for (const auto& alloc_id : oversized_alloc_ids)
    {
      pool->do_deallocate(alloc_id, oversized_block, alloc_id.alignment);
    }
    small_alloc_ids.clear();
    medium_alloc_ids.clear();
    large_alloc_ids.clear();
    oversized_alloc_ids.clear();

    // Request a new allocation that exceeds the upstream free bytes.
    // Ensure that the allocation is successful and only the remaining
    // in-use allocations exist:
    upstream.free_bytes     = not_enough_bytes;
    upstream.id_to_allocate = 4u;
    alloc_id a4             = pool->do_allocate(extra_large_block);
    ASSERT_EQUAL(a4.id, 4u);
    ASSERT_EQUAL(upstream.id_to_allocate, 0u);
    ASSERT_EQUAL(upstream.allocation_ids.size(), 4u);
    ASSERT_EQUAL(upstream.allocation_ids[0], 1u);
    ASSERT_EQUAL(upstream.allocation_ids[1], 2u);
    ASSERT_EQUAL(upstream.allocation_ids[2], 3u);
    ASSERT_EQUAL(upstream.allocation_ids[3], 4u);

    pool->release();
    upstream.assert_empty_and_reset();
  }

  {
    // Allocate many chunks worth of blocks from different pools and oversized,
    // then immediately deallocate so that no chunks remain in-use.
    std::vector<alloc_id> small_alloc_ids;
    std::vector<alloc_id> medium_alloc_ids;
    std::vector<alloc_id> large_alloc_ids;
    std::vector<alloc_id> oversized_alloc_ids;
    small_alloc_ids.reserve(many_chunks_of_blocks);
    medium_alloc_ids.reserve(many_chunks_of_blocks);
    large_alloc_ids.reserve(many_chunks_of_blocks);
    oversized_alloc_ids.reserve(many_chunks_of_blocks);
    for (std::size_t i = 0; i < many_chunks_of_blocks; ++i)
    {
      upstream.id_to_allocate = 100000u + i;
      small_alloc_ids.push_back(pool->do_allocate(small_block));
      upstream.id_to_allocate = 200000u + i;
      medium_alloc_ids.push_back(pool->do_allocate(medium_block));
      upstream.id_to_allocate = 300000u + i;
      large_alloc_ids.push_back(pool->do_allocate(large_block));
      upstream.id_to_allocate = 400000u + i;
      oversized_alloc_ids.push_back(pool->do_allocate(oversized_block));
    }
    for (const auto& alloc_id : small_alloc_ids)
    {
      pool->do_deallocate(alloc_id, small_block, alloc_id.alignment);
    }
    for (const auto& alloc_id : medium_alloc_ids)
    {
      pool->do_deallocate(alloc_id, medium_block, alloc_id.alignment);
    }
    for (const auto& alloc_id : large_alloc_ids)
    {
      pool->do_deallocate(alloc_id, large_block, alloc_id.alignment);
    }
    for (const auto& alloc_id : oversized_alloc_ids)
    {
      pool->do_deallocate(alloc_id, oversized_block, alloc_id.alignment);
    }
    small_alloc_ids.clear();
    medium_alloc_ids.clear();
    large_alloc_ids.clear();
    oversized_alloc_ids.clear();

    // Request a new allocation that exceeds the upstream free bytes.
    // Ensure that the allocation is successful and only the remaining
    // in-use allocation exist:
    upstream.free_bytes     = not_enough_bytes;
    upstream.id_to_allocate = 1u;
    alloc_id a5             = pool->do_allocate(extra_large_block);
    ASSERT_EQUAL(a5.id, 1u);
    ASSERT_EQUAL(upstream.id_to_allocate, 0u);
    ASSERT_EQUAL(upstream.allocation_ids.size(), 1u);
    ASSERT_EQUAL(upstream.allocation_ids[0], 1u);

    pool->release();
    upstream.assert_empty_and_reset();
  }

  // actually destroy the pool; reasons why RAII is not used outlined at the beginning
  // of this function
  delete pool;
  ASSERT_EQUAL(upstream.id_to_deallocate, 0u);
}

void TestDisjointUnsynchronizedPoolSqueeze()
{
  TestDisjointPoolSqueeze<thrust::mr::disjoint_unsynchronized_pool_resource>();
}
DECLARE_UNITTEST(TestDisjointUnsynchronizedPoolSqueeze);

void TestDisjointSynchronizedPoolSqueeze()
{
  TestDisjointPoolSqueeze<thrust::mr::disjoint_synchronized_pool_resource>();
}
DECLARE_UNITTEST(TestDisjointSynchronizedPoolSqueeze);
