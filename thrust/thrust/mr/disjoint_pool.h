/*
 *  Copyright 2018 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file
 *  \brief A caching and pooling memory resource adaptor which uses separate upstream resources for memory allocation
 *      and bookkeeping.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/config.h>

#include <thrust/binary_search.h>
#include <thrust/detail/seq.h>
#include <thrust/find.h>
#include <thrust/host_vector.h>
#include <thrust/mr/allocator.h>
#include <thrust/mr/memory_resource.h>
#include <thrust/mr/pool_options.h>

#include <cuda/__cmath/ilog.h>
#include <cuda/__cmath/pow2.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__cccl/algorithm_wrapper.h>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

THRUST_NAMESPACE_BEGIN
namespace mr
{
/** \addtogroup memory_resources Memory Resources
 *  \ingroup memory_management
 *  \{
 */

/*! A memory resource adaptor allowing for pooling and caching allocations from \p Upstream, using \p Bookkeeper for
 *      management of that cached and pooled memory, allowing to cache portions of memory inaccessible from the host.
 *
 *  On a typical memory resource, calls to \p allocate and \p deallocate actually allocate and deallocate memory.
 * Pooling memory resources only allocate and deallocate memory from an external resource (the upstream memory resource)
 * when there's no suitable memory currently cached; otherwise, they use memory they have acquired beforehand, to make
 *      memory allocation faster and more efficient.
 *
 *  The disjoint version of the pool resources uses a separate upstream memory resource, \p Bookkeeper, to allocate
 * memory necessary to manage the cached memory. There may be many reasons to do that; the canonical one is that \p
 * Upstream allocates memory that is inaccessible to the code of the pool resource, which means that it cannot embed the
 * necessary information in memory obtained from \p Upstream; for instance, \p Upstream can be a CUDA non-managed memory
 *      resource, or a CUDA managed memory resource whose memory we would prefer to not migrate back and forth between
 *      host and device when executing bookkeeping code.
 *
 *  This is not the only case where it makes sense to use a disjoint pool resource, though. In a multi-core environment
 *      it may be beneficial to avoid stealing cache lines from other cores by writing over bookkeeping information
 *      embedded in an allocated block of memory. In such a case, one can imagine wanting to use a disjoint pool where
 *      both the upstream and the bookkeeper are of the same type, to allocate memory consistently, but separately for
 *      those two purposes.
 *
 *  \tparam Upstream the type of memory resources that will be used for allocating memory blocks to be handed off to the
 * user \tparam Bookkeeper the type of memory resources that will be used for allocating bookkeeping memory
 */
template <typename Upstream, typename Bookkeeper>
class disjoint_unsynchronized_pool_resource final
    : public memory_resource<typename Upstream::pointer>
    , private validator2<Upstream, Bookkeeper>
{
public:
  /*! Get the default options for a disjoint pool. These are meant to be a sensible set of values for many use cases,
   *      and as such, may be tuned in the future. This function is exposed so that creating a set of options that are
   *      just a slight departure from the defaults is easy.
   */
  static pool_options get_default_options()
  {
    pool_options ret;

    ret.min_blocks_per_chunk = 16;
    ret.min_bytes_per_chunk  = 1024;
    ret.max_blocks_per_chunk = static_cast<std::size_t>(1) << 20;
    ret.max_bytes_per_chunk  = static_cast<std::size_t>(1) << 30;

    ret.smallest_block_size = THRUST_MR_DEFAULT_ALIGNMENT;
    ret.largest_block_size  = static_cast<std::size_t>(1) << 20;

    ret.alignment = THRUST_MR_DEFAULT_ALIGNMENT;

    ret.cache_oversized = true;

    ret.cached_size_cutoff_factor      = 16;
    ret.cached_alignment_cutoff_factor = 16;

    return ret;
  }

  /*! Constructor.
   *
   *  \param upstream the upstream memory resource for allocations
   *  \param bookkeeper the upstream memory resource for bookkeeping
   *  \param options pool options to use
   */
  disjoint_unsynchronized_pool_resource(
    Upstream* upstream, Bookkeeper* bookkeeper, pool_options options = get_default_options())
      : m_upstream(upstream)
      , m_bookkeeper(bookkeeper)
      , m_options(options)
      , m_smallest_block_log2(::cuda::ceil_ilog2(m_options.smallest_block_size))
      , m_pools(m_bookkeeper)
      , m_allocated(m_bookkeeper)
      , m_cached_oversized(m_bookkeeper)
      , m_oversized(m_bookkeeper)
  {
    assert(m_options.validate());

    pointer_vector free(m_bookkeeper);
    pool p(free);
    m_pools.resize(::cuda::ceil_ilog2(m_options.largest_block_size) - m_smallest_block_log2 + 1, p);
  }

  // TODO: C++11: use delegating constructors

  /*! Constructor. Upstream and bookkeeping resources are obtained by calling \p get_global_resource for their types.
   *
   *  \param options pool options to use
   */
  disjoint_unsynchronized_pool_resource(pool_options options = get_default_options())
      : m_upstream(get_global_resource<Upstream>())
      , m_bookkeeper(get_global_resource<Bookkeeper>())
      , m_options(options)
      , m_smallest_block_log2(::cuda::ceil_ilog2(m_options.smallest_block_size))
      , m_pools(m_bookkeeper)
      , m_allocated(m_bookkeeper)
      , m_cached_oversized(m_bookkeeper)
      , m_oversized(m_bookkeeper)
  {
    assert(m_options.validate());

    pointer_vector free(m_bookkeeper);
    pool p(free);
    m_pools.resize(::cuda::ceil_ilog2(m_options.largest_block_size) - m_smallest_block_log2 + 1, p);
  }

  /*! Destructor. Releases all held memory to upstream.
   */
  ~disjoint_unsynchronized_pool_resource()
  {
    release();
  }

private:
  using void_ptr = typename Upstream::pointer;
  using char_ptr = typename thrust::detail::pointer_traits<void_ptr>::template rebind<char>::other;

  struct chunk_descriptor
  {
    std::size_t size;
    void_ptr pointer;
    std::size_t pool_idx;
  };

  using chunk_vector = thrust::host_vector<chunk_descriptor, allocator<chunk_descriptor, Bookkeeper>>;

  struct oversized_block_descriptor
  {
    std::size_t size;
    std::size_t alignment;
    void_ptr pointer;

    _CCCL_HOST_DEVICE bool operator==(const oversized_block_descriptor& other) const
    {
      return size == other.size && alignment == other.alignment && pointer == other.pointer;
    }

    _CCCL_HOST_DEVICE bool operator<(const oversized_block_descriptor& other) const
    {
      return size < other.size || (size == other.size && alignment < other.alignment);
    }
  };

  struct equal_pointers
  {
  public:
    _CCCL_HOST_DEVICE equal_pointers(void_ptr p)
        : p(p)
    {}

    _CCCL_HOST_DEVICE bool operator()(const oversized_block_descriptor& desc) const
    {
      return desc.pointer == p;
    }

  private:
    void_ptr p;
  };

  struct matching_alignment
  {
  public:
    _CCCL_HOST_DEVICE matching_alignment(std::size_t requested)
        : requested(requested)
    {}

    _CCCL_HOST_DEVICE bool operator()(const oversized_block_descriptor& desc) const
    {
      return desc.alignment >= requested;
    }

  private:
    std::size_t requested;
  };

  using oversized_block_vector =
    thrust::host_vector<oversized_block_descriptor, allocator<oversized_block_descriptor, Bookkeeper>>;

  using pointer_vector = thrust::host_vector<void_ptr, allocator<void_ptr, Bookkeeper>>;

  struct pool
  {
    _CCCL_HOST pool(const pointer_vector& free)
        : free_blocks(free)
        , previous_allocated_count(0)
    {}

    _CCCL_HOST pool(const pool& other)
        : free_blocks(other.free_blocks)
        , previous_allocated_count(other.previous_allocated_count)
    {}

    _CCCL_EXEC_CHECK_DISABLE
    pool& operator=(const pool&) = default;

    _CCCL_HOST ~pool() {}

    pointer_vector free_blocks;
    std::size_t previous_allocated_count;
  };

  using pool_vector = thrust::host_vector<pool, allocator<pool, Bookkeeper>>;

  Upstream* m_upstream;
  Bookkeeper* m_bookkeeper;

  pool_options m_options;
  std::size_t m_smallest_block_log2;

  // buckets containing free lists for each pooled size
  pool_vector m_pools;
  // list of all allocations from upstream for the above
  chunk_vector m_allocated;
  // list of all cached oversized/overaligned blocks that have been returned to the pool to cache
  oversized_block_vector m_cached_oversized;
  // list of all oversized/overaligned allocations from upstream
  oversized_block_vector m_oversized;

public:
  /*! Releases all held memory to upstream.
   */
  void release()
  {
    // reset the buckets
    for (std::size_t i = 0; i < m_pools.size(); ++i)
    {
      m_pools[i].free_blocks.clear();
      m_pools[i].previous_allocated_count = 0;
    }

    // deallocate memory allocated for the buckets
    for (std::size_t i = 0; i < m_allocated.size(); ++i)
    {
      m_upstream->do_deallocate(m_allocated[i].pointer, m_allocated[i].size, m_options.alignment);
    }

    // deallocate cached oversized/overaligned memory
    for (std::size_t i = 0; i < m_oversized.size(); ++i)
    {
      m_upstream->do_deallocate(m_oversized[i].pointer, m_oversized[i].size, m_oversized[i].alignment);
    }

    m_allocated.clear();
    m_oversized.clear();
    m_cached_oversized.clear();
  }

  void squeeze()
  {
    // Find all unused chunks and deallocate them
    for (auto it = m_allocated.begin(); it != m_allocated.end();)
    {
      const auto pool_idx = (*it).pool_idx;
      auto& pool          = m_pools[pool_idx];

      const std::size_t bytes_log2  = pool_idx + m_smallest_block_log2;
      const std::size_t bucket_size = static_cast<std::size_t>(1) << bytes_log2;
      const std::size_t n           = (*it).size / bucket_size;
      assert((*it).size % bucket_size == 0);

      bool in_use = false;
      for (std::size_t i = 0; i < n; ++i)
      {
        const auto ptr = static_cast<void_ptr>(static_cast<char_ptr>((*it).pointer) + i * bucket_size);
        if (find(pool.free_blocks.begin(), pool.free_blocks.end(), ptr) == pool.free_blocks.end())
        {
          in_use = true;
          break;
        }
      }

      if (!in_use)
      {
        // Remove all free blocks cut from this chunk:
        for (std::size_t i = 0; i < n; ++i)
        {
          const auto ptr = static_cast<void_ptr>(static_cast<char_ptr>((*it).pointer) + i * bucket_size);
          pool.free_blocks.erase(find(pool.free_blocks.begin(), pool.free_blocks.end(), ptr));
        }

        // Deallocate and remove this chunk from the list of allocated chunks
        m_upstream->do_deallocate((*it).pointer, (*it).size, m_options.alignment);
        it = m_allocated.erase(it);
      }
      else
      {
        ++it;
      }
    }

    // Remove all cached oversized allocations
    for (auto it = m_cached_oversized.begin(); it != m_cached_oversized.end();)
    {
      m_upstream->do_deallocate((*it).pointer, (*it).size, (*it).alignment);
      m_oversized.erase(find(m_oversized.begin(), m_oversized.end(), *it));
      it = m_cached_oversized.erase(it);
    }
  }

  [[nodiscard]] virtual void_ptr
  do_allocate(std::size_t bytes, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) override
  {
    try
    {
      return do_allocate_impl(bytes, alignment);
    }
    catch (std::bad_alloc&)
    {
      this->squeeze();
    }

    return do_allocate_impl(bytes, alignment);
  }

  [[nodiscard]] void_ptr do_allocate_impl(std::size_t bytes, std::size_t alignment)
  {
    bytes = (std::max) (bytes, m_options.smallest_block_size);
    assert(::cuda::is_power_of_two(alignment));

    // an oversized and/or overaligned allocation requested; needs to be allocated separately
    if (bytes > m_options.largest_block_size || alignment > m_options.alignment)
    {
      oversized_block_descriptor oversized;
      oversized.size      = bytes;
      oversized.alignment = alignment;

      if (m_options.cache_oversized && !m_cached_oversized.empty())
      {
        typename oversized_block_vector::iterator it =
          thrust::lower_bound(thrust::seq, m_cached_oversized.begin(), m_cached_oversized.end(), oversized);

        // if the size is bigger than the requested size by a factor
        // bigger than or equal to the specified cutoff for size,
        // allocate a new block
        if (it != m_cached_oversized.end())
        {
          std::size_t size_factor = (*it).size / bytes;
          if (size_factor >= m_options.cached_size_cutoff_factor)
          {
            it = m_cached_oversized.end();
          }
        }

        if (it != m_cached_oversized.end() && (*it).alignment < alignment)
        {
          it = find_if(it + 1, m_cached_oversized.end(), matching_alignment(alignment));
        }

        // if the alignment is bigger than the requested one by a factor
        // bigger than or equal to the specified cutoff for alignment,
        // allocate a new block
        if (it != m_cached_oversized.end())
        {
          std::size_t alignment_factor = (*it).alignment / alignment;
          if (alignment_factor >= m_options.cached_alignment_cutoff_factor)
          {
            it = m_cached_oversized.end();
          }
        }

        if (it != m_cached_oversized.end())
        {
          oversized.pointer = (*it).pointer;
          m_cached_oversized.erase(it);
          return oversized.pointer;
        }
      }

      // no fitting cached block found; allocate a new one that's just up to the specs
      oversized.pointer = m_upstream->do_allocate(bytes, alignment);
      m_oversized.push_back(oversized);

      return oversized.pointer;
    }

    // the request is NOT for oversized and/or overaligned memory
    // allocate a block from an appropriate bucket
    std::size_t bytes_log2 = ::cuda::ceil_ilog2(bytes);
    std::size_t pool_idx   = bytes_log2 - m_smallest_block_log2;
    pool& bucket           = m_pools[pool_idx];

    // if the free list of the bucket has no elements, allocate a new chunk
    // and split it into blocks pushed to the free list
    if (bucket.free_blocks.empty())
    {
      std::size_t bucket_size = static_cast<std::size_t>(1) << bytes_log2;

      std::size_t n = bucket.previous_allocated_count;
      if (n == 0)
      {
        n = ::cuda::std::max(m_options.min_blocks_per_chunk, //
                             m_options.min_bytes_per_chunk >> bytes_log2);
      }
      else
      {
        n = ::cuda::std::min({n * 3 / 2, //
                              m_options.max_bytes_per_chunk >> bytes_log2,
                              m_options.max_blocks_per_chunk});
      }

      bytes = n << bytes_log2;

      assert(n >= m_options.min_blocks_per_chunk);
      assert(n <= m_options.max_blocks_per_chunk);
      assert(bytes >= m_options.min_bytes_per_chunk);
      assert(bytes <= m_options.max_bytes_per_chunk);

      chunk_descriptor allocated;
      allocated.size     = bytes;
      allocated.pointer  = m_upstream->do_allocate(bytes, m_options.alignment);
      allocated.pool_idx = pool_idx;
      m_allocated.push_back(allocated);
      bucket.previous_allocated_count = n;

      for (std::size_t i = 0; i < n; ++i)
      {
        bucket.free_blocks.push_back(static_cast<void_ptr>(static_cast<char_ptr>(allocated.pointer) + i * bucket_size));
      }
    }

    // allocate a block from the front of the bucket's free list
    void_ptr ret = bucket.free_blocks.back();
    bucket.free_blocks.pop_back();
    return ret;
  }

  virtual void do_deallocate(void_ptr p, std::size_t n, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) override
  {
    n = (std::max) (n, m_options.smallest_block_size);
    assert(::cuda::is_power_of_two(alignment));

    // verify that the pointer is at least as aligned as claimed
    assert(reinterpret_cast<::cuda::std::intmax_t>(detail::pointer_traits<void_ptr>::get(p)) % alignment == 0);

    // the deallocated block is oversized and/or overaligned
    if (n > m_options.largest_block_size || alignment > m_options.alignment)
    {
      typename oversized_block_vector::iterator it = find_if(m_oversized.begin(), m_oversized.end(), equal_pointers(p));
      assert(it != m_oversized.end());

      oversized_block_descriptor oversized = *it;

      if (m_options.cache_oversized)
      {
        typename oversized_block_vector::iterator position =
          lower_bound(m_cached_oversized.begin(), m_cached_oversized.end(), oversized);
        m_cached_oversized.insert(position, oversized);
        return;
      }

      m_oversized.erase(it);

      m_upstream->do_deallocate(p, oversized.size, oversized.alignment);

      return;
    }

    // push the block to the front of the appropriate bucket's free list
    std::size_t n_log2   = ::cuda::ceil_ilog2(n);
    std::size_t pool_idx = n_log2 - m_smallest_block_log2;
    pool& bucket         = m_pools[pool_idx];

    bucket.free_blocks.push_back(p);
  }
};

/*! \} // memory_resource
 */
} // namespace mr
THRUST_NAMESPACE_END
