// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Cooperative global-memory prefetch hints for tiles of data, placed explicitly in device code
//! at points where the hint has lead time before the corresponding loads.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_type.cuh>

#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/unwrap_contiguous_iterator.h>

#include <cuda/__memcpy_async/elect_one.h>
#include <cuda/__memory/align_down.h>
#include <cuda/__memory/align_up.h>
#include <cuda/__memory/ptr_rebind.h>
#include <cuda/std/__host_stdlib/ostream>

#include <nv/target>

CUB_NAMESPACE_BEGIN

namespace detail
{
//! Enumerates the cache levels a tile prefetch hint can target.
enum class LoadPrefetch : int
{
  //! No prefetch hint emitted.
  none,
  //! Emit an L2 prefetch hint (``prefetch.global.L2``) to all affected cache lines.
  l2,
  //! Emit an L1 prefetch hint (``prefetch.global.L1``) to all affected cache lines. The emitted instruction is
  //! always ``prefetch.global.L1``; on architectures without real L1 prefetch the hardware itself services the
  //! hint as an L2 prefetch (e.g. on Blackwell).
  l1,
  //! Emit a TMA bulk prefetch into L2 (``cp.async.bulk.prefetch``) for the whole tile on SM_90 or later
  //! (Hopper/Blackwell). On older architectures falls back to the same strided per-cache-line L2 prefetch
  //! as ``l2``, so the tile still reaches L2 — via a different mechanism with different (unmeasured) cost.
  bulk_l2,
};

#if _CCCL_HOSTED()
inline ::std::ostream& operator<<(::std::ostream& os, LoadPrefetch prefetch)
{
  switch (prefetch)
  {
    case LoadPrefetch::none:
      return os << "detail::LoadPrefetch::none";
    case LoadPrefetch::l2:
      return os << "detail::LoadPrefetch::l2";
    case LoadPrefetch::l1:
      return os << "detail::LoadPrefetch::l1";
    case LoadPrefetch::bulk_l2:
      return os << "detail::LoadPrefetch::bulk_l2";
    default:
      return os << "<unknown detail::LoadPrefetch: " << static_cast<int>(prefetch) << ">";
  }
}
#endif // _CCCL_HOSTED()

// Prefetch hints require a raw address, so only contiguous iterators qualify. thrust::is_contiguous_iterator
// covers raw pointers, std/libcu++ contiguous iterators, thrust vector iterators, and proclaimed types.
// TODO(#9893): replace with the unified CCCL iterator-to-pointer mechanism once it exists.
// Note: ``cub::CacheModifiedInputIterator`` does not qualify: it routes loads through an explicit cache path
// (e.g. ``LOAD_NC``) that a prefetch hint would defeat.
template <typename It>
inline constexpr bool can_prefetch_from = THRUST_NS_QUALIFIER::is_contiguous_iterator_v<It>;

//! A block-scope collective that cooperatively emits global-memory prefetch hints for a tile. The block size
//! ``ThreadsPerBlock`` is fixed on the type; the cache level (``PrefetchLevel``) and the per-cache-line hint
//! spacing (``PrefetchStride``, bytes) are template configuration. Placement in the kernel schedule is the
//! caller's — fire it where there is lead time before the corresponding loads.
//!
//! @tparam ThreadsPerBlock Number of threads in the block cooperating on the hints.
//! @tparam PrefetchLevel  Which cache level to target (see ``LoadPrefetch``); ``none`` makes ``Prefetch()`` a no-op.
//! @tparam PrefetchStride Byte stride between successive hints; one hint per cache line (default 128 B).
template <int ThreadsPerBlock, LoadPrefetch PrefetchLevel = LoadPrefetch::l2, int PrefetchStride = 128>
struct BlockPrefetch
{
  static_assert(ThreadsPerBlock > 0, "ThreadsPerBlock must be positive");
  static_assert(PrefetchStride > 0, "PrefetchStride must be positive");

  //! Cooperatively emit prefetch hints covering the tile ``[tile_base, tile_base + items_to_prefetch)``.
  //!
  //! @param tile_base Iterator to the first item of the calling block's tile.
  //! @param items_to_prefetch Total number of items in the tile, across all threads of the block — NOT a
  //!   per-thread count. Must be non-negative.
  template <typename It>
  static _CCCL_DEVICE_API _CCCL_FORCEINLINE void Prefetch(It tile_base, int items_to_prefetch)
  {
    if constexpr (PrefetchLevel != LoadPrefetch::none && can_prefetch_from<It>)
    {
      // The tile is strided by the linear thread id, which this implementation derives from threadIdx.x alone
      _CCCL_ASSERT(blockDim.y == 1 && blockDim.z == 1, "BlockPrefetch requires a one-dimensional thread block");
      _CCCL_ASSERT(items_to_prefetch >= 0, "items_to_prefetch must be non-negative");

      const int total_bytes = items_to_prefetch * int{sizeof(it_value_t<It>)};
      const auto src_ptr    = ::cuda::ptr_rebind<char>(THRUST_NS_QUALIFIER::unwrap_contiguous_iterator(tile_base));

      if constexpr (PrefetchLevel == LoadPrefetch::bulk_l2)
      {
        // One elected thread issues a single TMA bulk prefetch for the whole tile.
        // cp.async.bulk.prefetch is fire-and-forget: no commit_group/wait_group needed.
        // Requires SM_90+; on older architectures fall back to the strided L2 prefetch.
        NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
                          ({
                            if (::cuda::device::__block_elect_one())
                            {
                              // srcMem must be 16-byte aligned per PTX ISA; align the range outward on both ends to
                              // compensate
                              const auto* aligned_base = ::cuda::align_down(src_ptr, 16);
                              const auto* aligned_end  = ::cuda::align_up(src_ptr + total_bytes, 16);
                              const auto aligned_size  = static_cast<unsigned>(aligned_end - aligned_base);
                              asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                                           :
                                           : "l"(::__cvta_generic_to_global(aligned_base)), "r"(aligned_size)
                                           : "memory");
                            }
                          }),
                          ({ __strided_prefetch<LoadPrefetch::l2>(src_ptr, total_bytes); }))
      }
      else
      {
        __strided_prefetch<PrefetchLevel>(src_ptr, total_bytes);
      }
    }
  }

private:
  //! The block's threads cooperatively walk ``[src_ptr, src_ptr + total_bytes)`` in ``PrefetchStride``-byte
  //! steps, each issuing one prefetch hint per cache line targeting ``Level`` (``l1`` or ``l2``).
  template <LoadPrefetch Level>
  static _CCCL_DEVICE_API _CCCL_FORCEINLINE void __strided_prefetch(const char* src_ptr, int total_bytes)
  {
    const int start    = static_cast<int>(threadIdx.x) * PrefetchStride;
    constexpr int step = ThreadsPerBlock * PrefetchStride;
    _CCCL_PRAGMA_NOUNROLL()
    for (int offset = start; offset < total_bytes; offset += step)
    {
      const auto gmem_ptr = ::__cvta_generic_to_global(src_ptr + offset);
      // TODO: replace with cuda::ptx::prefetch_L1/L2 once exposed in libcudacxx
      if constexpr (Level == LoadPrefetch::l1)
      {
        asm volatile("prefetch.global.L1 [%0];" : : "l"(gmem_ptr) : "memory");
      }
      else
      {
        asm volatile("prefetch.global.L2 [%0];" : : "l"(gmem_ptr) : "memory");
      }
    }
  }
};
} // namespace detail

CUB_NAMESPACE_END
