// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

//! @file
//! Cooperative global-memory prefetch hints for tiles of data, placed explicitly by algorithm agents (or kernel
//! authors) at points in the kernel schedule where the hint has lead time before the corresponding loads.

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

#include <cuda/__cmath/round_up.h>
#include <cuda/__memcpy_async/elect_one.h>
#include <cuda/__memory/align_down.h>
#include <cuda/__ptx/ptx_helper_functions.h>
#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__memory/pointer_traits.h>

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
  //! Emit an L1 prefetch hint (``prefetch.global.L1``) to all affected cache lines. Falls back to L2 on
  //! architectures that do not support real L1 prefetch.
  l1,
  //! Emit a TMA bulk prefetch into L2 (``cp.async.bulk.prefetch``) for the whole tile.
  //! Requires SM_90 or later (Hopper/Blackwell). Falls back to a no-op on older architectures.
  bulk_l2,
};

#if _CCCL_HOSTED() && !defined(_CCCL_DOXYGEN_INVOKED)
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
#endif // _CCCL_HOSTED() && !_CCCL_DOXYGEN_INVOKED

// Prefetch hints require a raw address, so only contiguous iterators qualify.
// Note: ``cub::CacheModifiedInputIterator`` does not qualify: it routes loads through an explicit cache path
// (e.g. ``LOAD_NC``) that a prefetch hint would defeat.
template <typename RandomAccessIterator>
inline constexpr bool can_prefetch_from =
  ::cuda::std::contiguous_iterator<RandomAccessIterator> && ::cuda::std::__can_to_address<RandomAccessIterator>;

//! A block-scope collective that cooperatively emits global-memory prefetch hints for a tile. The element type
//! ``T`` and block size ``ThreadsPerBlock`` are fixed on the type; the cache level (``Prefetch``) and the
//! per-cache-line hint spacing (``PrefetchStride``, bytes) are template configuration. Placement in the kernel
//! schedule is the caller's — fire it where there is lead time before the corresponding loads.
//!
//! @tparam T             Element type of the tile being prefetched (drives the byte-size math).
//! @tparam ThreadsPerBlock Number of threads in the block cooperating on the hints.
//! @tparam Prefetch      Which cache level to target (see ``LoadPrefetch``); ``none`` makes ``Prefetch()`` a no-op.
//! @tparam PrefetchStride Byte stride between successive hints; one hint per cache line (default 128 B).
template <typename T, int ThreadsPerBlock, LoadPrefetch Prefetch = LoadPrefetch::l2, int PrefetchStride = 128>
struct BlockPrefetch
{
  //! Cooperatively emit prefetch hints covering the tile ``[block_src_it, block_src_it + items_to_prefetch)``
  template <typename RandomAccessIterator>
  static _CCCL_DEVICE _CCCL_FORCEINLINE void Prefetch(RandomAccessIterator block_src_it, int items_to_prefetch)
  {
    if constexpr (Prefetch != LoadPrefetch::none && can_prefetch_from<RandomAccessIterator>)
    {
      static_assert(sizeof(cub::detail::it_value_t<RandomAccessIterator>) == sizeof(T),
                    "BlockPrefetch element type T must match the iterator's value type size");
      const int linear_tid           = static_cast<int>(threadIdx.x);
      const unsigned int total_bytes = static_cast<unsigned int>(items_to_prefetch) * unsigned{sizeof(T)};
      const auto* const src_ptr      = reinterpret_cast<const char*>(::cuda::std::to_address(block_src_it));

      if constexpr (Prefetch == LoadPrefetch::bulk_l2)
      {
        // One elected thread issues a single TMA bulk prefetch for the whole tile.
        // cp.async.bulk.prefetch is fire-and-forget: no commit_group/wait_group needed.
        // Requires SM_90+; a no-op on older architectures.
        NV_IF_TARGET(NV_PROVIDES_SM_90, (if (::cuda::device::__block_elect_one()) {
                       // srcMem must be 16-byte aligned per PTX ISA; align base down and extend size to compensate
                       const auto* const aligned_base  = ::cuda::align_down(src_ptr, 16);
                       const unsigned int prefix       = static_cast<unsigned int>(src_ptr - aligned_base);
                       const unsigned int aligned_size = ::cuda::round_up(total_bytes + prefix, 16u);
                       if (aligned_size > 0)
                       {
                         asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                                      :
                                      : "l"(::cuda::ptx::__as_ptr_gmem(aligned_base)), "r"(aligned_size)
                                      : "memory");
                       }
                     }))
      }
      else
      {
        _CCCL_PRAGMA_NOUNROLL()
        for (unsigned int offset = static_cast<unsigned int>(linear_tid) * PrefetchStride; offset < total_bytes;
             offset += static_cast<unsigned int>(ThreadsPerBlock) * PrefetchStride)
        {
          // TODO: replace with cuda::ptx::prefetch_L1/L2 once exposed in libcudacxx
          if constexpr (Prefetch == LoadPrefetch::l1)
          {
            asm volatile("prefetch.global.L1 [%0];" : : "l"(::cuda::ptx::__as_ptr_gmem(src_ptr + offset)) : "memory");
          }
          else
          {
            asm volatile("prefetch.global.L2 [%0];" : : "l"(::cuda::ptx::__as_ptr_gmem(src_ptr + offset)) : "memory");
          }
        }
      }
    }
  }
};
} // namespace detail

CUB_NAMESPACE_END
