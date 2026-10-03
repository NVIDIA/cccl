// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/function.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/detail/static_assert.h> // for depend_on_instantiation
#include <thrust/detail/temporary_array.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/sequential/copy_if.h>
#include <thrust/system/omp/detail/execution_policy.h>
#include <thrust/system/omp/detail/pragma_omp.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/fill_n.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/atomic>
#include <cuda/std/cstdint>

// don't attempt to #include this file without omp support
#if (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
#  include <omp.h>
#endif // omp support

THRUST_NAMESPACE_BEGIN
namespace system::omp::detail
{
// Below this many elements, copy_if runs on the calling thread
inline constexpr int parallel_copy_if_threshold = 1024;

namespace copy_if_detail
{
// Measured on a 64-core Grace with 1- to 32-byte elements: small tiles spread a costly predicate over all threads just
// above the threshold, while more and smaller tiles make the shared tile counter a bottleneck for large inputs, and
// larger tiles lose cache locality for large elements.
inline constexpr int min_tile_size = 64;
inline constexpr int max_tile_size = 8192;

using status_word = ::cuda::std::uint64_t;

// A tile's status word holds a number of selected elements, shifted left by two, and one of these flags
inline constexpr status_word status_partial = 1; // selected elements in this tile
inline constexpr status_word status_prefix  = 2; // selected elements in this tile and all earlier ones
} // namespace copy_if_detail

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename Predicate>
OutputIterator copy_if(
  execution_policy<DerivedPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 stencil,
  OutputIterator result,
  Predicate pred)
{
  // we're attempting to launch an omp kernel, assert we're compiling with omp support
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to enable OpenMP support in your compiler.                  X
  // ========================================================================
  static_assert(thrust::detail::depend_on_instantiation<InputIterator1,
                                                        (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)>::value,
                "OpenMP compiler support is not enabled");

  using Size = thrust::detail::it_difference_t<InputIterator1>;
  using copy_if_detail::status_word;

  const Size n = ::cuda::std::distance(first, last);

#if (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
  const int num_threads = omp_get_max_threads();
#else
  const int num_threads = 1;
#endif // THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE

  if (n < parallel_copy_if_threshold || num_threads <= 1)
  {
    return thrust::system::detail::sequential::copy_if(exec, first, last, stencil, result, pred);
  }

  // Single pass with decoupled look-back. Threads take tiles in increasing order from a shared counter. For its tile, a
  // thread evaluates pred once per element into a local flag buffer and publishes the tile's count in the tile's status
  // word. It then adds up the counts of the preceding tiles, walking back until it finds a published prefix, publishes
  // its own prefix, and copies the selected elements of the tile to their final position.
  //
  // Progress: a thread only waits for the status of an earlier tile. That tile was handed out earlier, so a thread is
  // working on it, and that thread publishes the tile's count before it waits for anything. The waits cannot form a
  // cycle and only require the owners of earlier tiles to get scheduled, whatever the team size. Publishers call
  // atomic_ref::notify_all for the waiters. libcu++ implements the host wait by polling with a back-off from spinning
  // to yielding and sleeping, which it does on an idle machine too, and which lets more threads than cores progress.
  //
  // Memory order: a status word packs the flag with the count, so a reader never sees one without the other, and the
  // counts are the only data a thread reads from another one. The output ranges of the tiles are disjoint and only read
  // after the barrier that ends the parallel region, so relaxed accesses are sufficient.
  const Size tile_size = ::cuda::std::clamp(
    ::cuda::ceil_div(n, static_cast<Size>(num_threads)),
    static_cast<Size>(copy_if_detail::min_tile_size),
    static_cast<Size>(copy_if_detail::max_tile_size));
  const Size num_tiles = ::cuda::ceil_div(n, tile_size);

  thrust::detail::temporary_array<status_word, DerivedPolicy> status_storage(exec, num_tiles);
  status_word* const tile_status = thrust::raw_pointer_cast(status_storage.data());
  ::cuda::std::fill_n(tile_status, num_tiles, status_word{0});

  ::cuda::std::atomic<Size> next_tile{0};

  // Set by the thread that handles the last tile, or by a team of one thread
  Size num_selected = 0;

  const thrust::detail::wrapped_function<Predicate, bool> wrapped_pred{pred};

  THRUST_PRAGMA_OMP(parallel)
  {
#if (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
    const bool team_of_one = omp_get_num_threads() == 1;
#else
    const bool team_of_one = true;
#endif // THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE

    if (team_of_one)
    {
      // E.g. inside another parallel region with nested parallelism disabled, where tiles only add overhead
      num_selected = static_cast<Size>(::cuda::std::distance(
        result, thrust::system::detail::sequential::copy_if(exec, first, last, stencil, result, pred)));
    }
    else
    {
      bool selected[copy_if_detail::max_tile_size];

      for (;;)
      {
        const Size tile = next_tile.fetch_add(1, ::cuda::std::memory_order_relaxed);
        if (tile >= num_tiles)
        {
          break;
        }

        const Size tile_begin = tile * tile_size;
        const Size tile_items = (::cuda::std::min) (tile_size, n - tile_begin);

        Size count                = 0;
        InputIterator2 stencil_it = stencil + tile_begin;
        for (Size i = 0; i < tile_items; ++i, ++stencil_it)
        {
          selected[i] = wrapped_pred(*stencil_it);
          count += static_cast<Size>(selected[i]);
        }

        // Number of selected elements in all earlier tiles
        Size offset = 0;
        ::cuda::std::atomic_ref<status_word> own_status(tile_status[tile]);
        if (tile > 0)
        {
          own_status.store((static_cast<status_word>(count) << 2) | copy_if_detail::status_partial,
                           ::cuda::std::memory_order_relaxed);
          own_status.notify_all();
          for (Size earlier = tile - 1;; --earlier)
          {
            ::cuda::std::atomic_ref<status_word> earlier_status(tile_status[earlier]);
            status_word word;
            while ((word = earlier_status.load(::cuda::std::memory_order_relaxed)) == 0)
            {
              earlier_status.wait(status_word{0}, ::cuda::std::memory_order_relaxed);
            }
            offset += static_cast<Size>(word >> 2);
            if (word & copy_if_detail::status_prefix)
            {
              break;
            }
          }
        }
        own_status.store((static_cast<status_word>(offset + count) << 2) | copy_if_detail::status_prefix,
                         ::cuda::std::memory_order_relaxed);
        own_status.notify_all();

        if (tile == num_tiles - 1)
        {
          num_selected = offset + count;
        }

        OutputIterator out = result + offset;
        InputIterator1 in  = first + tile_begin;
        for (Size i = 0; i < tile_items; ++i, ++in)
        {
          if (selected[i])
          {
            *out = *in;
            ++out;
          }
        }
      }
    }
  }

  return result + num_selected;
} // end copy_if()
} // end namespace system::omp::detail
THRUST_NAMESPACE_END
