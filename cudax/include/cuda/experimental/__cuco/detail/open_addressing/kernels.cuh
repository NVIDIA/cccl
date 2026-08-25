//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_DETAIL_OPEN_ADDRESSING_KERNELS_CUH
#define _CUDAX___CUCO_DETAIL_OPEN_ADDRESSING_KERNELS_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_reduce.cuh>

#include <cuda/__atomic/atomic.h>
#include <cuda/__memory/uninitialized_array.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/__cuco/detail/utility/cuda.cuh>

#include <cooperative_groups.h>

#include <cooperative_groups/reduce.h>
#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CUDA_COMPILATION()

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")

namespace cuda::experimental::cuco::__open_addressing
{
//! @brief Scalar (cooperative-group size 1) functor inserting `first[i]` when `pred(stencil[i])` holds.
template <class _InputIt, class _StencilIt, class _Predicate, class _Ref>
struct __insert_if_fn
{
  _InputIt __first;
  _StencilIt __stencil;
  _Predicate __pred;
  _Ref __ref;

  _CCCL_DEVICE_API void operator()(detail::__index_type __idx)
  {
    if (__pred(*(__stencil + __idx)))
    {
      __ref.insert(*(__first + __idx));
    }
  }
};

template <class _InputIt, class _StencilIt, class _Predicate, class _Ref>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES __insert_if_fn(_InputIt, _StencilIt, _Predicate, _Ref)
  -> __insert_if_fn<_InputIt, _StencilIt, _Predicate, _Ref>;

//! @brief Scalar (cooperative-group size 1) functor writing `pred(stencil[i]) ? contains(first[i]) : false`.
template <class _InputIt, class _StencilIt, class _Predicate, class _OutputIt, class _Ref>
struct __contains_if_fn
{
  _InputIt __first;
  _StencilIt __stencil;
  _Predicate __pred;
  _OutputIt __output_begin;
  _Ref __ref;

  _CCCL_DEVICE_API void operator()(detail::__index_type __idx) const
  {
    *(__output_begin + __idx) = __pred(*(__stencil + __idx)) ? __ref.contains(*(__first + __idx)) : false;
  }
};

template <class _InputIt, class _StencilIt, class _Predicate, class _OutputIt, class _Ref>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES __contains_if_fn(_InputIt, _StencilIt, _Predicate, _OutputIt, _Ref)
  -> __contains_if_fn<_InputIt, _StencilIt, _Predicate, _OutputIt, _Ref>;

//! @brief Inserts all elements in the range `[first, first + n)` and returns the number of
//! successful insertions if `pred` of the corresponding stencil returns true.
template <int _CgSize, int _BlockSize, class _InputIt, class _StencilIt, class _Predicate, class _Ref>
_CCCL_KERNEL_ATTRIBUTES _CCCL_LAUNCH_BOUNDS(_BlockSize) void __insert_if_n(
  _InputIt __first,
  detail::__index_type __n,
  _StencilIt __stencil,
  _Predicate __pred,
  typename _Ref::size_type* __num_successes,
  _Ref __ref)
{
  using __block_reduce = CUB_NS_QUALIFIER::BlockReduce<typename _Ref::size_type, _BlockSize>;
  __shared__ typename __block_reduce::TempStorage __temp_storage;
  typename _Ref::size_type __thread_num_successes = 0;

  const auto __loop_stride = detail::__grid_stride() / _CgSize;
  auto __idx               = detail::__global_thread_id() / _CgSize;

  while (__idx < __n)
  {
    if (__pred(*(__stencil + __idx)))
    {
      using __value_t = typename ::cuda::std::iterator_traits<_InputIt>::value_type;
      const __value_t __insert_element{*(__first + __idx)};
      if constexpr (_CgSize == 1)
      {
        if (__ref.insert(__insert_element))
        {
          __thread_num_successes++;
        }
      }
      else
      {
        const auto __tile = ::cooperative_groups::tiled_partition<_CgSize, ::cooperative_groups::thread_block>(
          ::cooperative_groups::this_thread_block());
        if (__ref.insert(__tile, __insert_element) && __tile.thread_rank() == 0)
        {
          __thread_num_successes++;
        }
      }
    }
    __idx += __loop_stride;
  }

  const auto __block_num_successes = __block_reduce(__temp_storage).Sum(__thread_num_successes);
  if (threadIdx.x == 0)
  {
    ::cuda::atomic_ref<typename _Ref::size_type, _Ref::thread_scope>{*__num_successes}.fetch_add(
      __block_num_successes, ::cuda::std::memory_order_relaxed);
  }
}

//! @brief Inserts all elements in the range `[first, first + n)` if `pred` of the corresponding
//! stencil returns true.
template <int _CgSize, int _BlockSize, class _InputIt, class _StencilIt, class _Predicate, class _Ref>
_CCCL_KERNEL_ATTRIBUTES _CCCL_LAUNCH_BOUNDS(_BlockSize) void
__insert_if_n(_InputIt __first, detail::__index_type __n, _StencilIt __stencil, _Predicate __pred, _Ref __ref)
{
  const auto __loop_stride = detail::__grid_stride() / _CgSize;
  auto __idx               = detail::__global_thread_id() / _CgSize;

  while (__idx < __n)
  {
    if (__pred(*(__stencil + __idx)))
    {
      using __value_t = typename ::cuda::std::iterator_traits<_InputIt>::value_type;
      const __value_t __insert_element{*(__first + __idx)};
      const auto __tile = ::cooperative_groups::tiled_partition<_CgSize, ::cooperative_groups::thread_block>(
        ::cooperative_groups::this_thread_block());
      __ref.insert(__tile, __insert_element);
    }
    __idx += __loop_stride;
  }
}

//! @brief Contains test with predicate.
template <int _CgSize, int _BlockSize, class _InputIt, class _StencilIt, class _Predicate, class _OutputIt, class _Ref>
_CCCL_KERNEL_ATTRIBUTES _CCCL_LAUNCH_BOUNDS(_BlockSize) void __contains_if_n(
  _InputIt __first,
  detail::__index_type __n,
  _StencilIt __stencil,
  _Predicate __pred,
  _OutputIt __output_begin,
  _Ref __ref)
{
  const auto __block       = ::cooperative_groups::this_thread_block();
  const auto __loop_stride = detail::__grid_stride() / _CgSize;
  auto __idx               = detail::__global_thread_id() / _CgSize;

  while (__idx < __n)
  {
    const auto __tile     = ::cooperative_groups::tiled_partition<_CgSize, ::cooperative_groups::thread_block>(__block);
    using __value_t       = typename ::cuda::std::iterator_traits<_InputIt>::value_type;
    const __value_t __key = *(__first + __idx);
    const auto __found    = __pred(*(__stencil + __idx)) ? __ref.contains(__tile, __key) : false;
    if (__tile.thread_rank() == 0)
    {
      *(__output_begin + __idx) = __found;
    }
    __idx += __loop_stride;
  }
}

//! @brief Helper to determine the buffer type for the find kernel.
template <class _Container, class = void>
struct __find_buffer
{
  using type = typename _Container::key_type;
};

//! @brief Helper to determine the buffer type for the find kernel when `mapped_type` exists.
template <class _Container>
struct __find_buffer<_Container, ::cuda::std::void_t<typename _Container::mapped_type>>
{
  using type = typename _Container::mapped_type;
};

//! @brief Converts a find result to the output value or the appropriate empty sentinel.
template <class _Ref, class _Iterator>
[[nodiscard]] _CCCL_DEVICE_API typename __find_buffer<_Ref>::type __find_output(_Ref const& __ref, _Iterator __found)
{
  constexpr bool __has_payload = !::cuda::std::is_same_v<typename _Ref::key_type, typename _Ref::value_type>;

  if constexpr (__has_payload)
  {
    return __found == __ref.end() ? __ref.empty_value_sentinel() : __found->second;
  }
  else
  {
    return __found == __ref.end() ? __ref.empty_key_sentinel() : *__found;
  }
}

//! @brief Find with predicate.
template <int _CgSize, int _BlockSize, class _InputIt, class _StencilIt, class _Predicate, class _OutputIt, class _Ref>
_CCCL_KERNEL_ATTRIBUTES _CCCL_LAUNCH_BOUNDS(_BlockSize) void __find_if_n(
  _InputIt __first,
  detail::__index_type __n,
  _StencilIt __stencil,
  _Predicate __pred,
  _OutputIt __output_begin,
  _Ref __ref)
{
  const auto __block       = ::cooperative_groups::this_thread_block();
  const auto __thread_idx  = __block.thread_rank();
  const auto __loop_stride = detail::__grid_stride() / _CgSize;
  auto __idx               = detail::__global_thread_id() / _CgSize;

  using __output_type = typename __find_buffer<_Ref>::type;
  __shared__ __output_type __output_buffer[_BlockSize / _CgSize];

  while ((__idx - __thread_idx / _CgSize) < __n)
  {
    if constexpr (_CgSize == 1)
    {
      if (__idx < __n)
      {
        using __value_t       = typename ::cuda::std::iterator_traits<_InputIt>::value_type;
        const __value_t __key = *(__first + __idx);
        const auto __selected = __pred(*(__stencil + __idx));
        const auto __found    = __selected ? __ref.find(__key) : __ref.end();
        /*
         * The ld.relaxed.gpu instruction causes L1 to flush more frequently, causing increased
         * sector stores from L2 to global memory. By writing results to shared memory and then
         * synchronizing before writing back to global, we no longer rely on L1, preventing the
         * increase in sector stores from L2 to global and improving performance.
         */
        __output_buffer[__thread_idx] = __find_output(__ref, __found);
      }
      __block.sync();
      if (__idx < __n)
      {
        *(__output_begin + __idx) = __output_buffer[__thread_idx];
      }
    }
    else
    {
      const auto __tile = ::cooperative_groups::tiled_partition<_CgSize, ::cooperative_groups::thread_block>(__block);
      if (__idx < __n)
      {
        using __value_t       = typename ::cuda::std::iterator_traits<_InputIt>::value_type;
        const __value_t __key = *(__first + __idx);

        bool __selected = false;
        if (__tile.thread_rank() == 0)
        {
          __selected = __pred(*(__stencil + __idx));
        }
        __selected         = __tile.shfl(__selected, 0);
        const auto __found = __selected ? __ref.find(__tile, __key) : __ref.end();

        if (__tile.thread_rank() == 0)
        {
          *(__output_begin + __idx) = __find_output(__ref, __found);
        }
      }
    }
    __idx += __loop_stride;
  }
}

//! @brief Reinserts all filled slots from old storage into a container.
//!
//! Each thread examines one physical slot, so a block stages at most `_BlockSize` elements
//! regardless of the storage's bucket size.
//!
//! @tparam _BlockSize Number of threads in a block
//! @tparam _StorageRef Old slot storage reference type
//! @tparam _ContainerRef Destination container reference type
//! @tparam _Predicate Predicate identifying filled slots
//!
//! @param[in] __old_storage Old slot storage
//! @param[in] __container_ref Destination container reference
//! @param[in] __is_filled Predicate identifying filled slots
template <int _BlockSize, class _StorageRef, class _ContainerRef, class _Predicate>
_CCCL_KERNEL_ATTRIBUTES _CCCL_LAUNCH_BOUNDS(_BlockSize) void
__rehash(_StorageRef __old_storage, _ContainerRef __container_ref, _Predicate __is_filled)
{
  using __value_type = typename _ContainerRef::value_type;

  // `__value_type` is not trivially default constructible, so a plain `__shared__` array would
  // require initialization, which is not allowed for shared variables.
  __shared__ ::cuda::__uninitialized_array<__value_type, _BlockSize> __buffer;
  __shared__ ::cuda::std::uint32_t __buffer_size;

  constexpr auto __cg_size         = _ContainerRef::cg_size;
  constexpr auto __tiles_per_block = _BlockSize / __cg_size;

  const auto __block = ::cooperative_groups::this_thread_block();
  const auto __tile  = ::cooperative_groups::tiled_partition<__cg_size, ::cooperative_groups::thread_block>(__block);
  const auto __thread_rank = __block.thread_rank();
  const auto __tile_rank   = __tile.meta_group_rank();
  const auto __loop_stride = detail::__grid_stride();
  const auto __num_slots   = __old_storage.capacity();
  auto __idx               = detail::__global_thread_id();

  while (__idx - __thread_rank < __num_slots)
  {
    if (__thread_rank == 0)
    {
      __buffer_size = 0;
    }
    __block.sync();

    if (__idx < __num_slots)
    {
      const auto __slot = *(__old_storage.data() + __idx);
      if (__is_filled(__slot))
      {
        const auto __buffer_idx =
          ::cuda::atomic_ref<::cuda::std::uint32_t, ::cuda::thread_scope_block>{__buffer_size}.fetch_add(
            1, ::cuda::std::memory_order_relaxed);
        __buffer[__buffer_idx] = __slot;
      }
    }
    __block.sync();

    const auto __local_buffer_size = __buffer_size;
    for (auto __buffer_idx = __tile_rank; __buffer_idx < __local_buffer_size; __buffer_idx += __tiles_per_block)
    {
      __container_ref.insert(__tile, __buffer[__buffer_idx]);
    }
    __block.sync();

    __idx += __loop_stride;
  }
}
} // namespace cuda::experimental::cuco::__open_addressing

_CCCL_DIAG_POP

#endif // _CCCL_CUDA_COMPILATION()

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_DETAIL_OPEN_ADDRESSING_KERNELS_CUH
