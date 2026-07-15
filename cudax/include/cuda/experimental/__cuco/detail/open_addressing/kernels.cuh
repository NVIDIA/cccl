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
#include <cuda/__functional/proclaim_return_type.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/void_t.h>

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
__insert_if_fn(_InputIt, _StencilIt, _Predicate, _Ref) -> __insert_if_fn<_InputIt, _StencilIt, _Predicate, _Ref>;

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
__contains_if_fn(_InputIt, _StencilIt, _Predicate, _OutputIt, _Ref)
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

  constexpr bool __has_payload = !::cuda::std::is_same_v<typename _Ref::key_type, typename _Ref::value_type>;

  const auto __sentinel = [&]() {
    if constexpr (__has_payload)
    {
      return __ref.empty_value_sentinel();
    }
    else
    {
      return __ref.empty_key_sentinel();
    }
  }();

  const auto __output = ::cuda::proclaim_return_type<__output_type>([&] _CCCL_DEVICE(auto __found) {
    if constexpr (__has_payload)
    {
      return __found == __ref.end() ? __sentinel : __found->second;
    }
    else
    {
      return __found == __ref.end() ? __sentinel : *__found;
    }
  });

  while ((__idx - __thread_idx / _CgSize) < __n)
  {
    if constexpr (_CgSize == 1)
    {
      if (__idx < __n)
      {
        using __value_t               = typename ::cuda::std::iterator_traits<_InputIt>::value_type;
        const __value_t __key         = *(__first + __idx);
        const auto __found            = __ref.find(__key);
        __output_buffer[__thread_idx] = __pred(*(__stencil + __idx)) ? __output(__found) : __sentinel;
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
        const auto __found    = __ref.find(__tile, __key);

        if (__tile.thread_rank() == 0)
        {
          *(__output_begin + __idx) = __pred(*(__stencil + __idx)) ? __output(__found) : __sentinel;
        }
      }
    }
    __idx += __loop_stride;
  }
}
} // namespace cuda::experimental::cuco::__open_addressing

_CCCL_DIAG_POP

#endif // _CCCL_CUDA_COMPILATION()

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_DETAIL_OPEN_ADDRESSING_KERNELS_CUH
