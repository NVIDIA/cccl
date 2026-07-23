// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_BUFFER_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_BUFFER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/vector_base.h>

#include <cuda/__container/buffer.h>
#include <cuda/__device/device_ref.h>
#include <cuda/__functional/lazy_call_or.h>
#include <cuda/__memory_pool/device_memory_pool.h>
#include <cuda/__memory_resource/get_memory_resource.h>
#include <cuda/__utility/no_init.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__type_traits/is_trivially_default_constructible.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/move.h>

#include <cuda/experimental/__multi_gpu/algorithm/common.h>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental::__detail
{
// Maps an _Env to the memory resource type produced by __resource_from_env.
template <class _Env>
using __pool_type_for = decltype(::cuda::device_default_memory_pool(::cuda::std::declval<::cuda::device_ref>()));

template <class _Env>
using __resource_type_for = ::cuda::std::remove_cvref_t<
  ::cuda::__lazy_call_result_or_t<::cuda::mr::get_memory_resource_t, __pool_type_for<_Env>(void), _Env>>;

// cuda::buffer has no concept of size vs capacity. We need both below because
// we want to be able to shrink a buffer without reallocating a new one.
template <class _Up, class _Resource>
struct __buffer
{
  template <class _Up2>
  using __buffer_type_for = typename ::cuda::__buffer_type_for_props<_Up2, typename _Resource::default_queries>;

  using __buff_type = __buffer_type_for<_Up>;

  // The move operations reset the moved-from __actual_size_ to 0 so that a
  // moved-from buffer reports size() == 0, matching the emptied underlying
  // allocation.
  _CCCL_HOST_API __buffer(__buffer&& __other) noexcept
      : __buf_{::cuda::std::move(__other.__buf_)}
      , __actual_size_{::cuda::std::exchange(__other.__actual_size_, 0)}
  {}

  _CCCL_HOST_API __buffer& operator=(__buffer&& __other) noexcept
  {
    __buf_         = ::cuda::std::move(__other.__buf_);
    __actual_size_ = ::cuda::std::exchange(__other.__actual_size_, 0);
    return *this;
  }

  __buffer(const __buffer& __other) = default;

  __buffer& operator=(const __buffer& __other)
  {
    // cuda::buffer has no copy assignment operator
    __buf_         = __buff_type{__other.__buf_};
    __actual_size_ = __other.__actual_size_;
    return *this;
  }

  // Recreate the most common constructors for buffer that we use below to
  // avoid needing to call make_buffer() over and over again.
  template <class _Resource2, class _EnvT>
  _CCCL_HOST_API explicit __buffer(::cuda::stream_ref __stream, _Resource2&& __resource, const _EnvT& __env)
      : __buffer{__buff_type{__stream,
                             ::cuda::std::forward<_Resource2>(__resource),
                             ::cuda::experimental::__detail::__sanitize_buffer_env(__env)}}
  {}

  template <class _Resource2, class _EnvT>
  _CCCL_HOST_API explicit __buffer(
    ::cuda::stream_ref __stream,
    _Resource2&& __resource,
    ::cuda::std::size_t __size,
    ::cuda::no_init_t,
    const _EnvT& __env)
      : __buffer{__buff_type{
          __stream,
          ::cuda::std::forward<_Resource2>(__resource),
          __size,
          ::cuda::no_init,
          ::cuda::experimental::__detail::__sanitize_buffer_env(__env)}}
  {}

  template <class _Resource2, class _EnvT>
  _CCCL_HOST_API explicit __buffer(
    ::cuda::stream_ref __stream,
    _Resource2&& __resource,
    ::cuda::std::size_t __size,
    const _Up& __value,
    const _EnvT& __env)
      : __buffer{::cuda::make_buffer<_Up>(
          __stream,
          ::cuda::std::forward<_Resource2>(__resource),
          __size,
          __value,
          ::cuda::experimental::__detail::__sanitize_buffer_env(__env))}
  {}

  _CCCL_HOST_API explicit __buffer(__buff_type __buf)
      : __buf_{::cuda::std::move(__buf)}
      , __actual_size_{__buf_.size()}
  {}

  _CCCL_HOST_API void resize(::cuda::std::size_t __new_size)
  {
    // TODO(jfaibussowit):
    //
    // This is WRONG
    resize(__new_size, ::cuda::no_init);
  }

  // Grow or shrink the buffer to __new_size. This is effectively
  // std::vector::resize() except that it never touches original values. On
  // growth it will allocate a new buffer with uninitialized values, while on
  // shrinkage it will leave the original values as-is.
  _CCCL_HOST_API void resize(::cuda::std::size_t __new_size, ::cuda::no_init_t)
  {
    if (__new_size > capacity())
    {
      // Don't use __make_empty_like() here. Even if the current buffer
      // doesn't say it can hold the new size, it's possible that the
      // underlying allocation actually *is* big enough. The only thing that
      // knows this is the memory resource, so we first return the existing
      // buffer to the memory resource before creating the new one.
      const auto __stream = __get().stream();
      auto __mr           = __get().memory_resource();
      const auto __align  = __get().alignment();

      __get().destroy();

      __get() = __buff_type{
        __stream,
        ::cuda::std::move(__mr),
        __new_size,
        ::cuda::no_init,
        ::cuda::std::execution::prop{::cuda::allocation_alignment, __align}};
    }
    __actual_size_ = __new_size;
  }

  template <class _Up2 = _Up>
  [[nodiscard]] _CCCL_HOST_API __buffer_type_for<_Up2> __make_empty_like(::cuda::std::size_t __new_size) const
  {
    // TODO(jfaibussowit):
    //
    // buffer ideally should have a make_buffer_like(source_buffer) helper
    // that does this for us, similar to e.g. numpy.empty_like()
    return __buffer_type_for<_Up2>{
      __get().stream(),
      __get().memory_resource(),
      __new_size,
      ::cuda::no_init,
      ::cuda::std::execution::prop{::cuda::allocation_alignment, __get().alignment()}};
  }

  template <class _Up2 = _Up>
  [[nodiscard]] _CCCL_HOST_API __buffer_type_for<_Up2> __make_empty_like() const
  {
    return __make_empty_like<_Up2>(size());
  }

  [[nodiscard]] _CCCL_HOST_API __buff_type& __get() noexcept
  {
    return __buf_;
  }

  [[nodiscard]] _CCCL_HOST_API const __buff_type& __get() const noexcept
  {
    return __buf_;
  }

  [[nodiscard]] _CCCL_HOST_API auto* data() noexcept
  {
    return __buf_.data();
  }

  [[nodiscard]] _CCCL_HOST_API const auto* data() const noexcept
  {
    return __buf_.data();
  }

  [[nodiscard]] _CCCL_HOST_API auto begin() noexcept
  {
    return __buf_.begin();
  }

  [[nodiscard]] _CCCL_HOST_API auto end() noexcept
  {
    return begin() + size();
  }

  [[nodiscard]] _CCCL_HOST_API auto begin() const noexcept
  {
    return __buf_.begin();
  }

  [[nodiscard]] _CCCL_HOST_API auto end() const noexcept
  {
    return begin() + size();
  }

  [[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t size() const noexcept
  {
    return __actual_size_;
  }

  [[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t capacity() const noexcept
  {
    return __get().size();
  }

private:
  __buff_type __buf_;
  ::cuda::std::size_t __actual_size_{};
};

template <class _Range>
_CCCL_CONCEPT __can_thrust_no_init_resize = _CCCL_REQUIRES_EXPR((_Range), _Range& __range)(
  requires(::cuda::std::is_trivially_default_constructible_v<::cuda::std::ranges::range_value_t<_Range>>),
  __range.resize(::cuda::std::size_t{}, ::thrust::no_init));

template <class _Range>
_CCCL_CONCEPT __can_cuda_no_init_resize = _CCCL_REQUIRES_EXPR((_Range), _Range& __range)(
  requires(::cuda::std::is_trivially_default_constructible_v<::cuda::std::ranges::range_value_t<_Range>>),
  __range.resize(::cuda::std::size_t{}, ::cuda::no_init));

template <class _Range>
_CCCL_HOST_API void __resize_for_overwrite(_Range& __range, ::cuda::std::size_t __size)
{
  if constexpr (__can_thrust_no_init_resize<_Range>)
  {
    __range.resize(__size, ::thrust::no_init);
  }
  else if constexpr (__can_cuda_no_init_resize<_Range>)
  {
    __range.resize(__size, ::cuda::no_init);
  }
  else
  {
    __range.resize(__size);
  }
}
} // namespace cuda::experimental::__detail

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_BUFFER_H
