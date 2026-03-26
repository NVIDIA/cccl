//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_CUDA_TEMPORARY_STORAGE_H
#define _CUDA_STD___PSTL_CUDA_TEMPORARY_STORAGE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_BACKEND_CUDA()

#  include <cuda/__cmath/round_up.h>
#  include <cuda/__functional/call_or.h>
#  include <cuda/__iterator/tabulate_output_iterator.h>
#  include <cuda/__memory/align_up.h>
#  include <cuda/__memory_pool/device_memory_pool.h>
#  include <cuda/__memory_resource/any_resource.h>
#  include <cuda/__memory_resource/get_memory_resource.h>
#  include <cuda/__memory_resource/properties.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__memory/construct_at.h>
#  include <cuda/std/__type_traits/is_callable.h>
#  include <cuda/std/__type_traits/type_list.h>
#  include <cuda/std/__utility/forward.h>
#  include <cuda/std/__utility/integer_sequence.h>
#  include <cuda/std/cstdint>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

template <class _ResultType>
struct __temporary_storage_construct_result
{
  _ResultType* __result_;

  _CCCL_HOST_API __temporary_storage_construct_result(_ResultType* __result = nullptr) noexcept
      : __result_(__result)
  {}

  template <class _Index, class _Up>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  operator()(_Index, _Up&& __value) noexcept(is_nothrow_constructible_v<_ResultType, _Up>)
  {
    ::cuda::std::__construct_at(__result_, ::cuda::std::forward<_Up>(__value));
  }
};

//! @brief Provides device accessible storage for a number of typed sequences and temporary storage the algorithm might
//! need.
template <class... _StoredTypes>
class __temporary_storage
{
  ::cuda::stream_ref __stream_;
  ::cuda::mr::resource_ref<::cuda::mr::device_accessible> __resource_;
  size_t __total_bytes_allocated_;
  array<void*, 1 + sizeof...(_StoredTypes)> __storage_;

  _CCCL_TEMPLATE(class... _Sizes)
  _CCCL_REQUIRES((sizeof...(_Sizes) == sizeof...(_StoredTypes)))
  [[nodiscard]] _CCCL_HOST_API static constexpr size_t
  __get_total_bytes_allocated(const size_t __num_bytes_storage, const _Sizes... __elements_stored) noexcept
  {
    return (::cuda::round_up(static_cast<size_t>(__elements_stored) * sizeof(_StoredTypes),
                             ::cuda::mr::default_cuda_malloc_alignment)
            + ... + ::cuda::round_up(__num_bytes_storage, ::cuda::mr::default_cuda_malloc_alignment));
  }

  template <size_t _Index>
  [[nodiscard]] _CCCL_HOST_API static constexpr array<void*, 1 + sizeof...(_StoredTypes)>
  __get_storage(array<void*, 1 + sizeof...(_StoredTypes)>& __storage,
                const array<size_t, sizeof...(_StoredTypes)>& __num_elements) noexcept
  {
    if constexpr (_Index == sizeof...(_StoredTypes))
    {
      return __storage;
    }
    else
    {
      using _StoredType     = __type_at_c<_Index, __type_list<_StoredTypes...>>;
      __storage[_Index + 1] = static_cast<void*>(
        ::cuda::align_up(static_cast<_StoredType*>(__storage[_Index]) + __num_elements[_Index],
                         ::cuda::mr::default_cuda_malloc_alignment));
      return __get_storage<_Index + 1>(__storage, __num_elements);
    }
  }

  _CCCL_TEMPLATE(class... _Sizes)
  _CCCL_REQUIRES((sizeof...(_Sizes) == sizeof...(_StoredTypes)))
  [[nodiscard]] _CCCL_HOST_API static constexpr array<void*, 1 + sizeof...(_StoredTypes)>
  __get_storage(void* __ptr, const _Sizes... __elements_stored) noexcept
  {
    array<void*, 1 + sizeof...(_StoredTypes)> __storage{__ptr};
    array<size_t, sizeof...(_StoredTypes)> __num_elements{static_cast<size_t>(__elements_stored)...};
    return __get_storage<0>(__storage, __num_elements);
  }

  //! @brief Helper function to retrieve a memory resource from a policy
  //!        In contrast to `__call_or` it does not require us to always call .device() on the stream
  template <class _Policy>
  [[nodiscard]] _CCCL_HOST_API static ::cuda::mr::resource_ref<::cuda::mr::device_accessible>
  __get_memory_resource_or(const _Policy& __policy) noexcept
  {
    if constexpr (::cuda::std::__is_callable_v<::cuda::mr::get_memory_resource_t, const _Policy&>)
    {
      return ::cuda::mr::get_memory_resource(__policy);
    }
    else if constexpr (::cuda::std::__is_callable_v<::cuda::get_stream_t, const _Policy&>)
    {
      return ::cuda::device_default_memory_pool(::cuda::get_stream(__policy).device());
    }
    else
    { // no stream no memory resource, assume device 0
      return ::cuda::device_default_memory_pool(0);
    }
  }

public:
  _CCCL_TEMPLATE(class _Policy, class... _Sizes)
  _CCCL_REQUIRES((sizeof...(_Sizes) == sizeof...(_StoredTypes)))
  _CCCL_HOST_API
  __temporary_storage(const _Policy& __policy, const size_t __num_bytes_storage, const _Sizes... __elements_stored)
      : __stream_(::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStreamPerThread}, __policy))
      , __resource_(__get_memory_resource_or(__policy))
      , __total_bytes_allocated_(__get_total_bytes_allocated(__num_bytes_storage, __elements_stored...))
      , __storage_(__get_storage(
          __resource_.allocate(__stream_, __total_bytes_allocated_, ::cuda::mr::default_cuda_malloc_alignment),
          __elements_stored...))
  {}

  _CCCL_HOST_API ~__temporary_storage()
  {
    __resource_.deallocate(
      __stream_, __storage_[0], __total_bytes_allocated_, ::cuda::mr::default_cuda_malloc_alignment);
  }

  //! We are dealing with uninitialized storage, so we might need to go through construct_at
  template <size_t _Index, class _OtherType = __type_at_c<_Index, __type_list<_StoredTypes...>>>
  [[nodiscard]] _CCCL_HOST_API auto __get_ptr() noexcept
  {
    static_assert(_Index < sizeof...(_StoredTypes), "__temporary_storage::__get_ptr: Invalid index");
    using _StoredType = __type_at_c<_Index, __type_list<_StoredTypes...>>;
    if constexpr (::cuda::std::__detail::__can_optimize_construct_at<_StoredType, _OtherType>)
    {
      return static_cast<_StoredType*>(__storage_[_Index]);
    }
    else
    {
      return ::cuda::tabulate_output_iterator{
        __temporary_storage_construct_result<_StoredType>{static_cast<_StoredType*>(__storage_[_Index])}};
    }
  }

  //! When we know we can just return a plain pointer
  template <size_t _Index>
  [[nodiscard]] _CCCL_HOST_API auto* __get_raw_ptr() noexcept
  {
    static_assert(_Index < sizeof...(_StoredTypes), "__temporary_storage::__get_ptr: Invalid index");
    using _StoredType = __type_at_c<_Index, __type_list<_StoredTypes...>>;
    return static_cast<_StoredType*>(__storage_[_Index]);
  }

  // The final pointer is always the temporary storage for the algorithm
  [[nodiscard]] _CCCL_HOST_API void* __get_temp_storage() noexcept
  {
    return __storage_[sizeof...(_StoredTypes)];
  }
};

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif /// _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA_STD___PSTL_CUDA_TEMPORARY_STORAGE_H
