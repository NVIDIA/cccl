//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_CUDA_UNIQUE_H
#define _CUDA_STD___PSTL_CUDA_UNIQUE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_BACKEND_CUDA()

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wshadow")
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-local-typedef")
_CCCL_DIAG_SUPPRESS_NVHPC(attribute_requires_external_linkage)

#  include <cub/device/device_select.cuh>

_CCCL_DIAG_POP

#  include <cuda/__execution/policy.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__algorithm/unique.h>
#  include <cuda/std/__algorithm/unique_copy.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/incrementable_traits.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__iterator/next.h>
#  include <cuda/std/__memory/pointer_traits.h>
#  include <cuda/std/__pstl/cuda/temporary_storage.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__utility/move.h>
#  include <cuda/std/cstdint>
#  include <cuda/std/tuple>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <>
struct __pstl_dispatch<__pstl_algorithm::__unique, __execution_backend::__cuda>
{
  template <CUB_NS_QUALIFIER::SelectImpl Select,
            class _Policy,
            class _InputIterator,
            class _OutputIterator,
            class _BinaryPredicate>
  [[nodiscard]] _CCCL_HOST_API static _InputIterator __par_impl(
    const _Policy& __policy,
    _InputIterator __first,
    _InputIterator __last,
    _OutputIterator __result,
    _BinaryPredicate __pred)
  {
    using _OffsetType    = iter_difference_t<_InputIterator>;
    using DispatchUnique = CUB_NS_QUALIFIER::DispatchSelectIf<
      _InputIterator,
      CUB_NS_QUALIFIER::NullType*,
      _InputIterator,
      _OffsetType*,
      CUB_NS_QUALIFIER::NullType,
      _BinaryPredicate,
      _OffsetType,
      Select>;

    const auto __count         = ::cuda::std::distance(__first, __last);
    _OffsetType __num_selected = 0;
    size_t __num_bytes         = 0;

    _CCCL_TRY_CUDA_API(
      DispatchUnique::Dispatch,
      "__pstl_cuda_unique: determination of device storage for cub::DispatchSelectIf::Dispatch failed",
      static_cast<void*>(nullptr),
      __num_bytes,
      __first,
      static_cast<CUB_NS_QUALIFIER::NullType*>(nullptr),
      __result,
      static_cast<_OffsetType*>(nullptr),
      CUB_NS_QUALIFIER::NullType{},
      __pred,
      __count,
      0);

    auto __stream = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStreamPerThread}, __policy);

    { // Create temporary storage for the return value as well as a copy of the input sequence as Unique is not inplace
      __temporary_storage<_OffsetType> __storage{__policy, __num_bytes, 1};

      _CCCL_TRY_CUDA_API(
        DispatchUnique::Dispatch,
        "__pstl_cuda_unique: kernel launch of cub::DispatchSelectIf::Dispatch failed",
        __storage.__get_temp_storage(),
        __num_bytes,
        ::cuda::std::move(__first),
        static_cast<CUB_NS_QUALIFIER::NullType*>(nullptr),
        __result,
        __storage.template __get_ptr<0>(),
        CUB_NS_QUALIFIER::NullType{},
        ::cuda::std::move(__pred),
        __count,
        __stream.get());

      _CCCL_TRY_CUDA_API(
        ::cudaMemcpyAsync,
        "__pstl_cuda_unique: copy of num_selected from device to host failed",
        ::cuda::std::addressof(__num_selected),
        __storage.template __get_ptr<0>(),
        sizeof(_OffsetType),
        cudaMemcpyDefault,
        __stream.get());
    }

    __stream.sync();
    return __result + static_cast<iter_difference_t<_OutputIterator>>(__num_selected);
  }

  _CCCL_TEMPLATE(class _Policy, class _InputIterator, class _BinaryPredicate)
  _CCCL_REQUIRES(__has_forward_traversal<_InputIterator>)
  [[nodiscard]] _CCCL_HOST_API _InputIterator operator()(
    [[maybe_unused]] const _Policy& __policy,
    _InputIterator __first,
    _InputIterator __last,
    _BinaryPredicate __pred) const
  {
    if constexpr (::cuda::std::__has_random_access_traversal<_InputIterator>)
    {
      try
      {
        return __par_impl<CUB_NS_QUALIFIER::SelectImpl::SelectPotentiallyInPlace>(
          __policy, __first, ::cuda::std::move(__last), __first, ::cuda::std::move(__pred));
      }
      catch (const ::cuda::cuda_error& __err)
      {
        if (__err.status() == ::cudaErrorMemoryAllocation)
        {
          _CCCL_THROW(::std::bad_alloc);
        }
        else
        {
          throw __err;
        }
      }
    }
    else
    {
      static_assert(__always_false_v<_Policy>,
                    "__pstl_dispatch: CUDA backend of cuda::std::unique requires at least random access iterators");
      return ::cuda::std::unique(::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__pred));
    }
  }

  _CCCL_TEMPLATE(class _Policy, class _InputIterator, class _OutputIterator, class _BinaryPredicate)
  _CCCL_REQUIRES(__has_forward_traversal<_OutputIterator>)
  [[nodiscard]] _CCCL_HOST_API _OutputIterator operator()(
    [[maybe_unused]] const _Policy& __policy,
    _InputIterator __first,
    _InputIterator __last,
    _OutputIterator __result,
    _BinaryPredicate __pred) const
  {
    if constexpr (::cuda::std::__has_random_access_traversal<_InputIterator>
                  && ::cuda::std::__has_random_access_traversal<_OutputIterator>)
    {
      try
      {
        return __par_impl<CUB_NS_QUALIFIER::SelectImpl::Select>(
          __policy,
          ::cuda::std::move(__first),
          ::cuda::std::move(__last),
          ::cuda::std::move(__result),
          ::cuda::std::move(__pred));
      }
      catch (const ::cuda::cuda_error& __err)
      {
        if (__err.status() == ::cudaErrorMemoryAllocation)
        {
          _CCCL_THROW(::std::bad_alloc);
        }
        else
        {
          throw __err;
        }
      }
    }
    else
    {
      static_assert(__always_false_v<_Policy>,
                    "__pstl_dispatch: CUDA backend of cuda::std::unique_copy requires at least random access "
                    "iterators");
      return ::cuda::std::unique_copy(
        ::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__result), ::cuda::std::move(__pred));
    }
  }
};

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA_STD___PSTL_CUDA_UNIQUE_H
