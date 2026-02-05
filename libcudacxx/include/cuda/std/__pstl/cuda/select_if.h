//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_CUDA_SELECT_IF_H
#define _CUDA_STD___PSTL_CUDA_SELECT_IF_H

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
#  include <cuda/__functional/call_or.h>
#  include <cuda/__memory_pool/device_memory_pool.h>
#  include <cuda/__memory_resource/get_memory_resource.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__algorithm/copy_if.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/incrementable_traits.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__new/bad_alloc.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__utility/move.h>

#  include <cuda_runtime.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <>
struct __pstl_dispatch<__pstl_algorithm::__select_if, __execution_backend::__cuda>
{
  //! Ensures we properly deallocate the memory allocated for the result
  template <class _OffsetType, class _Resource>
  struct __allocation_guard
  {
    ::cuda::stream_ref __stream_;
    _Resource& __resource_;
    _OffsetType* __ptr_;

    _CCCL_HOST_API __allocation_guard(::cuda::stream_ref __stream, _Resource& __resource, size_t __num_bytes)
        : __stream_(__stream)
        , __resource_(__resource)
        , __ptr_(static_cast<_OffsetType*>(
            __resource_.allocate(__stream_, sizeof(_OffsetType) + __num_bytes, alignof(_OffsetType))))
    {}

    _CCCL_HOST_API ~__allocation_guard()
    {
      __resource_.deallocate(__stream_, __ptr_, sizeof(_OffsetType), alignof(_OffsetType));
    }

    [[nodiscard]] _CCCL_HOST_API _OffsetType* __get_result_iter()
    {
      return __ptr_;
    }

    [[nodiscard]] _CCCL_HOST_API void* __get_temp_storage()
    {
      return static_cast<void*>(__ptr_ + 1);
    }
  };

  template <CUB_NS_QUALIFIER::SelectImpl _Select,
            class _Policy,
            class _InputIterator,
            class _OutputIterator,
            class _UnaryPredicate>
  [[nodiscard]] _CCCL_HOST_API static _OutputIterator __par_impl(
    const _Policy& __policy,
    _InputIterator __first,
    iter_difference_t<_InputIterator> __count,
    _OutputIterator __result,
    _UnaryPredicate __pred)
  {
    // DeviceSelect needs some additional data passed
    using _StencilIter = CUB_NS_QUALIFIER::NullType*;
    using _EqualityOp  = CUB_NS_QUALIFIER::NullType;
    using _OffsetType  = iter_difference_t<_InputIterator>;
    using _SelectIf    = CUB_NS_QUALIFIER::DispatchSelectIf<
         _InputIterator,
         _StencilIter,
         _OutputIterator,
         _OffsetType*,
         _UnaryPredicate,
         _EqualityOp,
         _OffsetType,
         _Select>;

    _OffsetType __ret;

    auto __stream   = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStreamPerThread}, __policy);
    auto __resource = ::cuda::__call_or(
      ::cuda::mr::get_memory_resource, ::cuda::device_default_memory_pool(__stream.device()), __policy);

    // Determine temporary device storage requirements
    void* __temp_storage = nullptr;
    size_t __num_bytes   = 0;
    _CCCL_TRY_CUDA_API(
      _SelectIf::Dispatch,
      "__pstl_cuda_select_if: determination of device storage for cub::DeviceSelect::Select failed",
      __temp_storage,
      __num_bytes,
      __first,
      static_cast<_StencilIter>(nullptr),
      __result,
      static_cast<_OffsetType*>(nullptr),
      __pred,
      _EqualityOp{},
      __count,
      __stream.get());

    {
      __allocation_guard<_OffsetType, decltype(__resource)> __guard{__stream, __resource, __num_bytes};

      // Run the kernel
      _CCCL_TRY_CUDA_API(
        _SelectIf::Dispatch,
        "__pstl_cuda_select_if: kernel launch of cub::DeviceSelect::Select failed",
        __guard.__get_temp_storage(),
        __num_bytes,
        ::cuda::std::move(__first),
        static_cast<_StencilIter>(nullptr),
        __result,
        __guard.__get_result_iter(),
        ::cuda::std::move(__pred),
        _EqualityOp{},
        __count,
        __stream.get());

      // Copy the result back from storage
      _CCCL_TRY_CUDA_API(
        ::cudaMemcpyAsync,
        "__pstl_cuda_select_if: copy of result from device to host failed",
        ::cuda::std::addressof(__ret),
        __guard.__ptr_,
        sizeof(_OffsetType),
        ::cudaMemcpyDefault,
        __stream.get());
    }

    __stream.sync();
    return __result + __ret;
  }

  _CCCL_TEMPLATE(class _Policy, class _InputIterator, class _OutputIterator, class _UnaryPredicate)
  _CCCL_REQUIRES(__has_forward_traversal<_OutputIterator>)
  [[nodiscard]] _CCCL_HOST_API _OutputIterator operator()(
    [[maybe_unused]] const _Policy& __policy,
    _InputIterator __first,
    iter_difference_t<_InputIterator> __count,
    _OutputIterator __result,
    _UnaryPredicate __pred) const
  {
    if constexpr (::cuda::std::__has_random_access_traversal<_OutputIterator>)
    {
      try
      {
        if constexpr (is_same_v<_InputIterator, _OutputIterator>)
        {
          if (__first == __result)
          { // This is the remove path
            return __par_impl<::cub::SelectImpl::SelectPotentiallyInPlace>(
              __policy, ::cuda::std::move(__first), __count, ::cuda::std::move(__result), ::cuda::std::move(__pred));
          }
        }

        // Default path with disjunct input / output ranges
        return __par_impl<::cub::SelectImpl::Select>(
          __policy, ::cuda::std::move(__first), __count, ::cuda::std::move(__result), ::cuda::std::move(__pred));
      }
      catch (const ::cuda::cuda_error& __err)
      {
        if (__err.status() == cudaErrorMemoryAllocation)
        {
          ::cuda::std::__throw_bad_alloc();
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
                    "__pstl_cuda_generate: CUDA backend of cuda::std::generate requires at least random access "
                    "iterators");
      auto __last = ::cuda::std::next(__first, __count);
      return ::cuda::std::copy_if(
        ::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__result), ::cuda::std::move(__pred));
    }
  }
};

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif /// _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA_STD___PSTL_CUDA_SELECT_IF_H
