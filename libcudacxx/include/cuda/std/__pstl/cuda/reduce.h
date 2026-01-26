//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_CUDA_REDUCE_H
#define _CUDA_STD___PSTL_CUDA_REDUCE_H

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

#  include <cub/device/device_reduce.cuh>

_CCCL_DIAG_POP

#  include <cuda/__execution/policy.h>
#  include <cuda/__iterator/tabulate_output_iterator.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__functional/invoke.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__memory/addressof.h>
#  include <cuda/std/__memory/construct_at.h>
#  include <cuda/std/__new/bad_alloc.h>
#  include <cuda/std/__numeric/reduce.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__utility/move.h>

#  include <cuda_runtime.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <>
struct __pstl_dispatch<__pstl_algorithm::__reduce, __execution_backend::__cuda>
{
  //! Ensures we properly deallocate the memory allocated for the result
  template <class _Tp, class _AccumT>
  struct __allocation_guard
  {
    //! This helper struct ensures that we can properly assign types with a nontrivial assignment operator
    struct __construct_result
    {
      _Tp* __ptr_;

      _CCCL_HOST_API __construct_result(_Tp* __ptr = nullptr) noexcept
          : __ptr_(__ptr)
      {}

      template <class _Index, class _Up>
      _CCCL_DEVICE_API void operator()(_Index, _Up&& __value)
      {
        ::cuda::std::__construct_at(__ptr_, ::cuda::std::forward<_Up>(__value));
      }
    };

    void* __ptr_;

    _CCCL_HOST_API __allocation_guard(size_t __num_bytes)
        : __ptr_(nullptr)
    {
      // Add the temporary storage from DeviceReduce to the allocation
      _CCCL_TRY_CUDA_API(::cudaMalloc,
                         "__pstl_cuda_reduce: allocation failed",
                         reinterpret_cast<void**>(&__ptr_),
                         sizeof(_Tp) + __num_bytes);
    }

    _CCCL_HOST_API ~__allocation_guard()
    {
      _CCCL_ASSERT_CUDA_API(::cudaFree, "__pstl_cuda_reduce: deallocate failed", __ptr_);
    }

    [[nodiscard]] _CCCL_HOST_API auto __get_result_iter()
    {
      if constexpr (::cuda::std::__detail::__can_optimize_construct_at<_Tp, _AccumT>)
      {
        return reinterpret_cast<_Tp*>(__ptr_);
      }
      else
      {
        return ::cuda::tabulate_output_iterator{__construct_result{reinterpret_cast<_Tp*>(__ptr_)}};
      }
    }

    [[nodiscard]] _CCCL_HOST_API void* __get_temp_storage()
    {
      return static_cast<void*>(static_cast<unsigned char*>(__ptr_) + sizeof(_Tp));
    }
  };

  template <class _Policy, class _Iter, class _Tp, class _BinaryOp>
  [[nodiscard]] _CCCL_HOST_API static _Tp
  __par_impl([[maybe_unused]] const _Policy& __policy, _Iter __first, _Iter __last, _Tp __init, _BinaryOp __func)
  {
    _Tp __ret;

    {
      // We need to know the accumulator type to determine whether we need construct_at for the return value
      using _AccumT = __accumulator_t<_BinaryOp, iter_reference_t<_Iter>, _Tp>;

      //!    // Determine temporary device storage requirements for reduce
      void* __temp_storage   = nullptr;
      size_t __num_bytes     = 0;
      const auto __num_items = ::cuda::std::distance(__first, __last);
      ::cub::DeviceReduce::Reduce(
        __temp_storage, __num_bytes, __first, static_cast<_Tp*>(nullptr), __num_items, __func, __init);

      // Allocate memory for result
      __allocation_guard<_Tp, _AccumT> __guard{__num_bytes};

      // Run the reduction
      ::cub::DeviceReduce::Reduce(
        __guard.__get_temp_storage(),
        __num_bytes,
        ::cuda::std::move(__first),
        __guard.__get_result_iter(),
        __num_items,
        ::cuda::std::move(__func),
        ::cuda::std::move(__init));

      // Copy the result back from storage
      _CCCL_TRY_CUDA_API(
        ::cudaMemcpy,
        "__pstl_cuda_reduce: copy of result from device to host failed",
        ::cuda::std::addressof(__ret),
        __guard.__ptr_,
        sizeof(_Tp),
        ::cudaMemcpyDeviceToHost);
    }

    return __ret;
  }

  template <class _Policy, class _Iter, class _Tp, class _BinaryOp>
  [[nodiscard]] _CCCL_HOST_API _Tp
  operator()([[maybe_unused]] const _Policy& __policy, _Iter __first, _Iter __last, _Tp __init, _BinaryOp __func) const
  {
    if constexpr (::cuda::std::__has_random_access_traversal<_Iter>)
    {
      try
      {
        return __par_impl(
          __policy,
          ::cuda::std::move(__first),
          ::cuda::std::move(__last),
          ::cuda::std::move(__init),
          ::cuda::std::move(__func));
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
                    "__pstl_dispatch: CUDA backend of cuda::std::reduce requires at least random access iterators");
      return ::cuda::std::reduce(
        ::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__init), ::cuda::std::move(__func));
    }
  }
};

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif /// _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA_STD___PSTL_CUDA_REDUCE_H
