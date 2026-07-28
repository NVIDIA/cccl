//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_CUDA_MINMAX_ELEMENT_H
#define _CUDA_STD___PSTL_CUDA_MINMAX_ELEMENT_H

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
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")
_CCCL_DIAG_SUPPRESS_NVHPC(attribute_requires_external_linkage)

#  include <cub/device/device_reduce.cuh>

_CCCL_DIAG_POP

#  include <cuda/__execution/policy.h>
#  include <cuda/__functional/call_or.h>
#  include <cuda/__iterator/counting_iterator.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__algorithm/minmax_element.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__memory/addressof.h>
#  include <cuda/std/__pstl/cuda/ensure_current_context.h>
#  include <cuda/std/__pstl/cuda/temporary_storage.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__utility/move.h>
#  include <cuda/std/__utility/pair.h>
#  include <cuda/std/cstdint>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

//! @brief Accumulator of the reduction, carrying the current first minimum and last maximum with their offsets
template <class _Tp>
struct __minmax_element_result
{
  _Tp __min_value_;
  _Tp __max_value_;
  int64_t __min_index_;
  int64_t __max_index_;
};

//! @brief Loads the element at the given offset and enters it into the reduction as both minimum and maximum
template <class _Iter>
struct __minmax_element_transform_fn
{
  _Iter __first_;

  [[nodiscard]] _CCCL_API __minmax_element_result<iter_value_t<_Iter>> operator()(int64_t __index) const
  {
    const iter_value_t<_Iter> __value = __first_[static_cast<iter_difference_t<_Iter>>(__index)];
    return __minmax_element_result<iter_value_t<_Iter>>{__value, __value, __index, __index};
  }
};

//! @brief Combines two accumulators in a single pass: on equivalent values the minimum keeps the smaller offset and
//!        the maximum keeps the larger offset, matching the first-minimum / last-maximum requirement of
//!        cuda::std::minmax_element
template <class _Tp, class _BinaryPred>
struct __minmax_element_reduce_fn
{
  _BinaryPred __pred_;

  [[nodiscard]] _CCCL_API constexpr __minmax_element_result<_Tp>
  operator()(const __minmax_element_result<_Tp>& __lhs, const __minmax_element_result<_Tp>& __rhs) const
  {
    // The initial value of the reduction is flagged with offset -1 and loses against every element
    if (__lhs.__min_index_ < 0)
    {
      return __rhs;
    }
    if (__rhs.__min_index_ < 0)
    {
      return __lhs;
    }

    __minmax_element_result<_Tp> __result = __lhs;
    // first minimum: a strictly smaller value wins, equivalent values keep the smaller offset
    if (__pred_(__rhs.__min_value_, __lhs.__min_value_)
        || (!__pred_(__lhs.__min_value_, __rhs.__min_value_) && __rhs.__min_index_ < __lhs.__min_index_))
    {
      __result.__min_value_ = __rhs.__min_value_;
      __result.__min_index_ = __rhs.__min_index_;
    }
    // last maximum: a strictly greater value wins, equivalent values keep the larger offset
    if (__pred_(__lhs.__max_value_, __rhs.__max_value_)
        || (!__pred_(__rhs.__max_value_, __lhs.__max_value_) && __rhs.__max_index_ > __lhs.__max_index_))
    {
      __result.__max_value_ = __rhs.__max_value_;
      __result.__max_index_ = __rhs.__max_index_;
    }
    return __result;
  }
};

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <>
struct __pstl_dispatch<__pstl_algorithm::__minmax_element, __execution_backend::__cuda>
{
  template <class _Policy, class _InputIterator, class _BinaryPred>
  [[nodiscard]] _CCCL_HOST_API static ::cuda::std::pair<_InputIterator, _InputIterator> __par_impl(
    [[maybe_unused]] const _Policy& __policy, _InputIterator __first, _InputIterator __last, _BinaryPred __pred)
  {
    const auto __stream = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStream_t{}}, __policy);
    const auto __ctx    = ::cuda::std::execution::__pstl_ensure_current_ctx_for(__policy);

    using _ValueT  = iter_value_t<_InputIterator>;
    using _ResultT = __minmax_element_result<_ValueT>;

    const auto __count = static_cast<int64_t>(::cuda::std::distance(__first, __last));

    // cuda::std::minmax_element returns the *first* minimum and the *last* maximum. Both are found
    // in a single traversal of the range: every element enters the reduction as an accumulator
    // {value, value, index, index} and the reduction operator resolves ties via the offsets. The
    // initial value is flagged with offset -1 so that it loses against every element of the range.
    __minmax_element_transform_fn<_InputIterator> __transform_op{__first};
    __minmax_element_reduce_fn<_ValueT, _BinaryPred> __reduce_op{::cuda::std::move(__pred)};
    _ResultT __init{_ValueT{}, _ValueT{}, -1, -1};

    // Determine temporary device storage requirements for the reduction
    size_t __num_bytes = 0;
    _CCCL_TRY_CUDA_API(
      CUB_NS_QUALIFIER::DeviceReduce::TransformReduce,
      "__pstl_cuda_minmax_element: determination of device storage for cub::DeviceReduce::TransformReduce failed",
      static_cast<void*>(nullptr),
      __num_bytes,
      ::cuda::counting_iterator<int64_t>{0},
      static_cast<_ResultT*>(nullptr),
      __count,
      __reduce_op,
      __transform_op,
      __init);

    _ResultT __result;
    {
      __temporary_storage<_ResultT> __storage{__policy, __num_bytes, 1};

      // Run the reduction
      _CCCL_TRY_CUDA_API(
        CUB_NS_QUALIFIER::DeviceReduce::TransformReduce,
        "__pstl_cuda_minmax_element: kernel launch of cub::DeviceReduce::TransformReduce failed",
        __storage.__get_temp_storage(),
        __num_bytes,
        ::cuda::counting_iterator<int64_t>{0},
        __storage.template __get_ptr<0, _ResultT>(),
        __count,
        ::cuda::std::move(__reduce_op),
        ::cuda::std::move(__transform_op),
        ::cuda::std::move(__init),
        __stream.get());

      // Copy the result back from storage
      _CCCL_TRY_CUDA_API(
        ::cudaMemcpyAsync,
        "__pstl_cuda_minmax_element: copy of result from device to host failed",
        ::cuda::std::addressof(__result),
        __storage.template __get_raw_ptr<0>(),
        sizeof(_ResultT),
        ::cudaMemcpyDefault,
        __stream.get());
    }

    __stream.sync();

    using __diff_t = iter_difference_t<_InputIterator>;
    return ::cuda::std::pair<_InputIterator, _InputIterator>{
      __first + static_cast<__diff_t>(__result.__min_index_), __first + static_cast<__diff_t>(__result.__max_index_)};
  }

  template <class _Policy, class _InputIterator, class _BinaryPred>
  [[nodiscard]] _CCCL_HOST_API ::cuda::std::pair<_InputIterator, _InputIterator> operator()(
    [[maybe_unused]] const _Policy& __policy, _InputIterator __first, _InputIterator __last, _BinaryPred __pred) const
  {
    if constexpr (::cuda::std::__has_random_access_traversal<_InputIterator>)
    {
      _CCCL_TRY
      {
        return __par_impl(__policy, ::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__pred));
      }
      _CCCL_CATCH (const ::cuda::cuda_error& __err)
      {
        if (__err.status() == cudaErrorMemoryAllocation)
        {
          _CCCL_THROW(::std::bad_alloc);
        }
        else
        {
          _CCCL_RETHROW;
        }
      }
      _CCCL_CATCH_FALLTHROUGH
    }
    else
    {
      static_assert(__always_false_v<_Policy>,
                    "__pstl_dispatch: CUDA backend of cuda::std::minmax_element requires at least random access "
                    "iterators");
      return ::cuda::std::minmax_element(
        ::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__pred));
    }
  }
};

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA_STD___PSTL_CUDA_MINMAX_ELEMENT_H
