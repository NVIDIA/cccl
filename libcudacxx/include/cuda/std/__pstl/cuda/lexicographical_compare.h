//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_CUDA_LEXICOGRAPHICAL_COMPARE_H
#define _CUDA_STD___PSTL_CUDA_LEXICOGRAPHICAL_COMPARE_H

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

#  include <cub/device/device_find.cuh>

_CCCL_DIAG_POP

#  include <cuda/__execution/policy.h>
#  include <cuda/__functional/call_or.h>
#  include <cuda/__iterator/zip_transform_iterator.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__algorithm/lexicographical_compare.h>
#  include <cuda/std/__algorithm/min.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__iterator/concepts.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__memory/addressof.h>
#  include <cuda/std/__pstl/cuda/temporary_storage.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/common_type.h>
#  include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#  include <cuda/std/__utility/move.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

// Tri-valued result of comparing a pair (a, b) drawn from the two input ranges
// under `comp`: `__less` if `a` orders before `b`, `__greater` if `b` orders
// before `a`, `__equal` otherwise.
enum class __lex_ordering : int
{
  __less    = -1,
  __equal   = 0,
  __greater = 1,
};

// Maps a pair (a, b) to its `__lex_ordering`. Used as the transform of a
// `cuda::zip_transform_iterator`, so the per-element state is computed lazily by
// the device kernels that walk the iterator. Both a const and a non-const call
// operator are provided so that comparators with a non-const `operator()` work.
template <class _Compare>
struct __lex_state_fn
{
  _Compare __comp_;

  _CCCL_API explicit constexpr __lex_state_fn(_Compare __comp) noexcept(is_nothrow_move_constructible_v<_Compare>)
      : __comp_(::cuda::std::move(__comp))
  {}

  template <class _Tp, class _Up>
  [[nodiscard]] _CCCL_DEVICE_API constexpr __lex_ordering operator()(const _Tp& __a, const _Up& __b)
  {
    if (__comp_(__a, __b))
    {
      return __lex_ordering::__less;
    }
    if (__comp_(__b, __a))
    {
      return __lex_ordering::__greater;
    }
    return __lex_ordering::__equal;
  }

  template <class _Tp, class _Up>
  [[nodiscard]] _CCCL_DEVICE_API constexpr __lex_ordering operator()(const _Tp& __a, const _Up& __b) const
  {
    if (__comp_(__a, __b))
    {
      return __lex_ordering::__less;
    }
    if (__comp_(__b, __a))
    {
      return __lex_ordering::__greater;
    }
    return __lex_ordering::__equal;
  }
};

// Predicate handed to the find pass: locates the first non-equivalent pair.
struct __lex_non_equal
{
  [[nodiscard]] _CCCL_DEVICE_API constexpr bool operator()(__lex_ordering __state) const noexcept
  {
    return __state != __lex_ordering::__equal;
  }
};

// Resolves the final answer on device from the index found by the find pass,
// avoiding a host round-trip and a second device reduction. `__result` holds the
// first non-equivalent index on entry and the boolean answer (0/1) on exit:
//   - if no divergence was found within the common prefix, the shorter range is
//     "less" (`__shorter_is_less`);
//   - otherwise the ordering at the divergence position decides.
template <class _StateIter, class _OffsetType>
_CCCL_KERNEL_ATTRIBUTES void __lexicographical_compare_result_kernel(
  _StateIter __state_first, _OffsetType __count, bool __shorter_is_less, _OffsetType* __result)
{
  const _OffsetType __k = *__result;
  *__result =
    static_cast<_OffsetType>((__k == __count) ? __shorter_is_less : (*(__state_first + __k) == __lex_ordering::__less));
}

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <>
struct __pstl_dispatch<__pstl_algorithm::__lexicographical_compare, __execution_backend::__cuda>
{
  template <class _Policy, class _InputIter1, class _InputIter2, class _Compare>
  [[nodiscard]] _CCCL_HOST_API static bool __par_impl(
    const _Policy& __policy,
    _InputIter1 __first1,
    _InputIter1 __last1,
    _InputIter2 __first2,
    _InputIter2 __last2,
    _Compare __comp)
  {
    if (__first1 == __last1)
    {
      return __first2 != __last2;
    }
    if (__first2 == __last2)
    {
      return false;
    }

    using _OffsetType   = common_type_t<iter_difference_t<_InputIter1>, iter_difference_t<_InputIter2>>;
    const auto __count1 = static_cast<_OffsetType>(::cuda::std::distance(__first1, __last1));
    const auto __count2 = static_cast<_OffsetType>(::cuda::std::distance(__first2, __last2));
    const auto __count  = ::cuda::std::min(__count1, __count2);

    // Lazy per-element ordering over the two input ranges, capped at the common
    // length so neither side reads past its own end.
    auto __state_first = ::cuda::zip_transform_iterator{
      __lex_state_fn<_Compare>{::cuda::std::move(__comp)}, ::cuda::std::move(__first1), ::cuda::std::move(__first2)};

    // Determine the find pass' temporary storage requirement up front.
    size_t __find_bytes = 0;
    _CCCL_TRY_CUDA_API(
      CUB_NS_QUALIFIER::DeviceFind::FindIf,
      "__pstl_cuda_lexicographical_compare: determining temporary storage for cub::DeviceFind::FindIf failed",
      static_cast<void*>(nullptr),
      __find_bytes,
      __state_first,
      static_cast<_OffsetType*>(nullptr),
      __lex_non_equal{},
      __count);

    auto __stream = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStream_t{}}, __policy);

    // Single allocation: slot<0> holds the find index and then the boolean answer,
    // plus the find pass' scratch region.
    __temporary_storage<_OffsetType> __storage{__policy, __find_bytes, 1};
    auto __result_ptr = __storage.template __get_raw_ptr<0>();

    // Pass 1 (early-terminating): index of the first non-equivalent pair.
    _CCCL_TRY_CUDA_API(
      CUB_NS_QUALIFIER::DeviceFind::FindIf,
      "__pstl_cuda_lexicographical_compare: kernel launch of cub::DeviceFind::FindIf failed",
      __storage.__get_temp_storage(),
      __find_bytes,
      __state_first,
      __storage.template __get_ptr<0>(),
      __lex_non_equal{},
      __count,
      __stream.get());

    // Pass 2: resolve the answer in place on device, so only one value is copied
    // back to the host (no second reduce launch, no offset round-trip).
    const bool __shorter_is_less = (__count1 < __count2);
    const void* __kernel =
      reinterpret_cast<const void*>(&__lexicographical_compare_result_kernel<decltype(__state_first), _OffsetType>);
    void* __kernel_args[] = {
      const_cast<void*>(reinterpret_cast<const void*>(::cuda::std::addressof(__state_first))),
      const_cast<void*>(reinterpret_cast<const void*>(::cuda::std::addressof(__count))),
      const_cast<void*>(reinterpret_cast<const void*>(::cuda::std::addressof(__shorter_is_less))),
      const_cast<void*>(reinterpret_cast<const void*>(::cuda::std::addressof(__result_ptr)))};
    _CCCL_TRY_CUDA_API(
      ::cudaLaunchKernel,
      "__pstl_cuda_lexicographical_compare: kernel launch of result resolution failed",
      __kernel,
      ::dim3{1},
      ::dim3{1},
      __kernel_args,
      size_t{0},
      __stream.get());

    _OffsetType __result;
    _CCCL_TRY_CUDA_API(
      ::cudaMemcpyAsync,
      "__pstl_cuda_lexicographical_compare: copy of result from device to host failed",
      ::cuda::std::addressof(__result),
      __result_ptr,
      sizeof(_OffsetType),
      ::cudaMemcpyDefault,
      __stream.get());
    __stream.sync();

    return __result != _OffsetType{0};
  }

  _CCCL_TEMPLATE(class _Policy, class _InputIter1, class _InputIter2, class _Compare)
  _CCCL_REQUIRES(__has_forward_traversal<_InputIter1> _CCCL_AND __has_forward_traversal<_InputIter2>)
  [[nodiscard]] _CCCL_HOST_API bool operator()(
    [[maybe_unused]] const _Policy& __policy,
    _InputIter1 __first1,
    _InputIter1 __last1,
    _InputIter2 __first2,
    _InputIter2 __last2,
    _Compare __comp) const
  {
    if constexpr (__has_random_access_traversal<_InputIter1> && __has_random_access_traversal<_InputIter2>)
    {
      try
      {
        return __par_impl(
          __policy,
          ::cuda::std::move(__first1),
          ::cuda::std::move(__last1),
          ::cuda::std::move(__first2),
          ::cuda::std::move(__last2),
          ::cuda::std::move(__comp));
      }
      catch (const ::cuda::cuda_error& __err)
      {
        if (__err.status() == cudaErrorMemoryAllocation)
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
                    "CUDA backend of cuda::std::lexicographical_compare requires random access iterators");
      return ::cuda::std::lexicographical_compare(
        ::cuda::std::move(__first1),
        ::cuda::std::move(__last1),
        ::cuda::std::move(__first2),
        ::cuda::std::move(__last2),
        ::cuda::std::move(__comp));
    }
  }
};

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA_STD___PSTL_CUDA_LEXICOGRAPHICAL_COMPARE_H
