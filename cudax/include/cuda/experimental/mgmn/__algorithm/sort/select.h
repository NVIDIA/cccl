// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_SELECT_H
#define _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_SELECT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuda_runtime.h>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

_CCCL_BEGIN_NAMESPACE_CUDA_MGMN
namespace __detail::__sort
{
_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT
using __index_type = ::cuda::std::int64_t;

// Counts and both possible next pivots travel together. Once global counts choose
// a branch, its candidates are already available, saving one collective per round.
// Keep keys in their native representation, including user-defined key types.
template <class _Tp>
struct __selection_message
{
  __index_type __less_;
  __index_type __equal_;
  __index_type __greater_;
  _Tp __lower_pivot_;
  _Tp __upper_pivot_;
};

template <class _Tp>
struct __selection_state
{
  __index_type __first_;
  __index_type __last_;
  __index_type __lower_;
  __index_type __upper_;
  __index_type __cut_;
  __index_type __remaining_;
  _Tp __pivot_;
  bool __done_;
};

template <class _Tp, class _BinaryOp>
[[nodiscard]] _CCCL_DEVICE_API __index_type __partition_bound(
  const _Tp* __keys, __index_type __first, __index_type __last, const _Tp& __key, _BinaryOp __cmp, bool __upper)
{
  while (__first < __last)
  {
    const auto __mid    = __first + (__last - __first) / 2;
    const bool __before = __upper ? !__cmp(__key, __keys[__mid]) : __cmp(__keys[__mid], __key);
    if (__before)
    {
      __first = __mid + 1;
    }
    else
    {
      __last = __mid;
    }
  }
  return __first;
}

template <class _Tp>
[[nodiscard]] _CCCL_DEVICE_API __index_type __weight(const __selection_message<_Tp>& __message, bool __lower)
{
  return __lower ? __message.__less_ : __message.__greater_;
}

template <class _Tp>
[[nodiscard]] _CCCL_DEVICE_API const _Tp& __candidate(const __selection_message<_Tp>& __message, bool __lower)
{
  return __lower ? __message.__lower_pivot_ : __message.__upper_pivot_;
}

template <class _Tp, class _BinaryOp>
[[nodiscard]] _CCCL_DEVICE_API _Tp
__choose_pivot(const __selection_message<_Tp>* __messages, int __ranks, int __boundary, bool __lower, _BinaryOp __cmp)
{
  const auto __stride  = static_cast<::cuda::std::size_t>(__ranks) + 1;
  __index_type __total = 0;
  int __first          = -1;
  for (int __peer = 0; __peer < __ranks; ++__peer)
  {
    const auto __count =
      ::cuda::experimental::mgmn::__detail::__sort::__weight(__messages[__peer * __stride + __boundary], __lower);
    __total += __count;
    if (__count > 0 && __first < 0)
    {
      __first = __peer;
    }
  }
  if (__first < 0)
  {
    return _Tp{};
  }
  const auto& __initial = __messages[__first * __stride + __boundary];
  // Prefer the inexpensive first candidate while its interval is substantial.
  // Otherwise the weighted median prevents a tiny rank from dominating progress.
  if (::cuda::experimental::mgmn::__detail::__sort::__weight(__initial, __lower)
      >= __total / (__index_type{2} * __ranks))
  {
    return ::cuda::experimental::mgmn::__detail::__sort::__candidate(__initial, __lower);
  }
  for (int __peer = 0; __peer < __ranks; ++__peer)
  {
    const auto& __message = __messages[__peer * __stride + __boundary];
    if (::cuda::experimental::mgmn::__detail::__sort::__weight(__message, __lower) == 0)
    {
      continue;
    }
    const auto& __pivot  = ::cuda::experimental::mgmn::__detail::__sort::__candidate(__message, __lower);
    __index_type __less  = 0;
    __index_type __equal = 0;
    for (int __other = 0; __other < __ranks; ++__other)
    {
      const auto& __sample = __messages[__other * __stride + __boundary];
      const auto __count   = ::cuda::experimental::mgmn::__detail::__sort::__weight(__sample, __lower);
      if (__count == 0)
      {
        continue;
      }
      const auto& __key = ::cuda::experimental::mgmn::__detail::__sort::__candidate(__sample, __lower);
      if (__cmp(__key, __pivot))
      {
        __less += __count;
      }
      else if (!__cmp(__pivot, __key))
      {
        __equal += __count;
      }
    }
    if (__less <= __total / 2 && __total / 2 - __less < __equal)
    {
      return __pivot;
    }
  }
  _CCCL_UNREACHABLE();
}

// Uniform disjoint runs need only a rank permutation. The zero-boundary slot
// carries endpoints in the first exchange. Touching endpoints follow source rank.
template <class _Tp, class _BinaryOp>
[[nodiscard]] _CCCL_DEVICE_API int
__disjoint_position(const __selection_message<_Tp>* __messages, int __rank, int __ranks, _BinaryOp __cmp)
{
  const auto __stride = static_cast<::cuda::std::size_t>(__ranks) + 1;
  for (int __peer = 0; __peer < __ranks; ++__peer)
  {
    const auto& __left = __messages[__peer * __stride];
    for (int __other = __peer + 1; __other < __ranks; ++__other)
    {
      const auto& __right = __messages[__other * __stride];
      if (__cmp(__right.__lower_pivot_, __left.__upper_pivot_) && !__cmp(__right.__upper_pivot_, __left.__lower_pivot_))
      {
        return -1;
      }
    }
  }
  int __position        = 0;
  const auto& __minimum = __messages[__rank * __stride].__lower_pivot_;
  for (int __peer = 0; __peer < __ranks; ++__peer)
  {
    const auto& __other = __messages[__peer * __stride].__lower_pivot_;
    if (__cmp(__other, __minimum) || (!__cmp(__minimum, __other) && __peer < __rank))
    {
      ++__position;
    }
  }
  return __position;
}

template <class _Tp, class _BinaryOp>
_CCCL_DEVICE_API void
__count_pivot(__selection_state<_Tp>& __state, const _Tp* __keys, __selection_message<_Tp>& __message, _BinaryOp __cmp)
{
  __message = {};
  if (__state.__done_)
  {
    return;
  }
  __state.__lower_ = ::cuda::experimental::mgmn::__detail::__sort::__partition_bound(
    __keys, __state.__first_, __state.__last_, __state.__pivot_, __cmp, false);
  __state.__upper_ = ::cuda::experimental::mgmn::__detail::__sort::__partition_bound(
    __keys, __state.__lower_, __state.__last_, __state.__pivot_, __cmp, true);
  __message.__less_    = __state.__lower_ - __state.__first_;
  __message.__equal_   = __state.__upper_ - __state.__lower_;
  __message.__greater_ = __state.__last_ - __state.__upper_;
  if (__message.__less_ != 0)
  {
    __message.__lower_pivot_ = __keys[__state.__first_ + __message.__less_ / 2];
  }
  if (__message.__greater_ != 0)
  {
    __message.__upper_pivot_ = __keys[__state.__upper_ + __message.__greater_ / 2];
  }
}

enum class __selection_phase
{
  __initialize,
  __first_count,
  __advance,
  __export_cuts
};

template <__selection_phase _Phase, class _Tp, class _BinaryOp>
_CCCL_DEVICE_API void __select_step(
  const _Tp* __keys,
  __index_type __size,
  int __rank,
  int __ranks,
  __selection_message<_Tp>* __source,
  const __selection_message<_Tp>* __gathered,
  __selection_state<_Tp>* __states,
  int* __unfinished,
  _BinaryOp __cmp,
  const __index_type* __targets,
  bool __uniform,
  __index_type* __cuts)
{
  if constexpr (_Phase == __selection_phase::__advance)
  {
    if (threadIdx.x == 0)
    {
      *__unfinished = 0;
    }
    __syncthreads();
  }
  [[maybe_unused]] __shared__ int __position;
  if constexpr (_Phase == __selection_phase::__first_count)
  {
    if (threadIdx.x == 0)
    {
      __position =
        __uniform && __size > 0
          ? ::cuda::experimental::mgmn::__detail::__sort::__disjoint_position(__gathered, __rank, __ranks, __cmp)
          : -1;
    }
    __syncthreads();
  }
  for (int __boundary = threadIdx.x; __boundary <= __ranks; __boundary += blockDim.x)
  {
    auto& __state = __states[__boundary];
    if constexpr (_Phase == __selection_phase::__initialize)
    {
      __state              = {};
      __state.__last_      = __size;
      __state.__remaining_ = __targets[__boundary];
      __state.__cut_       = __boundary == __ranks ? __size : 0;
      __state.__done_      = __boundary == 0 || __boundary == __ranks;
      auto& __message      = __source[__boundary];
      __message            = {};
      if (!__state.__done_ && __size > 0)
      {
        __message.__greater_     = __size;
        __message.__upper_pivot_ = __keys[__size / 2];
      }
      if (__boundary == 0 && __size > 0)
      {
        __message.__lower_pivot_ = __keys[0];
        __message.__upper_pivot_ = __keys[__size - 1];
      }
    }
    else if constexpr (_Phase == __selection_phase::__export_cuts)
    {
      __cuts[__boundary] = __state.__cut_;
    }
    else
    {
      bool __lower = false;
      if constexpr (_Phase == __selection_phase::__first_count)
      {
        if (__position >= 0)
        {
          __state.__cut_  = __position < __boundary ? __size : 0;
          __state.__done_ = true;
        }
      }
      else if (!__state.__done_)
      {
        __index_type __less         = 0;
        __index_type __equal        = 0;
        __index_type __equal_before = 0;
        const auto __stride         = static_cast<::cuda::std::size_t>(__ranks) + 1;
        for (int __peer = 0; __peer < __ranks; ++__peer)
        {
          const auto& __message = __gathered[__peer * __stride + __boundary];
          __less += __message.__less_;
          __equal += __message.__equal_;
          if (__peer < __rank)
          {
            __equal_before += __message.__equal_;
          }
        }
        if (__state.__remaining_ < __less)
        {
          __state.__last_ = __state.__lower_;
          __lower         = true;
        }
        else if (__state.__remaining_ > __less + __equal)
        {
          __state.__first_ = __state.__upper_;
          __state.__remaining_ -= __less + __equal;
        }
        else
        {
          // Equivalent keys straddling a target are divided exactly in source-rank order.
          auto __take            = __state.__remaining_ - __less - __equal_before;
          __take                 = __take < 0 ? 0 : __take;
          const auto __available = __state.__upper_ - __state.__lower_;
          __state.__cut_         = __state.__lower_ + (__take < __available ? __take : __available);
          __state.__done_        = true;
        }
      }
      if (!__state.__done_)
      {
        __state.__pivot_ =
          ::cuda::experimental::mgmn::__detail::__sort::__choose_pivot(__gathered, __ranks, __boundary, __lower, __cmp);
      }
      ::cuda::experimental::mgmn::__detail::__sort::__count_pivot(__state, __keys, __source[__boundary], __cmp);
      if constexpr (_Phase == __selection_phase::__advance)
      {
        if (!__state.__done_)
        {
          atomicExch(__unfinished, 1);
        }
      }
    }
  }
}

template <__selection_phase _Phase, class _Tp, class _BinaryOp>
_CCCL_KERNEL_ATTRIBUTES void __select(
  const _Tp* __keys,
  __index_type __size,
  int __rank,
  int __ranks,
  __selection_message<_Tp>* __source,
  const __selection_message<_Tp>* __gathered,
  __selection_state<_Tp>* __states,
  int* __unfinished,
  _BinaryOp __cmp,
  const __index_type* __targets,
  bool __uniform,
  __index_type* __cuts)
{
  ::cuda::experimental::mgmn::__detail::__sort::__select_step<_Phase>(
    __keys, __size, __rank, __ranks, __source, __gathered, __states, __unfinished, __cmp, __targets, __uniform, __cuts);
}
_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace __detail::__sort
_CCCL_END_NAMESPACE_CUDA_MGMN

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>
#endif // _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_SELECT_H
