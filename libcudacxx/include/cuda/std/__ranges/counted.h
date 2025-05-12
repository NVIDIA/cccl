// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_COUNTED_H
#define _LIBCUDACXX___RANGES_COUNTED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/counted_iterator.h>
#include <cuda/std/__iterator/default_sentinel.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__ranges/subrange.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/span>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__counted)

struct __fn
{
  _CCCL_TEMPLATE(class _It)
  _CCCL_REQUIRES(contiguous_iterator<_It>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr auto __go(_It __it, iter_difference_t<_It> __count) noexcept(
    noexcept(span(_CUDA_VSTD::to_address(__it), static_cast<size_t>(__count))))
  // Deliberately omit return-type SFINAE, because to_address is not SFINAE-friendly
  {
    return span(_CUDA_VSTD::to_address(__it), static_cast<size_t>(__count));
  }

  _CCCL_TEMPLATE(class _It)
  _CCCL_REQUIRES((!contiguous_iterator<_It>) _CCCL_AND random_access_iterator<_It>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr auto
  __go(_It __it, iter_difference_t<_It> __count) noexcept(noexcept(subrange(__it, __it + __count))) -> subrange<_It>
  {
    return subrange(__it, __it + __count);
  }

  _CCCL_TEMPLATE(class _It)
  _CCCL_REQUIRES((!contiguous_iterator<_It>) _CCCL_AND(!random_access_iterator<_It>))
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr auto __go(_It __it, iter_difference_t<_It> __count) noexcept(
    noexcept(subrange(counted_iterator(_CUDA_VSTD::move(__it), __count), default_sentinel)))
    -> subrange<counted_iterator<_It>, default_sentinel_t>
  {
    return subrange(counted_iterator(_CUDA_VSTD::move(__it), __count), default_sentinel);
  }

  _CCCL_TEMPLATE(class _It, class _Diff)
  _CCCL_REQUIRES(convertible_to<_Diff, iter_difference_t<_It>> _CCCL_AND input_or_output_iterator<decay_t<_It>>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_It&& __it, _Diff&& __count) const
    noexcept(noexcept(__go(_CUDA_VSTD::forward<_It>(__it), _CUDA_VSTD::forward<_Diff>(__count))))
      -> decltype(__go(_CUDA_VSTD::forward<_It>(__it), _CUDA_VSTD::forward<_Diff>(__count)))
  {
    return __go(_CUDA_VSTD::forward<_It>(__it), _CUDA_VSTD::forward<_Diff>(__count));
  }
};

_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto counted = __counted::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_VIEWS

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___RANGES_COUNTED_H
