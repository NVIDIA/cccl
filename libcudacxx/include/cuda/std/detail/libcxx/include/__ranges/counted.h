// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_COUNTED_H
#define _LIBCUDACXX___RANGES_COUNTED_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__concepts/convertible_to.h"
#include "../__iterator/concepts.h"
#include "../__iterator/counted_iterator.h"
#include "../__iterator/default_sentinel.h"
#include "../__iterator/incrementable_traits.h"
#include "../__iterator/iterator_traits.h"
#include "../__memory/pointer_traits.h"
#include "../__ranges/subrange.h"
#include "../__utility/forward.h"
#include "../__utility/move.h"
#include "../span"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif


#if _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__counted)

  struct __fn {
    _LIBCUDACXX_TEMPLATE(class _It)
      (requires contiguous_iterator<_It>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    static constexpr auto __go(_It __it, iter_difference_t<_It> __count)
      noexcept(noexcept(span(_CUDA_VSTD::to_address(__it), static_cast<size_t>(__count))))
      // Deliberately omit return-type SFINAE, because to_address is not SFINAE-friendly
      { return          span(_CUDA_VSTD::to_address(__it), static_cast<size_t>(__count)); }

    _LIBCUDACXX_TEMPLATE(class _It)
      (requires (!contiguous_iterator<_It>) _LIBCUDACXX_AND random_access_iterator<_It>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    static constexpr auto __go(_It __it, iter_difference_t<_It> __count)
      noexcept(noexcept(subrange(__it, __it + __count)))
      -> decltype(      subrange(__it, __it + __count))
      { return          subrange(__it, __it + __count); }

    _LIBCUDACXX_TEMPLATE(class _It)
      (requires (!contiguous_iterator<_It>) _LIBCUDACXX_AND (!random_access_iterator<_It>))
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    static constexpr auto __go(_It __it, iter_difference_t<_It> __count)
      noexcept(noexcept(subrange(counted_iterator(_CUDA_VSTD::move(__it), __count), default_sentinel)))
      -> decltype(      subrange(counted_iterator(_CUDA_VSTD::move(__it), __count), default_sentinel))
      { return          subrange(counted_iterator(_CUDA_VSTD::move(__it), __count), default_sentinel); }

    _LIBCUDACXX_TEMPLATE(class _It, class _Diff)
      (requires convertible_to<_Diff, iter_difference_t<_It>> _LIBCUDACXX_AND
                input_or_output_iterator<decay_t<_It>>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_It&& __it, _Diff&& __count) const
      noexcept(noexcept(__go(_CUDA_VSTD::forward<_It>(__it), _CUDA_VSTD::forward<_Diff>(__count))))
      -> decltype(      __go(_CUDA_VSTD::forward<_It>(__it), _CUDA_VSTD::forward<_Diff>(__count)))
      { return          __go(_CUDA_VSTD::forward<_It>(__it), _CUDA_VSTD::forward<_Diff>(__count)); }
  };

_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto counted = __counted::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_VIEWS

#endif // _LIBCUDACXX_STD_VER > 14

#endif // _LIBCUDACXX___RANGES_COUNTED_H
