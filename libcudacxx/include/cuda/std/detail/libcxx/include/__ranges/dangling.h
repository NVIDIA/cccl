// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANGES_DANGLING_H
#define _LIBCUDACXX___RANGES_DANGLING_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__ranges/access.h"
#include "../__ranges/concepts.h"
#include "../__type_traits/enable_if.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

#if _LIBCUDACXX_STD_VER > 14

struct dangling {
  dangling() = default;
  template <class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr dangling(_Args&&...) noexcept {}
};

#if _LIBCUDACXX_STD_VER > 17
template <range _Rp>
using borrowed_iterator_t = _If<borrowed_range<_Rp>, iterator_t<_Rp>, dangling>;
#else
template <class _Rp>
using borrowed_iterator_t = enable_if_t<range<_Rp>, _If<borrowed_range<_Rp>, iterator_t<_Rp>, dangling>>;
#endif

// borrowed_subrange_t defined in <__ranges/subrange.h>

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _LIBCUDACXX___RANGES_DANGLING_H
