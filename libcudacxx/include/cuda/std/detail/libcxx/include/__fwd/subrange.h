//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FWD_SUBRANGE_H
#define _LIBCUDACXX___FWD_SUBRANGE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#include "../__iterator/concepts.h"

#if _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

enum class _LIBCUDACXX_ENUM_VIS subrange_kind : bool { unsized, sized };

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI
#if _LIBCUDACXX_STD_VER > 17
template<input_or_output_iterator _Iter, sentinel_for<_Iter> _Sent = _Iter,
         subrange_kind _Kind = sized_sentinel_for<_Sent, _Iter> ? subrange_kind::sized : subrange_kind::unsized>
  requires (_Kind == subrange_kind::sized || !sized_sentinel_for<_Sent, _Iter>)
class _LIBCUDACXX_TEMPLATE_VIS subrange;
#else
template<class _Iter, class _Sent = _Iter,
          subrange_kind _Kind = sized_sentinel_for<_Sent, _Iter>
            ? subrange_kind::sized
            : subrange_kind::unsized,
  enable_if_t<input_or_output_iterator<_Iter>, int> = 0,
  enable_if_t<sentinel_for<_Sent, _Iter>, int> = 0,
  enable_if_t<(_Kind == subrange_kind::sized || !sized_sentinel_for<_Sent, _Iter>), int> = 0>
class _LIBCUDACXX_TEMPLATE_VIS subrange;
#endif // _LIBCUDACXX_STD_VER < 20

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI
_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _LIBCUDACXX_STD_VER > 14

#endif // _LIBCUDACXX___FWD_SUBRANGE_H
