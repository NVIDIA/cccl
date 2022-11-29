//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_DECAY_H
#define _LIBCUDACXX___TYPE_TRAITS_DECAY_H

#ifndef __cuda_std__
#include <__config>
#include <__type_traits/add_pointer.h>
#include <__type_traits/conditional.h>
#include <__type_traits/is_array.h>
#include <__type_traits/is_function.h>
#include <__type_traits/is_referenceable.h>
#include <__type_traits/remove_cv.h>
#include <__type_traits/remove_extent.h>
#include <__type_traits/remove_reference.h>
#else
#include "../__type_traits/add_pointer.h"
#include "../__type_traits/conditional.h"
#include "../__type_traits/is_array.h"
#include "../__type_traits/is_function.h"
#include "../__type_traits/is_referenceable.h"
#include "../__type_traits/remove_cv.h"
#include "../__type_traits/remove_extent.h"
#include "../__type_traits/remove_reference.h"
#endif // __cuda_std__

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if __has_builtin(__decay)
template <class _Tp>
struct decay {
  using type _LIBCUDACXX_NODEBUG = __decay(_Tp);
};

#else

template <class _Up, bool>
struct __decay {
    typedef _LIBCUDACXX_NODEBUG __remove_cv_t<_Up> type;
};

template <class _Up>
struct __decay<_Up, true> {
public:
    typedef _LIBCUDACXX_NODEBUG typename conditional
                     <
                         is_array<_Up>::value,
                         __remove_extent_t<_Up>*,
                         typename conditional
                         <
                              is_function<_Up>::value,
                              typename add_pointer<_Up>::type,
                              __remove_cv_t<_Up>
                         >::type
                     >::type type;
};

template <class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS decay
{
private:
    typedef _LIBCUDACXX_NODEBUG __libcpp_remove_reference_t<_Tp> _Up;
public:
  typedef _LIBCUDACXX_NODEBUG typename __decay<_Up, __libcpp_is_referenceable<_Up>::value>::type type;
};
#endif // __has_builtin(__decay)

#if _LIBCUDACXX_STD_VER > 11
template <class _Tp> using decay_t = typename decay<_Tp>::type;
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_DECAY_H
