//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_CHAR_LIKE_TYPE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_CHAR_LIKE_TYPE_H

#ifndef __cuda_std__
#include <__config>
#include <__type_traits/conjunction.h>
#include <__type_traits/is_standard_layout.h>
#include <__type_traits/is_trivial.h>
#else
#include "../__type_traits/conjunction.h"
#include "../__type_traits/is_standard_layout.h"
#include "../__type_traits/is_trivial.h"
#endif // __cuda_std__

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _CharT>
using _IsCharLikeType = _And<is_standard_layout<_CharT>, is_trivial<_CharT> >;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_CHAR_LIKE_TYPE_H
