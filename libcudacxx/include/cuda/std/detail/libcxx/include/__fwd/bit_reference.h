//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FWD_BIT_REFERENCE_H
#define _LIBCUDACXX___FWD_BIT_REFERENCE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#if !defined(_LIBCUDACXX_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Cp, bool _IsConst, typename _Cp::__storage_type = 0>
class __bit_iterator;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FWD_BIT_REFERENCE_H
