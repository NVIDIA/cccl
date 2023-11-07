// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_DEFAULT_SEARCHER_H
#define _LIBCUDACXX___FUNCTIONAL_DEFAULT_SEARCHER_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__algorithm/search.h"
#include "../__functional/identity.h"
#include "../__functional/operations.h"
#include "../__iterator/iterator_traits.h"
#include "../__utility/pair.h"

#if defined(_CCCL_COMPILER_NVHPC) && defined(_CCCL_USE_IMPLICIT_SYSTEM_DEADER)
#pragma GCC system_header
#else // ^^^ _CCCL_COMPILER_NVHPC ^^^ / vvv !_CCCL_COMPILER_NVHPC vvv
_CCCL_IMPLICIT_SYSTEM_HEADER
#endif // !_CCCL_COMPILER_NVHPC

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#ifndef __cuda_std__

#if _LIBCUDACXX_STD_VER > 14

// default searcher
template<class _ForwardIterator, class _BinaryPredicate = equal_to<>>
class _LIBCUDACXX_TEMPLATE_VIS default_searcher {
public:
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
    default_searcher(_ForwardIterator __f, _ForwardIterator __l,
                       _BinaryPredicate __p = _BinaryPredicate())
        : __first_(__f), __last_(__l), __pred_(__p) {}

    template <typename _ForwardIterator2>
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
    pair<_ForwardIterator2, _ForwardIterator2>
    operator () (_ForwardIterator2 __f, _ForwardIterator2 __l) const
    {
        return _CUDA_VSTD::__search(__f, __l, __first_, __last_, __pred_,
            typename _CUDA_VSTD::iterator_traits<_ForwardIterator>::iterator_category(),
            typename _CUDA_VSTD::iterator_traits<_ForwardIterator2>::iterator_category());
    }

private:
    _ForwardIterator __first_;
    _ForwardIterator __last_;
    _BinaryPredicate __pred_;
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(default_searcher);

#endif // _LIBCUDACXX_STD_VER > 14
#endif // __cuda_std__

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FUNCTIONAL_DEFAULT_SEARCHER_H
