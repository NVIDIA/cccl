// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_REFERENCE_WRAPPER_H
#define _LIBCUDACXX___FUNCTIONAL_REFERENCE_WRAPPER_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__functional/invoke.h"
#include "../__functional/weak_result_type.h"
#include "../__memory/addressof.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/remove_cvref.h"
#include "../__utility/declval.h"
#include "../__utility/forward.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
class _LIBCUDACXX_TEMPLATE_VIS reference_wrapper : public __weak_result_type<_Tp>
{
public:
    // types
    typedef _Tp type;
private:
    type* __f_;

    static _LIBCUDACXX_INLINE_VISIBILITY void __fun(_Tp&) noexcept;
    static void __fun(_Tp&&) = delete;

public:
    template <class _Up, class = __enable_if_t<!__is_same_uncvref<_Up, reference_wrapper>::value, decltype(__fun(declval<_Up>())) > >
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
    reference_wrapper(_Up&& __u) noexcept(noexcept(__fun(declval<_Up>()))) {
        type& __f = static_cast<_Up&&>(__u);
        __f_ = _CUDA_VSTD::addressof(__f);
    }

    // access
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
    operator type&() const noexcept {return *__f_;}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
    type& get() const noexcept {return *__f_;}

    // invoke
    template <class... _ArgTypes>
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
    typename __invoke_of<type&, _ArgTypes...>::type
    operator() (_ArgTypes&&... __args) const
#if _CCCL_STD_VER > 2011
        // Since is_nothrow_invocable requires C++11 LWG3764 is not backported
        // to earlier versions.
        noexcept(_LIBCUDACXX_TRAIT(is_nothrow_invocable, _Tp&, _ArgTypes...))
#endif
    {
        return _CUDA_VSTD::__invoke(get(), _CUDA_VSTD::forward<_ArgTypes>(__args)...);
    }
};

#if _CCCL_STD_VER > 2014 && !defined(_LIBCUDACXX_HAS_NO_DEDUCTION_GUIDES)
template <class _Tp>
_LIBCUDACXX_HOST_DEVICE reference_wrapper(_Tp&) -> reference_wrapper<_Tp>;
#endif

template <class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
reference_wrapper<_Tp>
ref(_Tp& __t) noexcept
{
    return reference_wrapper<_Tp>(__t);
}

template <class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
reference_wrapper<_Tp>
ref(reference_wrapper<_Tp> __t) noexcept
{
    return __t;
}

template <class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
reference_wrapper<const _Tp>
cref(const _Tp& __t) noexcept
{
    return reference_wrapper<const _Tp>(__t);
}

template <class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
reference_wrapper<const _Tp>
cref(reference_wrapper<_Tp> __t) noexcept
{
    return __t;
}

template <class _Tp> void ref(const _Tp&&) = delete;
template <class _Tp> void cref(const _Tp&&) = delete;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FUNCTIONAL_REFERENCE_WRAPPER_H
