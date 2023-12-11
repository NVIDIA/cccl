//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_ALIGNED_STORAGE_H
#define _LIBCUDACXX___TYPE_TRAITS_ALIGNED_STORAGE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/conditional.h"
#include "../__type_traits/integral_constant.h"
#include "../__type_traits/nat.h"
#include "../__type_traits/type_list.h"
#include "../cstddef"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __align_type
{
    static const size_t value = _LIBCUDACXX_PREFERRED_ALIGNOF(_Tp);
    typedef _Tp type;
};

struct __struct_double {long double __lx;};
struct __struct_double4 {double __lx[4];};

typedef
    __type_list<__align_type<unsigned char>,
    __type_list<__align_type<unsigned short>,
    __type_list<__align_type<unsigned int>,
    __type_list<__align_type<unsigned long>,
    __type_list<__align_type<unsigned long long>,
    __type_list<__align_type<double>,
    __type_list<__align_type<long double>,
    __type_list<__align_type<__struct_double>,
    __type_list<__align_type<__struct_double4>,
    __type_list<__align_type<int*>,
    __nat
    > > > > > > > > > > __all_types;

template <size_t _Align>
struct _ALIGNAS(_Align) __fallback_overaligned {};

template <class _TL, size_t _Align> struct __find_pod;

template <class _Hp, size_t _Align>
struct __find_pod<__type_list<_Hp, __nat>, _Align>
{
    typedef __conditional_t<_Align == _Hp::value, typename _Hp::type, __fallback_overaligned<_Align> > type;
};

template <class _Hp, class _Tp, size_t _Align>
struct __find_pod<__type_list<_Hp, _Tp>, _Align>
{
    typedef __conditional_t<_Align == _Hp::value, typename _Hp::type, typename __find_pod<_Tp, _Align>::type> type;
};

template <class _TL, size_t _Len> struct __find_max_align;

template <class _Hp, size_t _Len>
struct __find_max_align<__type_list<_Hp, __nat>, _Len> : public integral_constant<size_t, _Hp::value> {};

template <size_t _Len, size_t _A1, size_t _A2>
struct __select_align
{
private:
    static const size_t __min = _A2 < _A1 ? _A2 : _A1;
    static const size_t __max = _A1 < _A2 ? _A2 : _A1;
public:
    static const size_t value = _Len < __max ? __min : __max;
};

template <class _Hp, class _Tp, size_t _Len>
struct __find_max_align<__type_list<_Hp, _Tp>, _Len>
    : public integral_constant<size_t, __select_align<_Len, _Hp::value, __find_max_align<_Tp, _Len>::value>::value> {};

template <size_t _Len, size_t _Align = __find_max_align<__all_types, _Len>::value>
struct _LIBCUDACXX_TEMPLATE_VIS aligned_storage
{
    typedef typename __find_pod<__all_types, _Align>::type _Aligner;
    union type
    {
        _Aligner __align;
        unsigned char __data[(_Len + _Align - 1)/_Align * _Align];
    };
};

#if _CCCL_STD_VER > 2011
template <size_t _Len, size_t _Align = __find_max_align<__all_types, _Len>::value>
    using aligned_storage_t = typename aligned_storage<_Len, _Align>::type;
#endif

#define _CREATE_ALIGNED_STORAGE_SPECIALIZATION(n) \
template <size_t _Len>\
struct _LIBCUDACXX_TEMPLATE_VIS aligned_storage<_Len, n>\
{\
    struct _ALIGNAS(n) type\
    {\
        unsigned char __lx[(_Len + n - 1)/n * n];\
    };\
}

_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x1);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x2);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x4);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x8);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x10);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x20);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x40);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x80);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x100);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x200);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x400);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x800);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x1000);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x2000);
// PE/COFF does not support alignment beyond 8192 (=0x2000)
#if !defined(_LIBCUDACXX_OBJECT_FORMAT_COFF)
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x4000);
#endif // !defined(_LIBCUDACXX_OBJECT_FORMAT_COFF)

#undef _CREATE_ALIGNED_STORAGE_SPECIALIZATION

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_ALIGNED_STORAGE_H
