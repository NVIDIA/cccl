//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___UTILITY_BASIC_ANY_FWD_H
#define _CUDA___UTILITY_BASIC_ANY_FWD_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/cstddef> // for max_align_t
#include <cuda/std/cstdint> // for uint8_t

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Interface>
struct __ireference;

template <class _Interface>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __basic_any;

template <class _Interface>
struct _CCCL_DECLSPEC_EMPTY_BASES __basic_any<__ireference<_Interface>>;

template <class _Interface>
struct __basic_any<_Interface*>;

template <class _Interface>
struct __basic_any<_Interface&>;

template <auto _Value>
using __constant = ::cuda::std::integral_constant<decltype(_Value), _Value>;

template <class _InterfaceOrModel, class... _VirtualFnsOrOverrides>
struct __overrides_list;

template <class _InterfaceOrModel, auto... _VirtualFnsOrOverrides>
using __overrides_for = __overrides_list<_InterfaceOrModel, __constant<_VirtualFnsOrOverrides>...>;

template <class _Interface, auto... _Mbrs>
struct _CCCL_DECLSPEC_EMPTY_BASES __basic_vtable;

struct __rtti_base;

struct __rtti;

template <size_t NbrBases>
struct __rtti_ex;

template <class...>
struct __extends;

template <template <class...> class, class = __extends<>, size_t = 0, size_t = 0>
struct __basic_interface;

template <class _Interface, class... _Super>
using __rebind_interface _CCCL_NODEBUG_ALIAS = typename _Interface::template __rebind<_Super...>;

struct __iunknown;

template <class...>
struct __iset_;

template <class...>
struct __iset_vptr;

template <class...>
struct __imovable;

template <class...>
struct __icopyable;

template <class...>
struct __iequality_comparable;

template <class... _Tp>
using __tag _CCCL_NODEBUG_ALIAS = ::cuda::std::__type_list_ptr<_Tp...>;

template <auto...>
struct __ctag_;

template <auto... _Is>
using __ctag _CCCL_NODEBUG_ALIAS = __ctag_<_Is...>*;

constexpr size_t __word                       = sizeof(void*);
constexpr size_t __default_small_object_size  = 3 * __word;
constexpr size_t __default_small_object_align = alignof(::cuda::std::max_align_t);

using __make_type_list _CCCL_NODEBUG_ALIAS = ::cuda::std::__type_quote<::cuda::std::__type_list>;

[[noreturn]] _CCCL_API void __throw_bad_any_cast();

enum class __vtable_kind : uint8_t
{
  __normal,
  __rtti,
};

inline constexpr uint8_t __basic_any_version = 0;

template <class _Interface>
extern _Interface __remove_ireference_v; // specialized in interfaces.cuh

template <class _Interface>
using __remove_ireference_t _CCCL_NODEBUG_ALIAS = decltype(__remove_ireference_v<_Interface>);

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___UTILITY_BASIC_ANY_FWD_H
