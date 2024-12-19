//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DETAIL_BASIC_ANY_FWD_H
#define __CUDAX_DETAIL_BASIC_ANY_FWD_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/cstddef> // for max_align_t
#include <cuda/std/cstdint> // for uint8_t

#include <cuda/experimental/__detail/config.cuh> // IWYU pragma: keep export
#include <cuda/experimental/__detail/utility.cuh> // IWYU pragma: keep export

// Some functions defined here have their addresses appear in public types
// (e.g., in `cudax::overrides_for` specializations). If the function is declared
// `__attribute__((visibility("hidden")))`, and if the address appears, say, in
// the type of a member of a class that is declared
// `__attribute__((visibility("default")))`, GCC complains bitterly. So we
// avoid declaring those functions `hidden`. Instead of the typical `_CUDAX_HOST_API`
// macro, we use `_CUDAX_PUBLIC_API` for those functions.
#define _CUDAX_PUBLIC_API _CCCL_HOST

_CCCL_PUSH_MACROS
#undef interface

namespace cuda::experimental
{
template <class _Interface>
struct __ireference;

template <class _Interface>
struct _CCCL_TYPE_VISIBILITY_DEFAULT basic_any;

template <class _Interface>
struct _CCCL_DECLSPEC_EMPTY_BASES basic_any<__ireference<_Interface>>;

template <class _Interface>
struct basic_any<_Interface*>;

template <class _Interface>
struct basic_any<_Interface&>;

template <class _InterfaceOrModel, auto... _VirtualFnsOrOverrides>
struct overrides_for;

template <class _Interface, auto... _Mbrs>
struct _CCCL_DECLSPEC_EMPTY_BASES __basic_vtable;

struct __rtti_base;

struct __rtti;

template <size_t NbrBases>
struct __rtti_ex;

template <class...>
struct extends;

template <template <class...> class, class = extends<>, size_t = 0, size_t = 0>
struct interface;

template <class _Interface, class... _Super>
using __rebind_interface _CCCL_NODEBUG_ALIAS = typename _Interface::template __rebind<_Super...>;

struct iunknown;

template <class...>
struct __iset;

template <class...>
struct __iset_vptr;

template <class...>
struct imovable;

template <class...>
struct icopyable;

template <class...>
struct iequality_comparable;

template <class... _Tp>
using __tag _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__type_list_ptr<_Tp...>;

template <auto...>
struct __ctag_;

template <auto... _Is>
using __ctag _CCCL_NODEBUG_ALIAS = __ctag_<_Is...>*;

constexpr size_t __word                 = sizeof(void*);
constexpr size_t __default_buffer_size  = 3 * __word;
constexpr size_t __default_buffer_align = alignof(_CUDA_VSTD::max_align_t);

using __make_type_list _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__type_quote<_CUDA_VSTD::__type_list>;

_CCCL_NORETURN _CUDAX_HOST_API void __throw_bad_any_cast();

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

} // namespace cuda::experimental

_CCCL_POP_MACROS

#endif // __CUDAX_DETAIL_BASIC_ANY_FWD_H
