//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FUNCTIONAL_ADDRESS_STABILITY_H
#define _CUDA___FUNCTIONAL_ADDRESS_STABILITY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/not_fn.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__functional/ranges_operations.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_class.h>
#include <cuda/std/__type_traits/is_enum.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! Trait telling whether a function object type F does not rely on the memory addresses of its arguments. The nested
//! value is true when the addresses of the arguments do not matter and arguments can be provided from arbitrary copies
//! of the respective sources. This trait can be specialized for custom function objects types.
//! @see proclaim_copyable_arguments
template <typename F, typename SFINAE = void>
struct proclaims_copyable_arguments : ::cuda::std::false_type
{};

template <typename F, typename... Args>
inline constexpr bool proclaims_copyable_arguments_v = proclaims_copyable_arguments<F, Args...>::value;

// Wrapper for a callable to mark it as permitting copied arguments
template <typename F>
struct __callable_permitting_copied_arguments : F
{
  using F::operator();
};

template <typename F>
struct proclaims_copyable_arguments<__callable_permitting_copied_arguments<F>> : ::cuda::std::true_type
{};

//! Creates a new function object from an existing one, which is marked as permitting its arguments to be copies of
//! whatever source they come from. This implies that the addresses of the arguments are irrelevant to the function
//! object. Some algorithms, like thrust::transform, can benefit from this information and choose a more efficient
//! implementation.
//! @see proclaims_copyable_arguments
template <typename F>
[[nodiscard]] _CCCL_API constexpr auto proclaim_copyable_arguments(F&& f)
  -> __callable_permitting_copied_arguments<::cuda::std::decay_t<F>>
{
  return {::cuda::std::forward<F>(f)};
}

// Specializations for libcu++ function objects are provided here to not pull this include into `<cuda/std/...>` headers

template <typename _Fn>
struct proclaims_copyable_arguments<::cuda::std::__not_fn_t<_Fn>> : proclaims_copyable_arguments<_Fn>
{};

template <typename _Tp>
struct __has_builtin_operators
    : ::cuda::std::bool_constant<!::cuda::std::is_class_v<_Tp> && !::cuda::std::is_enum_v<_Tp>
                                 && !::cuda::std::is_void_v<_Tp>>
{};

#define _LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(functor)                                         \
  /*we know what plus<T> etc. does if T is not a type that could have a weird operatorX() */ \
  template <typename _Tp>                                                                    \
  struct proclaims_copyable_arguments<functor<_Tp>> : ::cuda::__has_builtin_operators<_Tp>   \
  {};                                                                                        \
  /*we do not know what plus<void> etc. does, which depends on the types it is invoked on */ \
  template <>                                                                                \
  struct proclaims_copyable_arguments<functor<void>> : ::cuda::std::false_type               \
  {};

_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(::cuda::std::plus)
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(::cuda::std::minus)
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(::cuda::std::multiplies)
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(::cuda::std::divides)
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(::cuda::std::modulus)
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(::cuda::std::negate)
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(::cuda::std::bit_and)
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(::cuda::std::bit_not)
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(::cuda::std::bit_or)
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(::cuda::std::bit_xor)
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(::cuda::std::equal_to)
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(::cuda::std::not_equal_to)
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(::cuda::std::less)
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(::cuda::std::less_equal)
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(::cuda::std::greater_equal)
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(::cuda::std::greater)
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(::cuda::std::logical_and)
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(::cuda::std::logical_not)
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(::cuda::std::logical_or)

#define _LIBCUDACXX_MARK_RANGE_FUNCTOR_CAN_COPY_ARGUMENTS(functor)                                              \
  /*we do not know what equal_to etc. does, which depends on the types and their operator== it is invoked on */ \
  template <>                                                                                                   \
  struct proclaims_copyable_arguments<functor> : ::cuda::std::false_type                                        \
  {};

_LIBCUDACXX_MARK_RANGE_FUNCTOR_CAN_COPY_ARGUMENTS(::cuda::std::ranges::equal_to)
_LIBCUDACXX_MARK_RANGE_FUNCTOR_CAN_COPY_ARGUMENTS(::cuda::std::ranges::not_equal_to)
_LIBCUDACXX_MARK_RANGE_FUNCTOR_CAN_COPY_ARGUMENTS(::cuda::std::ranges::less)
_LIBCUDACXX_MARK_RANGE_FUNCTOR_CAN_COPY_ARGUMENTS(::cuda::std::ranges::less_equal)
_LIBCUDACXX_MARK_RANGE_FUNCTOR_CAN_COPY_ARGUMENTS(::cuda::std::ranges::greater)
_LIBCUDACXX_MARK_RANGE_FUNCTOR_CAN_COPY_ARGUMENTS(::cuda::std::ranges::greater_equal)

#undef _LIBCUDACXX_MARK_RANGE_FUNCTOR_CAN_COPY_ARGUMENTS

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FUNCTIONAL_ADDRESS_STABILITY_H
