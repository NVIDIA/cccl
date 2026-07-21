//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_SFINAE_TRAITS_H
#define _CUDA_STD___TYPE_TRAITS_SFINAE_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

//! @brief Determines whether a constructor is valid and whether it is implicit or explicit
enum class __select_constructor
{
  __invalid, //!< The constructor is not valid
  __implicit, //!< The constructor is valid and implicit
  __explicit, //!< The constructor is valid and explicit
  __deleted, //!< The constructor is marked as deleted
};

template <__select_constructor _Trait>
inline constexpr bool __can_construct_implicitly = _Trait == __select_constructor::__implicit;
template <__select_constructor _Trait>
inline constexpr bool __can_construct_explicitly = _Trait == __select_constructor::__explicit;
template <__select_constructor _Trait>
inline constexpr bool __can_construct =
  (_Trait == __select_constructor::__implicit) || (_Trait == __select_constructor::__explicit);
template <__select_constructor _Trait>
inline constexpr bool __is_deleted = _Trait == __select_constructor::__deleted;

//! @brief Determines whether an assignment is valid and also whether it does not throw
enum class __select_assignment
{
  __invalid, //!< The assignment is invalid
  __is_nothrow, //!< The assignment is valid and noexcept
  __may_throw, //!< The assignment is valid but may throw
};

template <__select_assignment _Trait>
inline constexpr bool __can_assign = _Trait != __select_assignment::__invalid;
template <__select_assignment _Trait>
inline constexpr bool __can_nothrow_assign = _Trait == __select_assignment::__is_nothrow;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_SFINAE_TRAITS_H
