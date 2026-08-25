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
struct _ConstructorConstraint
{
  static constexpr bool __can_construct_implicitly = _Trait == __select_constructor::__implicit;
  static constexpr bool __can_construct_explicitly = _Trait == __select_constructor::__explicit;
  static constexpr bool __can_construct =
    _Trait == __select_constructor::__implicit || _Trait == __select_constructor::__explicit;
  static constexpr bool __is_deleted = _Trait == __select_constructor::__deleted;
};

template <bool _Trait>
struct _AssignmentConstraint
{
  static constexpr bool __can_assign = _Trait == true;
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_SFINAE_TRAITS_H
