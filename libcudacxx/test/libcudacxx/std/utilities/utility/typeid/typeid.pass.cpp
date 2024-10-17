//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/utility>

// _CCCL_TYPEID(<type>)
// _CCCL_CONSTEXPR_TYPEID(<type>)

#include <cuda/std/__utility/typeid.h>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include <type_traits>

#include "test_macros.h"

struct a_dummy_class_type
{};

int main(int, char**)
{
#ifndef _CCCL_NO_TYPEID
  ASSERT_SAME_TYPE(::cuda::std::type_info, ::std::type_info);
  ASSERT_SAME_TYPE(decltype((_CCCL_TYPEID(int))), ::std::type_info const&);
  ASSERT_NOEXCEPT(_CCCL_TYPEID(int));
  static_assert(!std::is_default_constructible<::cuda::std::type_info>::value, "");
  static_assert(!std::is_copy_constructible<::cuda::std::type_info>::value, "");
  assert(_CCCL_TYPEID(int).name()[0] == 'i');
#else
  ASSERT_SAME_TYPE(::cuda::std::type_info, ::cuda::std::__type_info);
  ASSERT_SAME_TYPE(decltype((_CCCL_TYPEID(int))), ::cuda::std::__type_info const&);
  ASSERT_NOEXCEPT(_CCCL_TYPEID(int));
  static_assert(!std::is_default_constructible<::cuda::std::type_info>::value, "");
  static_assert(!std::is_copy_constructible<::cuda::std::type_info>::value, "");
  assert(_CCCL_TYPEID(int).name()[0] == 'i');
#endif

  ASSERT_SAME_TYPE(decltype((_CCCL_CONSTEXPR_TYPEID(int))), ::cuda::std::__type_info const&);
  ASSERT_NOEXCEPT(_CCCL_CONSTEXPR_TYPEID(int));
  assert(_CCCL_CONSTEXPR_TYPEID(int).name()[0] == 'i');
  assert(_CCCL_CONSTEXPR_TYPEID(int).name() == _CCCL_CONSTEXPR_TYPEID(int).__name_view().begin());

  assert(_CCCL_TYPEID(int) == _CCCL_TYPEID(int));
  assert(!(_CCCL_TYPEID(int) != _CCCL_TYPEID(int)));
  assert(_CCCL_TYPEID(int) != _CCCL_TYPEID(float));
  assert(!(_CCCL_TYPEID(int) == _CCCL_TYPEID(float)));
  assert(_CCCL_TYPEID(const int) == _CCCL_TYPEID(int));
  assert(_CCCL_TYPEID(int&) != _CCCL_TYPEID(int));

  static_assert(_CCCL_CONSTEXPR_TYPEID(int) == _CCCL_CONSTEXPR_TYPEID(int), "");
  static_assert(!(_CCCL_CONSTEXPR_TYPEID(int) != _CCCL_CONSTEXPR_TYPEID(int)), "");
  static_assert(_CCCL_CONSTEXPR_TYPEID(int) != _CCCL_CONSTEXPR_TYPEID(float), "");
  static_assert(!(_CCCL_CONSTEXPR_TYPEID(int) == _CCCL_CONSTEXPR_TYPEID(float)), "");
  static_assert(_CCCL_CONSTEXPR_TYPEID(const int) == _CCCL_CONSTEXPR_TYPEID(int), "");
  static_assert(_CCCL_CONSTEXPR_TYPEID(int&) != _CCCL_CONSTEXPR_TYPEID(int), "");

  static_assert(&_CCCL_CONSTEXPR_TYPEID(int) == &_CCCL_CONSTEXPR_TYPEID(int), "");
  static_assert(&_CCCL_CONSTEXPR_TYPEID(int) != &_CCCL_CONSTEXPR_TYPEID(float), "");

  static_assert(_CCCL_CONSTEXPR_TYPEID(float).before(_CCCL_CONSTEXPR_TYPEID(int)), "");
  static_assert(!_CCCL_CONSTEXPR_TYPEID(int).before(_CCCL_CONSTEXPR_TYPEID(int)), "");
  static_assert(!_CCCL_CONSTEXPR_TYPEID(int).before(_CCCL_CONSTEXPR_TYPEID(float)), "");

  static_assert(_CCCL_CONSTEXPR_TYPEID(int).__name_view() == ::cuda::std::__string_view("int"), "");
  static_assert(_CCCL_CONSTEXPR_TYPEID(float).__name_view() == ::cuda::std::__string_view("float"), "");
  static_assert(_CCCL_CONSTEXPR_TYPEID(a_dummy_class_type).__name_view().find("a_dummy_class_type") != -1, "");

  return 0;
}
