//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_CONTAINERS_VIEWS_MDSPAN_CONVERTIBLE_TO_INTEGRAL_H
#define TEST_STD_CONTAINERS_VIEWS_MDSPAN_CONVERTIBLE_TO_INTEGRAL_H

#include "CommonHelpers.h"
#include "test_macros.h"

struct IntType
{
  int val;
  IntType() = default;
  TEST_FUNC constexpr IntType(int v) noexcept
      : val(v){};

  TEST_FUNC constexpr bool operator==(const IntType& rhs) const
  {
    return val == rhs.val;
  }
#if TEST_STD_VER < 2020
  TEST_FUNC constexpr bool operator!=(const IntType& rhs) const
  {
    return val != rhs.val;
  }
#endif // TEST_STD_VER < 2020
  TEST_FUNC constexpr operator int() const noexcept
  {
    return val;
  }
  TEST_FUNC constexpr operator unsigned char() const
  {
    return static_cast<unsigned char>(val);
  }
  TEST_FUNC constexpr operator char() const noexcept
  {
    return static_cast<char>(val);
  }
};

// only non-const convertible
struct IntTypeNC
{
  int val;
  IntTypeNC() = default;
  TEST_FUNC constexpr IntTypeNC(int v) noexcept
      : val(v){};

  TEST_FUNC constexpr bool operator==(const IntType& rhs) const
  {
    return val == rhs.val;
  }
#if TEST_STD_VER < 2020
  TEST_FUNC constexpr bool operator!=(const IntType& rhs) const
  {
    return val != rhs.val;
  }
#endif // TEST_STD_VER < 2020
  TEST_FUNC constexpr operator int() noexcept
  {
    return val;
  }
  TEST_FUNC constexpr operator unsigned()
  {
    return static_cast<unsigned>(val);
  }
  TEST_FUNC constexpr operator char() noexcept
  {
    return static_cast<char>(val);
  }
};

// weird configurability of convertibility to int
template <bool conv_c, bool conv_nc, bool ctor_nt_c, bool ctor_nt_nc>
struct IntConfig
{
  int val;
  TEST_FUNC constexpr explicit IntConfig(int val_)
      : val(val_)
  {}
  template <bool Convertible = conv_nc, cuda::std::enable_if_t<Convertible, int> = 0>
  TEST_FUNC constexpr operator int() noexcept(ctor_nt_nc)
  {
    return val;
  }
  template <bool Convertible = conv_c, cuda::std::enable_if_t<Convertible, int> = 0>
  TEST_FUNC constexpr operator int() const noexcept(ctor_nt_c)
  {
    return val;
  }
};

#endif // TEST_STD_CONTAINERS_VIEWS_MDSPAN_CONVERTIBLE_TO_INTEGRAL_H
