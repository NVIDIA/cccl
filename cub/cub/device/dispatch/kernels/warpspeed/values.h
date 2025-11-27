// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

CUB_NAMESPACE_BEGIN

struct Value
{
  int mValue;
  _CCCL_API constexpr Value(int value) noexcept
      : mValue(value)
  {}

  [[nodiscard]] _CCCL_API constexpr int value() const
  {
    return mValue;
  }
};

struct Stages : Value
{};
struct Elems : Value
{};
struct Warps : Value
{};
struct Align : Value
{};

[[nodiscard]] _CCCL_API Stages stages(int value) noexcept
{
  return Stages{value};
}

[[nodiscard]] _CCCL_API Elems elems(int value) noexcept
{
  return Elems{value};
}

[[nodiscard]] _CCCL_API Warps warps(int value) noexcept
{
  return Warps{value};
}

[[nodiscard]] _CCCL_API Align align(int value) noexcept
{
  return Align{value};
}

CUB_NAMESPACE_END
