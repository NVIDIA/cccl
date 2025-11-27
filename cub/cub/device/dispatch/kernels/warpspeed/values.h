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

struct Value
{
  int mValue;
  _CCCL_API Value(int value);
  _CCCL_API int value() const;
};

_CCCL_API Value::Value(int value)
    : mValue(value)
{}

_CCCL_API int Value::value() const
{
  return mValue;
}

struct Stages : Value
{};
struct Elems : Value
{};
struct Warps : Value
{};
struct Align : Value
{};

_CCCL_API Stages stages(int value)
{
  return Stages{value};
}

_CCCL_API Elems elems(int value)
{
  return Elems{value};
}

_CCCL_API Warps warps(int value)
{
  return Warps{value};
}

_CCCL_API Align align(int value)
{
  return Align{value};
}
