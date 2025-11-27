// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#pragma once

struct Value
{
  int mValue;
  __host__ __device__ inline Value(int value);
  __host__ __device__ inline int value() const;
};

__host__ __device__ inline Value::Value(int value)
    : mValue(value)
{}

__host__ __device__ inline int Value::value() const
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

__host__ __device__ inline Stages stages(int value)
{
  return Stages{value};
}

__host__ __device__ inline Elems elems(int value)
{
  return Elems{value};
}

__host__ __device__ inline Warps warps(int value)
{
  return Warps{value};
}

__host__ __device__ inline Align align(int value)
{
  return Align{value};
}
