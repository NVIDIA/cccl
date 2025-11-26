/***************************************************************************************************
 * Copyright (c) 2011-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
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
