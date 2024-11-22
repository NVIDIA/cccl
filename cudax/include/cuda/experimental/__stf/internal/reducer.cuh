//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/execution_space.h>

#include <limits>
#include <algorithm> // for std::min and std::max


namespace cuda::experimental::stf::reducer
{

template <typename T>
class sum
{
public:
  static __host__ __device__ void init_op(T& dst)
  { 
    dst = static_cast<T>(0);
  } 
  
  static __host__ __device__ void apply_op(T& dst, const T& src)
  {
    dst += src;
  }
};

template <typename T>
class maxval
{
public:
  static __host__ __device__ void init_op(T& dst)
  {
    dst = ::std::numeric_limits<T>::lowest();
  }

  static __host__ __device__ void apply_op(T& dst, const T& src)
  {
    dst = ::std::max(dst, src);
  }
};

template <typename T>
class minval
{
public:
  static __host__ __device__ void init_op(T& dst)
  {
    dst = ::std::numeric_limits<T>::max();
  }

  static __host__ __device__ void apply_op(T& dst, const T& src)
  {
    dst = ::std::min(dst, src);
  }
};

template <typename T>
class product
{
public:
  static __host__ __device__ void init_op(T& dst)
  {
    dst = static_cast<T>(1);
  }

  static __host__ __device__ void apply_op(T& dst, const T& src)
  {
    dst *= src;
  }
};

template <typename T>
class bitwise_and
{
public:
  static __host__ __device__ void init_op(T& dst)
  {
    dst = ~static_cast<T>(0);
  }

  static __host__ __device__ void apply_op(T& dst, const T& src)
  {
    dst &= src;
  }
};

template <typename T>
class bitwise_or
{
public:
  static __host__ __device__ void init_op(T& dst)
  {
    dst = static_cast<T>(0);
  }

  static __host__ __device__ void apply_op(T& dst, const T& src)
  {
    dst |= src;
  }
};

template <typename T>
class bitwise_xor
{
public:
  static __host__ __device__ void init_op(T& dst)
  {
    dst = static_cast<T>(0);
  }

  static __host__ __device__ void apply_op(T& dst, const T& src)
  {
    dst ^= src;
  }
};

template <typename T>
class logical_and
{
public:
  static __host__ __device__ void init_op(T& dst)
  {
    dst = true; // Logical AND identity
  }

  static __host__ __device__ void apply_op(T& dst, const T& src)
  {
    dst = dst && src;
  }
};

template <typename T>
class logical_or
{
public:
  static __host__ __device__ void init_op(T& dst)
  {
    dst = false; // Logical OR identity
  }

  static __host__ __device__ void apply_op(T& dst, const T& src)
  {
    dst = dst || src;
  }
};

} // end namespace cuda::experimental::stf::reducer
