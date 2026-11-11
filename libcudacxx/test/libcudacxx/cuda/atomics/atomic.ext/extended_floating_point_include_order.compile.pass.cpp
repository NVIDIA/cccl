//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// The atomic backends use inline PTX.

// clang-format off
#include <disable_nvfp_conversions_and_operators.h>
// clang-format on

#include "test_macros.h"

// Keep fp16 declared before the atomic helpers and bf16 declared after them.
#if _CCCL_HAS_NVFP16()
#  include <cuda_fp16.h>
#endif // _CCCL_HAS_NVFP16()

#include <cuda/atomic>

#if _CCCL_HAS_NVBF16()
#  include <cuda_bf16.h>
#endif // _CCCL_HAS_NVBF16()

template <class Atomic, class T>
TEST_HOST_DEVICE_FUNC void test_operations(Atomic& atom, T value)
{
  (void) atom.fetch_add(value, cuda::std::memory_order_relaxed);
  (void) atom.fetch_sub(value, cuda::std::memory_order_relaxed);
  (void) (atom += value);
  (void) (atom -= value);
}

template <class T>
TEST_HOST_DEVICE_FUNC void test_type()
{
  T value{};
  volatile T volatile_value{};
  cuda::atomic<T> owned{value};
  volatile cuda::atomic<T> volatile_owned{value};
  cuda::atomic_ref<T> ref{value};
  cuda::atomic_ref<volatile T> volatile_ref{volatile_value};
  cuda::std::atomic<T> std_owned{value};
  volatile cuda::std::atomic<T> volatile_std_owned{value};
  cuda::std::atomic_ref<T> std_ref{value};
  cuda::std::atomic_ref<volatile T> volatile_std_ref{volatile_value};
  test_operations(owned, value);
  test_operations(volatile_owned, value);
  test_operations(ref, value);
  test_operations(volatile_ref, value);
  test_operations(std_owned, value);
  test_operations(volatile_std_owned, value);
  test_operations(std_ref, value);
  test_operations(volatile_std_ref, value);
}

int main(int, char**)
{
#if _CCCL_HAS_NVFP16()
  test_type<__half>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_type<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16()
  return 0;
}
