//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/type_traits>
#include <cuda/vector_types>

int main(int, char**)
{
  [[maybe_unused]] cuda::vector_type_t<char, 1> v1; // char type is not a valid vector type
  [[maybe_unused]] cuda::vector_type_t<signed char, 8> v2; // 8 is not a valid vector size
  return 0;
}
