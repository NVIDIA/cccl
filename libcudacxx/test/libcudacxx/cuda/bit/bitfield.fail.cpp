//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using T = uint32_t;
  static_assert(cuda::bitfield_insert(T{0}, T{0}, -1, 1));
  static_assert(cuda::bitfield_insert(T{0}, T{0}, 0, -1));
  static_assert(cuda::bitfield_insert(T{0}, T{0}, 0, 33));
  static_assert(cuda::bitfield_insert(T{0}, T{0}, 32, 1));
  static_assert(cuda::bitfield_insert(T{0}, T{0}, 20, 20));

  static_assert(cuda::bitfield_extract(T{0}, -1, 1));
  static_assert(cuda::bitfield_extract(T{0}, 0, -1));
  static_assert(cuda::bitfield_extract(T{0}, 0, 33));
  static_assert(cuda::bitfield_extract(T{0}, 32, 1));
  static_assert(cuda::bitfield_extract(T{0}, 20, 20));
  return 0;
}
