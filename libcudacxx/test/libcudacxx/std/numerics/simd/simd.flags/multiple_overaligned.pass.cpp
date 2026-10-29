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
// SIMD host/device functions are not supported in tile mode.

// [simd.flags.oper], repeated and combined overalignment flags

#include <cuda/std/__simd_>
#include <cuda/std/cstddef>

#include "test_macros.h"

namespace simd = cuda::std::simd;

template <cuda::std::size_t Alignment, class... Flags>
TEST_HOST_DEVICE_FUNC constexpr bool has_alignment(simd::flags<Flags...>)
{
  return simd::__overaligned_alignment_v<Flags...> == Alignment;
}

TEST_HOST_DEVICE_FUNC void test()
{
  static_assert(has_alignment<0>(simd::flag_default));
  static_assert(has_alignment<0>(simd::flag_convert | simd::flag_aligned));
  static_assert(has_alignment<8>(simd::flag_overaligned<8> | simd::flag_overaligned<8>));
  static_assert(has_alignment<32>(simd::flag_overaligned<8> | simd::flag_overaligned<32>));
  static_assert(has_alignment<32>(simd::flag_overaligned<32> | simd::flag_overaligned<8>));

  constexpr auto mixed_flags = simd::flag_convert | simd::flag_overaligned<8> | simd::flag_aligned
                             | simd::flag_overaligned<32> | simd::flag_convert | simd::flag_aligned;
  static_assert(has_alignment<32>(mixed_flags));
}

int main(int, char**)
{
  test();
  return 0;
}
