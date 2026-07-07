//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: enable-tile
// error: return in loop statement is not supported

#include <cuda/devices>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>

#include "test_macros.h"

template <cuda::std::size_t N>
TEST_FUNC constexpr auto make_all_ccs_ref(cuda::std::array<int, N> vs, int scale)
{
  cuda::std::array<cuda::compute_capability, N> ret;
  for (cuda::std::size_t i = 0; i < N; ++i)
  {
    ret[i] = cuda::compute_capability{vs[i] / scale};
  }
  return ret;
}

TEST_FUNC constexpr bool test()
{
  // 1. Test signature.
  static_assert(cuda::std::__is_cuda_std_array_v<decltype(cuda::__target_compute_capabilities())>);
  static_assert(noexcept(cuda::__target_compute_capabilities()));

  // 2. Test that all values are present.
  const auto all_ccs = cuda::__target_compute_capabilities();

#if defined(__CUDA_ARCH_LIST__)
  const auto all_ccs_ref = make_all_ccs_ref(cuda::std::array{__CUDA_ARCH_LIST__}, 10);
#elif defined(NV_TARGET_SM_INTEGER_LIST)
  const auto all_ccs_ref = make_all_ccs_ref(cuda::std::array{NV_TARGET_SM_INTEGER_LIST}, 1);
#else
  const auto all_ccs_ref = ::cuda::__all_compute_capabilities();
#endif

  assert(all_ccs.size() == all_ccs_ref.size());
  for (cuda::std::size_t i = 0; i < all_ccs.size(); ++i)
  {
    assert(all_ccs[i] == all_ccs_ref[i]);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
