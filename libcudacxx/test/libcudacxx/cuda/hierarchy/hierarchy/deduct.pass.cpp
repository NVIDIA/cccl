//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// todo: enable with nvrtc
// UNSUPPORTED: nvrtc

#include <cuda/hierarchy>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

template <class... Args>
constexpr void test_deduction_guide(const Args&... args)
{
  auto ref = cuda::make_hierarchy(args...);

  // 1. Test direct construction.
  {
    cuda::hierarchy hier{args...};
    static_assert(cuda::std::is_same_v<decltype(hier), decltype(ref)>);
    assert(hier == ref);
  }

  // 2. Test construction from a tuple.
  {
    cuda::hierarchy hier{cuda::std::tuple{args...}};
    static_assert(cuda::std::is_same_v<decltype(hier), decltype(ref)>);
    assert(hier == ref);
  }
}

template <class BU, class... Args>
constexpr void test_deduction_guide_with_bottom_unit(const BU& bu, const Args&... args)
{
  auto ref = cuda::make_hierarchy<BU>(args...);

  // 1. Test direct construction.
  {
    cuda::hierarchy hier{bu, args...};
    static_assert(cuda::std::is_same_v<decltype(hier), decltype(ref)>);
    assert(hier == ref);
  }

  // 2. Test construction from a tuple.
  {
    cuda::hierarchy hier{bu, cuda::std::tuple{args...}};
    static_assert(cuda::std::is_same_v<decltype(hier), decltype(ref)>);
    assert(hier == ref);
  }
}

void test()
{
  // 1. Test hierarchy with levels and default bottom unit
  test_deduction_guide(cuda::grid_dims<13>());
  test_deduction_guide(cuda::block_dims(dim3{2}));
  test_deduction_guide(cuda::cluster_dims(dim3{3, 4}), cuda::block_dims<2, 3>());

  // 2. Test hierarchy with levels and bottom unit
  test_deduction_guide_with_bottom_unit(cuda::block, cuda::grid_dims<13>());
  test_deduction_guide_with_bottom_unit(cuda::gpu_thread, cuda::block_dims(dim3{2}));
  test_deduction_guide_with_bottom_unit(cuda::gpu_thread, cuda::cluster_dims(dim3{3, 4}), cuda::block_dims<2, 3>());
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
