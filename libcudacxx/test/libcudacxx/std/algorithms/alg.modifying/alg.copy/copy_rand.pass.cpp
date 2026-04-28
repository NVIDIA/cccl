//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// XFAIL: enable-tile
// nvbug6076227: ICE when validating tile MLIR

// <algorithm>

// template<InputIterator InIter, OutputIterator<auto, InIter::reference> OutIter>
//   constexpr OutIter   // constexpr after C++17
//   copy(InIter first, InIter last, OutIter result);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>

#include "copy_common.h"

TEST_CONSTEXPR_CXX20 TEST_FUNC bool test()
{
  test<random_access_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<random_access_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<random_access_iterator<const int*>, forward_iterator<int*>>();
  test<random_access_iterator<const int*>, bidirectional_iterator<int*>>();
  test<random_access_iterator<const int*>, random_access_iterator<int*>>();
  test<random_access_iterator<const int*>, int*>();

#if !TEST_COMPILER(NVRTC)
  NV_IF_TARGET(NV_IS_HOST, (test<const int*, host_only_iterator<int*>>();))
#endif // !TEST_COMPILER(NVRTC)
#if TEST_CUDA_COMPILATION()
  NV_IF_TARGET(NV_IS_DEVICE, (test<const int*, device_only_iterator<int*>>();))
#endif // TEST_CUDA_COMPILATION()

  return true;
}

int main(int, char**)
{
  test();

#if TEST_STD_VER >= 2020
  static_assert(test());
#endif // TEST_STD_VER >= 2020

  return 0;
}
