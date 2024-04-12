//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include <utility>

#include <nv/target>

template <class T1, class U1>
void test_conversion()
{
  const ::cuda::std::pair<T1, U1> input{T1{42}, U1{1337}};
  const ::std::pair<T1, U1> from_input{input};

  assert(from_input.first == T1{42});
  assert(from_input.second == U1{1337});
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test_conversion<int, float>();));

  return 0;
}
