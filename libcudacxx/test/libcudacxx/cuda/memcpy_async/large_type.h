//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

struct large_type
{
  constexpr static int size = 32;

  __host__ __device__ large_type(int val = 0)
  {
    for (cuda::std::size_t i = 0; i < size; ++i)
    {
      storage[i] = val;
    }
  }

  large_type(const large_type&)            = default;
  large_type& operator=(const large_type&) = default;

  __host__ __device__ friend bool operator==(const large_type& lhs, const large_type& rhs)
  {
    for (cuda::std::size_t i = 0; i < size; ++i)
    {
      if (lhs.storage[i] != rhs.storage[i])
      {
        return false;
      }
    }

    return true;
  }

  int storage[size];
};
