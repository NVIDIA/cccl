//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_CONTAINERS_VIEWS_MDSPAN_SUBMDSPAN_HELPER_H
#define TEST_STD_CONTAINERS_VIEWS_MDSPAN_SUBMDSPAN_HELPER_H

#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

_CCCL_TEMPLATE(class MDSpan)
_CCCL_REQUIRES((MDSpan::rank() == 0))
__host__ __device__ constexpr bool equal_to(const MDSpan& mdspan, const char* expected)
{
  return mdspan[cuda::std::array<size_t, 0>{}] == expected[0];
}

_CCCL_TEMPLATE(class MDSpan)
_CCCL_REQUIRES((MDSpan::rank() == 1))
__host__ __device__ constexpr bool equal_to(const MDSpan& mdspan, const char* expected)
{
  for (size_t i = 0; i != mdspan.size(); ++i)
  {
    if (mdspan[i] != expected[i])
    {
      return false;
    }
  }
  return true;
}

_CCCL_TEMPLATE(class MDSpan)
_CCCL_REQUIRES((MDSpan::rank() == 2))
__host__ __device__ constexpr bool equal_to(const MDSpan& mdspan, cuda::std::array<const char*, 2> expected)
{
  for (size_t i = 0; i != mdspan.extent(0); ++i)
  {
    for (size_t j = 0; j != mdspan.extent(1); ++j)
    {
      const cuda::std::array<size_t, 2> indices{i, j};
      if (mdspan[indices] != expected[i][j])
      {
        return false;
      }
    }
  }
  return true;
}

#endif // TEST_STD_CONTAINERS_VIEWS_MDSPAN_SUBMDSPAN_HELPER_H
