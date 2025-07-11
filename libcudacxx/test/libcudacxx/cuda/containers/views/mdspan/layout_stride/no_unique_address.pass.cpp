//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <cuda/std/cassert>
#include <cuda/std/mdspan>

#include "test_macros.h"

template <class Mapping>
__host__ __device__ void test([[maybe_unused]] Mapping map, size_t expected_size)
{
  using extents = typename Mapping::extents_type;
  if constexpr (extents::rank() > 0)
  {
    assert(map.extents().extent(0) == 42);
    assert(map.extents().extent(1) == 1337);
    assert(map.extents().extent(2) == 7);

    assert(map.strides()[0] == 1);
    assert(map.strides()[1] == map.extents().extent(0) * map.strides()[0]);
    assert(map.strides()[2] == map.extents().extent(1) * map.strides()[1]);
  }
  assert(sizeof(Mapping) == expected_size);
}

template <class Mapping>
__global__ void test_kernel(Mapping map, size_t expected_size)
{
  test(map, expected_size);
}

template <class Extent>
__host__ __device__ constexpr cuda::std::array<int, 3> get_strides(Extent src_exts)
{
  cuda::std::array<int, 3> strides{};
  strides[0] = 1;
  for (size_t r = 1; r < 3; r++)
  {
    strides[r] = static_cast<int>(src_exts.extent(r - 1) * strides[r - 1]);
  }
  return strides;
}

void test()
{
  { // all dynamic
    using extents =
      cuda::std::extents<size_t, cuda::std::dynamic_extent, cuda::std::dynamic_extent, cuda::std::dynamic_extent>;
    using mapping = cuda::std::layout_stride::mapping<extents>;
    extents ext{42, 1337, 7};
    mapping map{ext, get_strides(ext)};
    test(map, sizeof(mapping));
    test_kernel<<<1, 1>>>(map, sizeof(mapping));
  }

  { // middle static
    using extents = cuda::std::extents<size_t, cuda::std::dynamic_extent, 1337, cuda::std::dynamic_extent>;
    using mapping = cuda::std::layout_stride::mapping<extents>;
    extents ext{42, 7};
    mapping map{ext, get_strides(ext)};
    test(map, sizeof(mapping));
    test_kernel<<<1, 1>>>(map, sizeof(mapping));
  }

  { // middle dynamic
    using extents = cuda::std::extents<size_t, 42, cuda::std::dynamic_extent, 7>;
    using mapping = cuda::std::layout_stride::mapping<extents>;
    extents ext{1337};
    mapping map{ext, get_strides(ext)};
    test(map, sizeof(mapping));
    test_kernel<<<1, 1>>>(map, sizeof(mapping));
  }

  { // all dynamic
    using extents = cuda::std::extents<size_t, 42, 1337, 7>;
    using mapping = cuda::std::layout_stride::mapping<extents>;
    extents ext{};
    mapping map{ext, get_strides(ext)};
    test(map, sizeof(mapping));
    test_kernel<<<1, 1>>>(map, sizeof(mapping));
  }

  { // zero rank
    using extents = cuda::std::extents<size_t>;
    using mapping = cuda::std::layout_stride::mapping<extents>;
    extents ext{};
    mapping map{ext, cuda::std::array<int, 0>{}};
    test(map, sizeof(mapping));
    test_kernel<<<1, 1>>>(map, sizeof(mapping));
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
