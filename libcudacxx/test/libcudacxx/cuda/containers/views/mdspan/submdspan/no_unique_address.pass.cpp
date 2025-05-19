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

template <size_t ExpectedSize, class OffsetType, class ExtentType, class StrideType>
__host__ __device__ void test(cuda::std::strided_slice<OffsetType, ExtentType, StrideType> slice, size_t expected_size)
{
  using strided_slice = cuda::std::strided_slice<OffsetType, ExtentType, StrideType>;
  assert(slice.offset == 42);
  assert(slice.extent == 1337);
  assert(slice.stride == 7);
  assert(sizeof(strided_slice) == expected_size);
  static_assert(sizeof(strided_slice) == ExpectedSize, "Size mismatch");
}

template <size_t ExpectedSize, class OffsetType, class ExtentType, class StrideType>
__global__ void test_kernel(cuda::std::strided_slice<OffsetType, ExtentType, StrideType> slice, size_t expected_size)
{
  test<ExpectedSize>(slice, expected_size);
}

void test()
{
  { // all non_empty
    using strided_slice = cuda::std::strided_slice<int, int, int>;
    strided_slice slice{42, 1337, 7};
    test<sizeof(strided_slice)>(slice, sizeof(strided_slice));
    test_kernel<sizeof(strided_slice)><<<1, 1>>>(slice, sizeof(strided_slice));
  }

  { // OffsetType empty
    using strided_slice = cuda::std::strided_slice<cuda::std::integral_constant<int, 42>, int, int>;
    strided_slice slice{{}, 1337, 7};
    test<sizeof(strided_slice)>(slice, sizeof(strided_slice));
    test_kernel<sizeof(strided_slice)><<<1, 1>>>(slice, sizeof(strided_slice));
  }

  { // ExtentType empty
    using strided_slice = cuda::std::strided_slice<int, cuda::std::integral_constant<int, 1337>, int>;
    strided_slice slice{42, {}, 7};
    test<sizeof(strided_slice)>(slice, sizeof(strided_slice));
    test_kernel<sizeof(strided_slice)><<<1, 1>>>(slice, sizeof(strided_slice));
  }

  { // StrideType empty
    using strided_slice = cuda::std::strided_slice<int, int, cuda::std::integral_constant<int, 7>>;
    strided_slice slice{42, 1337, {}};
    test<sizeof(strided_slice)>(slice, sizeof(strided_slice));
    test_kernel<sizeof(strided_slice)><<<1, 1>>>(slice, sizeof(strided_slice));
  }

  { // OffsetType + StrideType empty
    using strided_slice =
      cuda::std::strided_slice<cuda::std::integral_constant<int, 42>, int, cuda::std::integral_constant<int, 7>>;
    strided_slice slice{{}, 1337, {}};
    test<sizeof(strided_slice)>(slice, sizeof(strided_slice));
    test_kernel<sizeof(strided_slice)><<<1, 1>>>(slice, sizeof(strided_slice));
  }

  { // All empty
    using strided_slice =
      cuda::std::strided_slice<cuda::std::integral_constant<int, 42>,
                               cuda::std::integral_constant<int, 1337>,
                               cuda::std::integral_constant<int, 7>>;
    strided_slice slice{};
    test<sizeof(strided_slice)>(slice, sizeof(strided_slice));
    // cannot call a kernel with an Empty parameter type
    // test_kernel<sizeof(strided_slice)><<<1, 1>>>(slice, sizeof(strided_slice));
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
