//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11
// UNSUPPORTED: msvc && c++14, msvc && c++17

#include <cuda/std/cassert>
#include <cuda/std/mdspan>

template <size_t N, class Extents>
__host__ __device__ void test_mdspan_size(cuda::std::array<char, N>& storage, Extents&& e)
{
  using extents_type = cuda::std::remove_cv_t<cuda::std::remove_reference_t<Extents>>;
  cuda::std::mdspan<char, extents_type> m(storage.data(), cuda::std::forward<Extents>(e));

  static_assert(cuda::std::is_same<decltype(m.size()), size_t>::value,
                "The return type of mdspan::size() must be size_t.");

  // m.size() must not overflow, as long as the product of extents
  // is representable as a value of type size_t.
  assert(m.size() == N);
}

int main(int, char**)
{
  // TEST(TestMdspan, MdspanSizeReturnTypeAndPrecondition)
  {
    cuda::std::array<char, 12 * 11> storage;

    static_assert(cuda::std::numeric_limits<int8_t>::max() == 127, "max int8_t != 127");
    test_mdspan_size(storage, cuda::std::extents<int8_t, 12, 11>{}); // 12 * 11 == 132
  }

  {
    cuda::std::array<char, 16 * 17> storage;

    static_assert(cuda::std::numeric_limits<uint8_t>::max() == 255, "max uint8_t != 255");
    test_mdspan_size(storage, cuda::std::extents<uint8_t, 16, 17>{}); // 16 * 17 == 272
  }

  return 0;
}
