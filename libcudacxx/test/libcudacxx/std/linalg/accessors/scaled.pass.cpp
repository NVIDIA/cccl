//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11, c++14
// UNSUPPORTED: msvc && c++17

#include <cuda/std/__linalg/scaled.h>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

__host__ __device__ void constexpr_test()
{
  // operator() type
  {
    using element_t = int;
    cuda::std::array<element_t, 2> d{42, 43};
    cuda::std::mdspan md(d.data(), 2);
    auto scaled_md = cuda::std::linalg::scaled(2.0f, md);

    static_assert(cuda::std::is_same<decltype(scaled_md(0)), float>::value, "wrong type");
    static_cast<void>(scaled_md);
  }
  // nested_accessor()
  {
    using element_t = int;
    cuda::std::array<element_t, 2> d{42, 43};
    cuda::std::mdspan md(d.data(), 2);
    auto scaled_md = cuda::std::linalg::scaled(2, md);

    static_assert(cuda::std::is_same<decltype(scaled_md.accessor().nested_accessor()),
                                     cuda::std::default_accessor<element_t>>::value,
                  "wrong type");
    static_cast<void>(scaled_md);
  }
}

__host__ __device__ void runtime_test()
{
  // operator() value
  {
    using element_t = int;
    cuda::std::array<element_t, 2> d{42, 43};
    cuda::std::mdspan md(d.data(), 2);
    auto scaled_md = cuda::std::linalg::scaled(2, md);

    assert(scaled_md(0) == 42 * 2);
    assert(scaled_md(1) == 43 * 2);
  }
  // access()
  {
    using element_t = int;
    cuda::std::array<element_t, 2> d{42, 43};
    cuda::std::mdspan md(d.data(), 2);
    auto scaled_md = cuda::std::linalg::scaled(2, md);

    assert(scaled_md.accessor().access(d.data(), 1) == 43 * 2);
  }
  // offset()
  {
    using element_t = int;
    cuda::std::array<element_t, 2> d{42, 43};
    cuda::std::mdspan md(d.data(), 2);
    auto scaled_md = cuda::std::linalg::scaled(2, md);

    assert(scaled_md.accessor().offset(d.data(), 1) == d.data() + 1);
  }
  // scaling_factor()
  {
    using element_t = int;
    cuda::std::array<element_t, 2> d{42, 43};
    cuda::std::mdspan md(d.data(), 2);
    auto scaled_md = cuda::std::linalg::scaled(2, md);

    assert(scaled_md.accessor().scaling_factor() == 2);
  }
  // composition
  {
    using element_t = int;
    cuda::std::array<element_t, 2> d{42, 43};
    cuda::std::mdspan md(d.data(), 2);
    auto scaled_md1 = cuda::std::linalg::scaled(2, md);
    auto scaled_md2 = cuda::std::linalg::scaled(3, scaled_md1);

    assert(scaled_md2(0) == 42 * 2 * 3);
    assert(scaled_md2(1) == 43 * 2 * 3);
  }
  // copy constructor
  {
    using element_t = int;
    cuda::std::array<element_t, 2> d{42, 43};
    cuda::std::mdspan md(d.data(), 2);
    auto scaled_md1 = cuda::std::linalg::scaled(2, md);
    auto scaled_md2 = scaled_md1;

    assert(scaled_md2(0) == 42 * 2);
    assert(scaled_md2(1) == 43 * 2);
  }
}

int main(int, char**)
{
  constexpr_test();
  runtime_test();
  return 0;
}
