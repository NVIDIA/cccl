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

#include <cuda/std/__linalg/conjugated.h>
#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/type_traits>

#include "cuda/std/__type_traits/decay.h"

__host__ __device__ void constexpr_test()
{
  // operator() arithmetic type
  {
    using element_t = float;
    cuda::std::array d{42.f, 43.f};
    cuda::std::mdspan md(d.data(), 2);
    auto conj_md = cuda::std::linalg::conjugated(md);

    static_assert(cuda::std::is_same<decltype(+conj_md(0)), element_t>::value, "wrong type");
    static_cast<void>(conj_md);
  }
  // operator() complex type
  {
    using complex_t = cuda::std::complex<float>;
    cuda::std::array d{complex_t{}, complex_t{}};
    cuda::std::mdspan md(d.data(), 2);
    auto conj_md = cuda::std::linalg::conjugated(md);

    static_assert(cuda::std::is_same<decltype(conj_md(0)), complex_t>::value, "wrong type");
    static_cast<void>(conj_md);
  }
  // nested_accessor() arithmetic type
  {
    using element_t = int;
    cuda::std::array d{42, 43};
    cuda::std::mdspan md(d.data(), 2);
    auto conj_md = cuda::std::linalg::conjugated(md);

    static_assert(cuda::std::is_same<cuda::std::decay_t<decltype(conj_md.accessor())>,
                                     cuda::std::default_accessor<element_t>>::value,
                  "wrong type");
    static_cast<void>(conj_md);
  }
  // nested_accessor() complex type
  {
    using complex_t = cuda::std::complex<float>;
    cuda::std::array d{complex_t{}, complex_t{}};
    cuda::std::mdspan md(d.data(), 2);
    auto conj_md = cuda::std::linalg::conjugated(md);

    static_assert(cuda::std::is_same<cuda::std::decay_t<decltype(conj_md.accessor().nested_accessor())>,
                                     cuda::std::default_accessor<complex_t>>::value,
                  "wrong type");
    static_cast<void>(conj_md);
  }
}

__host__ __device__ void runtime_test()
{
  // operator() float value
  {
    cuda::std::array d{42.f, 43.f};
    cuda::std::mdspan md(d.data(), 2);
    auto conj_md = cuda::std::linalg::conjugated(md);

    assert(conj_md(0) == 42.f);
    assert(conj_md(1) == 43.f);
  }
  // operator() complex value
  {
    using complex_t = cuda::std::complex<float>;
    cuda::std::array d{complex_t{42.f, 2.f}, complex_t{43.f, 3.f}};
    cuda::std::mdspan md(d.data(), 2);
    auto conj_md = cuda::std::linalg::conjugated(md);

    assert((conj_md(0) == complex_t{42.f, -2.f}));
    assert((conj_md(1) == complex_t{43.f, -3.f}));
  }
  // operator() integer value
  {
    cuda::std::array d{42, 43};
    cuda::std::mdspan md(d.data(), 2);
    auto conj_md = cuda::std::linalg::conjugated(md);

    assert(conj_md(0) == 42);
    assert(conj_md(1) == 43);
  }
  // operator() custom type
  {
    struct A
    {
      int x;
    };
    cuda::std::array d{A{42}, A{43}};
    cuda::std::mdspan md(d.data(), 2);
    auto conj_md = cuda::std::linalg::conjugated(md);

    assert(conj_md(0).x == 42);
    assert(conj_md(1).x == 43);
  }
  // access()
  {
    cuda::std::array d{42.f, 43.f};
    cuda::std::mdspan md(d.data(), 2);
    auto conj_md = cuda::std::linalg::conjugated(md);

    assert(conj_md.accessor().access(d.data(), 1) == 43.f);
  }
  // offset()
  {
    cuda::std::array d{42.f, 43.f};
    cuda::std::mdspan md(d.data(), 2);
    auto conj_md = cuda::std::linalg::conjugated(md);

    assert(conj_md.accessor().offset(d.data(), 1) == d.data() + 1);
  }
  // composition
  {
    using complex_t = cuda::std::complex<float>;
    cuda::std::array d{complex_t{42.f, 2.f}, complex_t{43.f, 3.f}};
    cuda::std::mdspan md(d.data(), 2);
    auto conj_md1 = cuda::std::linalg::conjugated(md);
    auto conj_md2 = cuda::std::linalg::conjugated(conj_md1);

    assert((conj_md2(0) == complex_t{42.f, 2.f}));
    assert((conj_md2(1) == complex_t{43.f, 3.f}));
  }
  // copy constructor
  {
    cuda::std::array d{42.f, 43.f};
    cuda::std::mdspan md(d.data(), 2);
    auto conj_md1 = cuda::std::linalg::conjugated(md);
    auto conj_md2 = conj_md1;

    assert(conj_md2(0) == 42.f);
    assert(conj_md2(1) == 43.f);
  }
}

int main(int, char**)
{
  constexpr_test();
  runtime_test();
  return 0;
}
