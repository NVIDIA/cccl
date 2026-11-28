// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// thrust::complex must be streamable without the host <complex> being included
#include <thrust/complex.h>

#include <sstream>

#include <unittest/unittest.h>

TEMPLATE_LIST_TEST_CASE("ComplexStreamWithoutHostComplex", "[complex]", FloatingPointTypes)
{
  using T = TestType;

  const thrust::complex<T> a(T(1), T(2));
  std::stringstream out;
  out << a;
  REQUIRE(out.str() == "(1,2)");
  thrust::complex<T> b{};
  out >> b;
  REQUIRE(b == a);

  std::istringstream bad("(1,2");
  thrust::complex<T> c{};
  bad >> c;
  REQUIRE(bad.fail());
}
