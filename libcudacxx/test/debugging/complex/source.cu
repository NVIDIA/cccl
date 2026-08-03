// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/__complex_>
#include <cuda/std/array>
#include <cuda/std/complex>

template <class T>
[[gnu::noinline]] void keep_for_debugger(const T& value)
{
  asm volatile("" : : "g"(&value) : "memory");
}

using complex_alias = cuda::std::complex<float>;

[[gnu::noinline]] void inspect_std_default(const cuda::std::complex<float>& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_std_double(const cuda::std::complex<double>& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_cuda_float(const cuda::complex<float>& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_alias(const complex_alias& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_nested(const cuda::std::array<cuda::std::complex<float>, 2>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_before_update(const cuda::std::complex<double>& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_after_update(const cuda::std::complex<double>& value)
{
  keep_for_debugger(value);
}

int main()
{
  const cuda::std::complex<float> std_default{};
  const cuda::std::complex<double> std_double{1.5, -2.25};
  const cuda::complex<float> cuda_float{0.5f, 42.5f};
  const complex_alias alias{-3.5f, 0.25f};
  const cuda::std::array<cuda::std::complex<float>, 2> nested = {{{13.5f, -5.5f}, {0.25f, 88.5f}}};
  cuda::std::complex<double> updated{6.5, -91.5};

  inspect_std_default(std_default);
  inspect_std_double(std_double);
  inspect_cuda_float(cuda_float);
  inspect_alias(alias);
  inspect_nested(nested);
  inspect_before_update(updated);
  updated = {3.25, 85.5};
  inspect_after_update(updated);
}
