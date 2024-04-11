//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// Test that UDT's convertible to an integral or floating point type do not
// participate in overload resolution.

#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/type_traits>

template <class IntT>
struct UDT
{
  operator IntT() const
  {
    return 1;
  }
};

UDT<float> ft;
UDT<double> dt;
// CUDA treats long double as double
// UDT<long double> ldt;
UDT<int> it;
UDT<unsigned long> uit;

int main(int, char**)
{
  {
    cuda::std::real(ft); // expected-error {{no matching function}}
    cuda::std::real(dt); // expected-error {{no matching function}}
    // cuda::std::real(ldt); // expected-error {{no matching function}}
    cuda::std::real(it); // expected-error {{no matching function}}
    cuda::std::real(uit); // expected-error {{no matching function}}
  }
  {
    cuda::std::imag(ft); // expected-error {{no matching function}}
    cuda::std::imag(dt); // expected-error {{no matching function}}
    // cuda::std::imag(ldt); // expected-error {{no matching function}}
    cuda::std::imag(it); // expected-error {{no matching function}}
    cuda::std::imag(uit); // expected-error {{no matching function}}
  }
  {
    cuda::std::arg(ft); // expected-error {{no matching function}}
    cuda::std::arg(dt); // expected-error {{no matching function}}
    // cuda::std::arg(ldt); // expected-error {{no matching function}}
    cuda::std::arg(it); // expected-error {{no matching function}}
    cuda::std::arg(uit); // expected-error {{no matching function}}
  }
  {
    cuda::std::norm(ft); // expected-error {{no matching function}}
    cuda::std::norm(dt); // expected-error {{no matching function}}
    // cuda::std::norm(ldt); // expected-error {{no matching function}}
    cuda::std::norm(it); // expected-error {{no matching function}}
    cuda::std::norm(uit); // expected-error {{no matching function}}
  }
  {
    cuda::std::conj(ft); // expected-error {{no matching function}}
    cuda::std::conj(dt); // expected-error {{no matching function}}
    // cuda::std::conj(ldt); // expected-error {{no matching function}}
    cuda::std::conj(it); // expected-error {{no matching function}}
    cuda::std::conj(uit); // expected-error {{no matching function}}
  }
  {
    cuda::std::proj(ft); // expected-error {{no matching function}}
    cuda::std::proj(dt); // expected-error {{no matching function}}
    // cuda::std::proj(ldt); // expected-error {{no matching function}}
    cuda::std::proj(it); // expected-error {{no matching function}}
    cuda::std::proj(uit); // expected-error {{no matching function}}
  }

  return 0;
}
