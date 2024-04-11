//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// <cuda/std/array>

#include <cuda/std/array>

#include "MoveOnly.h"
#include "test_macros.h"

// expected-warning@array:* 0-1 {{suggest braces around initialization of subobject}}

int main(int, char**)
{
  {
    char source[3][6] = {"hi", "world"};
    // expected-error@array:* {{to_array does not accept multidimensional arrays}}
    // expected-error@array:* {{to_array requires copy constructible elements}}
    // expected-error@array:* 3 {{cannot initialize}}
    cuda::std::to_array(source); // expected-note {{requested here}}
  }

  {
    MoveOnly mo[] = {MoveOnly{3}};
    // expected-error@array:* {{to_array requires copy constructible elements}}
    // expected-error-re@array:* {{{{(call to implicitly-deleted copy constructor of 'MoveOnly')|(call to deleted
    // constructor of 'MoveOnly')}}}}
    cuda::std::to_array(mo); // expected-note {{requested here}}
  }

  {
    const MoveOnly cmo[] = {MoveOnly{3}};
    // expected-error@array:* {{to_array requires move constructible elements}}
    // expected-error-re@array:* {{{{(call to implicitly-deleted copy constructor of 'MoveOnly')|(call to deleted
    // constructor of 'MoveOnly')}}}}
    cuda::std::to_array(cuda::std::move(cmo)); // expected-note {{requested here}}
  }

  return 0;
}
