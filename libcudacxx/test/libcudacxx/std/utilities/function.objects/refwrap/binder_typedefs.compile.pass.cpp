//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// REQUIRES: c++03 || c++11 || c++14 || c++17

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

// <functional>

// reference_wrapper

// check that binder typedefs exit

#define _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

struct UnaryFunction
{
  typedef long argument_type;
  typedef char result_type;
};

struct BinaryFunction
{
  typedef int first_argument_type;
  typedef char second_argument_type;
  typedef long result_type;
};

static_assert(cuda::std::is_same<cuda::std::reference_wrapper<int (UnaryFunction::*)()>::result_type, int>::value, "");
static_assert(
  cuda::std::is_same<cuda::std::reference_wrapper<int (UnaryFunction::*)()>::argument_type, UnaryFunction*>::value, "");

static_assert(cuda::std::is_same<cuda::std::reference_wrapper<int (BinaryFunction::*)(char)>::result_type, int>::value,
              "");
static_assert(cuda::std::is_same<cuda::std::reference_wrapper<int (BinaryFunction::*)(char)>::first_argument_type,
                                 BinaryFunction*>::value,
              "");
static_assert(
  cuda::std::is_same<cuda::std::reference_wrapper<int (BinaryFunction::*)(char)>::second_argument_type, char>::value,
  "");

static_assert(cuda::std::is_same<cuda::std::reference_wrapper<void (*)()>::result_type, void>::value, "");

int main(int, char**)
{
  return 0;
}
