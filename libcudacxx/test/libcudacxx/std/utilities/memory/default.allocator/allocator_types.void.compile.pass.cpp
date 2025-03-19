//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Check that the nested types of cuda::std::allocator<void> are provided.
// After C++17, those are not provided in the primary template and the
// explicit specialization doesn't exist anymore, so this test is moot.

// REQUIRES: c++03 || c++11 || c++14 || c++17

// template <>
// class allocator<void>
// {
// public:
//     typedef void*                                 pointer;
//     typedef const void*                           const_pointer;
//     typedef void                                  value_type;
//
//     template <class _Up> struct rebind {typedef allocator<_Up> other;};
// };

// ADDITIONAL_COMPILE_DEFINITIONS: _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda/std/__memory_>
#include <cuda/std/type_traits>

static_assert((cuda::std::is_same<cuda::std::allocator<void>::pointer, void*>::value), "");
static_assert((cuda::std::is_same<cuda::std::allocator<void>::const_pointer, const void*>::value), "");
static_assert((cuda::std::is_same<cuda::std::allocator<void>::value_type, void>::value), "");
static_assert((cuda::std::is_same<cuda::std::allocator<void>::rebind<int>::other, cuda::std::allocator<int>>::value),
              "");

int main(int, char**)
{
  return 0;
}
