//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/std/functional>
// #include <cuda/std/string>

template <class T>
struct is_transparent
{
private:
  struct two
  {
    char lx;
    char lxx;
  };
  template <class U>
  __host__ __device__ static two test(...);
  template <class U>
  __host__ __device__ static char test(typename U::is_transparent* = 0);

public:
  static const bool value = sizeof(test<T>(0)) == 1;
};

int main(int, char**)
{
  static_assert(!is_transparent<cuda::std::logical_and<int>>::value, "");
  // static_assert ( !is_transparent<cuda::std::logical_and<cuda::std::string>>::value, "" );
  static_assert(is_transparent<cuda::std::logical_and<void>>::value, "");
  static_assert(is_transparent<cuda::std::logical_and<>>::value, "");

  static_assert(!is_transparent<cuda::std::logical_or<int>>::value, "");
  // static_assert ( !is_transparent<cuda::std::logical_or<cuda::std::string>>::value, "" );
  static_assert(is_transparent<cuda::std::logical_or<void>>::value, "");
  static_assert(is_transparent<cuda::std::logical_or<>>::value, "");

  static_assert(!is_transparent<cuda::std::logical_not<int>>::value, "");
  // static_assert ( !is_transparent<cuda::std::logical_not<cuda::std::string>>::value, "" );
  static_assert(is_transparent<cuda::std::logical_not<void>>::value, "");
  static_assert(is_transparent<cuda::std::logical_not<>>::value, "");

  return 0;
}
