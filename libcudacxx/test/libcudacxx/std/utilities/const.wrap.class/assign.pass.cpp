//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// todo(dabayer): nvrtc doesn't support non-trivial types as static data members without -default-device, fails with:
//   A class static data member with non-const type is considered a host variable, and host variables are not allowed in
//   JIT mode. Consider using -default-device flag to process such data members as __device__ variables in JIT mode

// nvcc < 13.0 fails to compile this test due to:
//   lvalue required as left operand of assignment
// UNSUPPORTED: nvcc-12

// constant_wrapper

// template<constexpr-param R>
//   constexpr auto operator=(R) const noexcept
//     -> constant_wrapper<value = R::value> { return {}; }

#include <cuda/std/concepts>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "helpers.h"
#include "test_macros.h"

struct WithOps
{
  int value;

  TEST_FUNC constexpr WithOps(int v)
      : value(v)
  {}

  TEST_FUNC constexpr WithOps operator=(int i) const
  {
    return WithOps{value + i};
  }
};

struct OpsReturnNonStructural
{
  int value;

  TEST_FUNC constexpr OpsReturnNonStructural(int v)
      : value(v)
  {}

  TEST_FUNC constexpr NonStructural operator=(int i) const
  {
    return NonStructural{value + i};
  }
};

template <class T, class R, class = void>
inline constexpr bool HasAssign = false;
template <class T, class R>
inline constexpr bool
  HasAssign<T, R, cuda::std::void_t<decltype(cuda::std::declval<const T&>() = cuda::std::declval<R&>())>> = true;

template <class T, class R, class = void>
inline constexpr bool HasNoexceptAssign = false;
template <class T, class R>
inline constexpr bool
  HasNoexceptAssign<T, R, cuda::std::enable_if_t<noexcept(cuda::std::declval<const T&>() = cuda::std::declval<R&>())>> =
    true;

static_assert(!HasAssign<cuda::std::__constant_wrapper<5>, cuda::std::__constant_wrapper<3>>);
static_assert(!HasNoexceptAssign<cuda::std::__constant_wrapper<5>, cuda::std::__constant_wrapper<3>>);

#if TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)
static_assert(HasAssign<cuda::std::__constant_wrapper<WithOps{5}>, cuda::std::__constant_wrapper<3>>);
static_assert(HasNoexceptAssign<cuda::std::__constant_wrapper<WithOps{5}>, cuda::std::__constant_wrapper<3>>);

static_assert(!HasAssign<cuda::std::__constant_wrapper<OpsReturnNonStructural{5}>, cuda::std::__constant_wrapper<5>>);
#endif // TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)

TEST_FUNC constexpr bool test()
{
#if TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)
// nvcc == 13.0 produces invalid source file for the host compilers. It replaces contexpr variables with their values
// which doesn't work for assignment.
#  if !(_CCCL_CUDA_COMPILER(NVCC, ==, 13, 0) && _CCCL_HOST_COMPILATION())
  {
    // WithOps assignment
    const cuda::std::__constant_wrapper<WithOps{5}> cwOps5;
    cuda::std::__constant_wrapper<3> cw3;

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{8}>> decltype(auto) result = cwOps5 = cw3;
    static_assert(result.__get().value == 8);
  }

  {
    // with integral_constant
    const cuda::std::__constant_wrapper<WithOps{5}> cwOps5;
    cuda::std::integral_constant<int, 3> ic3;

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{8}>> decltype(auto) result = cwOps5 = ic3;
    static_assert(result.__get().value == 8);
  }
#  endif // !(_CCCL_CUDA_COMPILER(NVCC, ==, 13, 0) && _CCCL_HOST_COMPILATION())
#endif // TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
