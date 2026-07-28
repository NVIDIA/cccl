//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// gcc-10 segfaults with any use of constant_wrapper, gcc-11 fails to evaluate:
//   typename decltype(__cw_fixed_value(_Xp))::type
// UNSUPPORTED: gcc-10 || gcc-11

// nvcc < 13.0 fails to compile this test due to:
//   lvalue required as left operand of assignment
// UNSUPPORTED: nvcc-12

// todo(dabayer): Find a way to make this work for nvrtc.
// nvrtc doesn't allow accessing the static constexpr const auto& value member.
// UNSUPPORTED: nvrtc

// REQUIRES: !c++17

// constant_wrapper

// template<constexpr-param R>
//   constexpr auto operator=(R) const noexcept
//     -> constant_wrapper<value = R::value> { return {}; }

#include <cuda/std/concepts>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "helpers.h"
#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(20094) // a host member cannot be directly read in a __device__/__global__ function

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

template <class T, class R>
concept HasAssign = requires(const T t, R r) {
  { t = r };
};

template <class T, class R>
concept HasNoexceptAssign = requires(const T t, R r) {
  { t = r } noexcept;
};

static_assert(!HasAssign<cuda::std::__constant_wrapper<5>, cuda::std::__constant_wrapper<3>>);
static_assert(!HasNoexceptAssign<cuda::std::__constant_wrapper<5>, cuda::std::__constant_wrapper<3>>);

static_assert(HasAssign<cuda::std::__constant_wrapper<WithOps{5}>, cuda::std::__constant_wrapper<3>>);
static_assert(HasNoexceptAssign<cuda::std::__constant_wrapper<WithOps{5}>, cuda::std::__constant_wrapper<3>>);

static_assert(!HasAssign<cuda::std::__constant_wrapper<OpsReturnNonStructural{5}>, cuda::std::__constant_wrapper<5>>);

TEST_FUNC constexpr bool test()
{
// nvcc == 13.0 produces invalid source file for the host compilers. It replaces contexpr variables with their values
// which doesn't work for assignment.
#if !_CCCL_CUDA_COMPILER(NVCC, ==, 13, 0)
  {
    // WithOps assignment
    const cuda::std::__constant_wrapper<WithOps{5}> cwOps5;
    cuda::std::__constant_wrapper<3> cw3;

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{8}>> decltype(auto) result = cwOps5 = cw3;
    static_assert(result.value.value == 8);
  }

  {
    // with integral_constant
    const cuda::std::__constant_wrapper<WithOps{5}> cwOps5;
    cuda::std::integral_constant<int, 3> ic3;

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{8}>> decltype(auto) result = cwOps5 = ic3;
    static_assert(result.value.value == 8);
  }
#endif // !_CCCL_CUDA_COMPILER(NVCC, ==, 13, 0)

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
