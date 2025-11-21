//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#if !_CCCL_COMPILER(NVRTC)
#  include <sstream>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__random_>
#if !_CCCL_COMPILER(NVRTC)
#  include <sstream>
#endif // !_CCCL_COMPILER(NVRTC)

#include "test_macros.h"

template <typename Engine>
__host__ __device__ TEST_CONSTEXPR_CXX20 bool test_ctor()
{
  Engine e1;
  Engine e2(Engine::default_seed);
  assert(e1 == e2);
  Engine e3(42);
  assert(e3 != e2);
  auto seq = cuda::std::seed_seq{};
  Engine e4(seq);
  Engine e5 = e4;
  assert(e4 == e5);
  static_assert(noexcept(Engine()));
  static_assert(noexcept(Engine(42)));
  return true;
}

template <typename Engine>
__host__ __device__ TEST_CONSTEXPR_CXX20 bool test_copy()
{
  Engine e1;
  Engine e2 = e1;
  assert(e1 == e2);
  e1();
  assert(e1 != e2);
  e2 = e1;
  assert(e1 == e2);

  static_assert(noexcept(Engine(e1)));
  static_assert(noexcept(e2 = e1));

  return true;
}

template <typename Engine>
__host__ __device__ TEST_CONSTEXPR_CXX20 bool test_seed()
{
  Engine e1(23);
  Engine e2;
  e2.seed(Engine::default_seed);
  assert(e1 != e2);
  e1.seed(Engine::default_seed);
  assert(e1 == e2);

  auto seq = cuda::std::seed_seq{};
  static_assert(cuda::std::is_void_v<decltype(e1.seed(seq))>);
  static_assert(cuda::std::is_void_v<decltype(e1.seed())>);
  static_assert(cuda::std::is_void_v<decltype(e1.seed(23))>);
  static_assert(noexcept(e1.seed()));
  static_assert(noexcept(e1.seed(23)));
  return true;
}

template <typename Engine>
__host__ __device__ TEST_CONSTEXPR_CXX20 bool test_operator()
{
  Engine e1;
  static_assert(cuda::std::is_same_v<decltype(e1()), typename Engine::result_type>);
  e1();
  Engine e2;
  assert(e1 != e2);
  e2();
  assert(e1 == e2);
  return true;
}

template <typename Engine, typename Engine::result_type value_10000>
__host__ __device__ TEST_CONSTEXPR_CXX20 bool test_discard()
{
  Engine e;
  for (int i = 0; i < 100; ++i)
  {
    Engine e2;
    e2.discard(i);
    assert(e == e2);
    e();
  }

  e = Engine();
  e.discard(9999);
  assert(e() == value_10000);

  static_assert(cuda::std::is_void_v<decltype(e.discard(10))>);
  static_assert(noexcept(e.discard(10)));

  return true;
}

template <typename Engine>
__host__ __device__ TEST_CONSTEXPR_CXX20 bool test_equality()
{
  Engine e;
  assert(e == e);
  Engine e2;
  assert(e == e2);
  e();
  assert(e != e2);
  e  = Engine(3);
  e2 = Engine(3);
  assert(e == e2);
  e2 = Engine(4);
  assert(e != e2);

  static_assert(noexcept(e == e2));
  static_assert(noexcept(e != e2));
  return true;
}

template <typename Engine>
__host__ __device__ TEST_CONSTEXPR_CXX20 bool test_min_max()
{
  const auto seeds = {0, 29332, 9000};
  for (auto seed : seeds)
  {
    Engine e(seed);
    for (int i = 0; i < 100; ++i)
    {
      auto val = e();
      assert(val <= Engine::max());
      // Avoid pointless comparison of unsigned values with 0 warning
      if constexpr (Engine::min() > 0)
      {
        assert(val >= Engine::min());
      }
    }
  }
  static_assert(Engine::min() <= Engine::max());
  static_assert(noexcept(Engine::min()));
  static_assert(noexcept(Engine::max()));
  static_assert(cuda::std::is_same_v<decltype(Engine::min()), typename Engine::result_type>);
  static_assert(cuda::std::is_same_v<decltype(Engine::max()), typename Engine::result_type>);
  return true;
}

#if !_CCCL_COMPILER(NVRTC)
#  include <sstream>
template <typename Engine>
void test_save_restore()
{
  Engine e0;
  e0.discard(10000);
  std::stringstream ss;
  ss << e0;

  e0.discard(10000);
  Engine e1;
  ss >> e1;
  e1.discard(10000);
  assert(e0() == e1());
}
#endif // !_CCCL_COMPILER(NVRTC)

template <typename Engine, typename Engine::result_type value_10000>
__host__ __device__ TEST_CONSTEXPR_CXX20 bool test_engine()
{
  test_ctor<Engine>();
  test_seed<Engine>();
  test_copy<Engine>();
  test_operator<Engine>();
  test_discard<Engine, value_10000>();
  test_equality<Engine>();
  test_min_max<Engine>();
  NV_IF_TARGET(NV_IS_HOST, ({ test_save_restore<Engine>(); }));
#if TEST_STD_VER >= 2020
  static_assert(test_ctor<Engine>());
  static_assert(test_seed<Engine>());
  static_assert(test_copy<Engine>());
  static_assert(test_operator<Engine>());
  static_assert(test_discard<Engine, value_10000>());
  static_assert(test_equality<Engine>());
  static_assert(test_min_max<Engine>());
#endif
  return true;
}
