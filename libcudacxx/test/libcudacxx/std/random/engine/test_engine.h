//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

template <typename Engine, typename Engine::result_type value_10000>
__host__ __device__ constexpr bool test_discard()
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

  return true;
}

template <typename Engine>
__host__ __device__ constexpr bool test_equality()
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
  return true;
}

template <typename Engine>
__host__ __device__ constexpr bool test_min_max()
{
  const auto seeds = {0, 29332, 9000};
  for (auto seed : seeds)
  {
    Engine e;
    for (int i = 0; i < 100; ++i)
    {
      assert(e() <= Engine::max());
    }
  }
  return true;
}

#if !_CCCL_COMPILER(NVRTC)
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
__host__ __device__ constexpr bool test_engine()
{
  test_discard<Engine, value_10000>();
  static_assert(test_discard<Engine, value_10000>());
  test_equality<Engine>();
  static_assert(test_equality<Engine>());
  test_min_max<Engine>();
  static_assert(test_min_max<Engine>());
  NV_IF_TARGET(NV_IS_HOST, ({ test_save_restore<Engine>(); }));
  return true;
}
