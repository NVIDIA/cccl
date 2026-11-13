//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// The engine has no tile execution annotations.
// UNSUPPORTED: force-tile

#include <cuda/random>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/initializer_list>
#include <cuda/std/random>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::experimental::mwc128x32;
using cuda::experimental::mwc64x32;

static_assert(sizeof(mwc64x32) == 8);
static_assert(sizeof(mwc128x32) == 16);
static_assert(cuda::std::is_trivially_copyable_v<mwc64x32>);
static_assert(cuda::std::is_trivially_copyable_v<mwc128x32>);

struct seed_words
{
  uint32_t value;

  template <class Iterator>
  TEST_HOST_DEVICE_FUNC constexpr void generate(Iterator first, const Iterator last)
  {
    while (first != last)
    {
      *first++ = value;
    }
  }
};

template <class Engine>
TEST_HOST_DEVICE_FUNC constexpr bool test_interface()
{
  using cuda::std::is_same_v;
  static_assert(is_same_v<typename Engine::result_type, uint32_t>);
  static_assert(Engine::min() == 0);
  static_assert(Engine::max() == 0xFFFFFFFFu);
  static_assert(is_same_v<decltype(Engine::min()), typename Engine::result_type>);
  static_assert(is_same_v<decltype(Engine::max()), typename Engine::result_type>);
  static_assert(is_same_v<decltype(Engine{}()), typename Engine::result_type>);

  static_assert(noexcept(Engine{}));
  static_assert(noexcept(Engine{42}));
  static_assert(noexcept(Engine{}.seed()));
  static_assert(noexcept(Engine{}()));
  static_assert(noexcept(Engine{}.discard(0)));

  // Default construction uses default_seed
  Engine engine;
  assert(engine == Engine{Engine::default_seed});

  // Copy construction preserves the state and the next output
  auto copy = engine;
  assert(copy() == engine());

  // Advancing only one engine makes their states unequal
  static_cast<void>(engine());
  assert(copy != engine);

  // Copy assignment restores equal states
  copy = engine;
  assert(copy == engine);

  // Reseeding matches construction with the same seed
  engine.seed(0);
  assert(engine == Engine{0});

  // Discarding 16 outputs matches 16 calls to operator().
  engine.seed(1234);
  auto step_engine = engine;
  for (int i = 0; i < 16; ++i)
  {
    static_cast<void>(step_engine());
  }
  engine.discard(16);
  assert(engine == step_engine);

  return true;
}

template <class Engine>
TEST_HOST_DEVICE_FUNC constexpr void test_discard()
{
  for (auto count : {0ull, 1ull, 15ull, 16ull, 17ull, 127ull, 1000ull})
  {
    Engine step_engine{42};
    auto jump_engine = step_engine;
    for (unsigned long long i = 0; i < count; ++i)
    {
      static_cast<void>(step_engine());
    }
    jump_engine.discard(count);
    assert(step_engine == jump_engine);
    assert(step_engine() == jump_engine());
  }
}

template <class Engine>
TEST_HOST_DEVICE_FUNC TEST_CONSTEXPR_CXX20 void test_seed_sequence()
{
  cuda::std::seed_seq seq{0u, 1u, 2u, 3u, 0xFFFFFFFFu};
  Engine first{seq};
  Engine second;
  second.seed(seq);
  assert(first == second);

  // test distribution correctness
  cuda::std::uniform_int_distribution<unsigned> dist{0, 17};
  for (int i = 0; i < 100; ++i)
  {
    assert(dist(first) <= 17);
  }
}

struct reference_values
{
  uint64_t seed;
  unsigned long long discard;
  uint32_t output;
};

// Generated with Vigna's MWC64X_32.c / MWC128X_32.c, seeded from SplitMix64:
// https://github.com/vigna/MWCPRNG/tree/07f35668a41e6b18c20439867c2ad11fa33bff73/prngs
template <class Engine, cuda::std::size_t Size>
TEST_HOST_DEVICE_FUNC constexpr void test_reference(const reference_values (&vectors)[Size])
{
  for (auto vector : vectors)
  {
    Engine engine{vector.seed};
    engine.discard(vector.discard);
    assert(engine() == vector.output);
  }
}

TEST_HOST_DEVICE_FUNC constexpr bool test()
{
  assert(test_interface<mwc64x32>());
  assert(test_interface<mwc128x32>());
  return true;
}

// Full jump tests exceed constexpr step limits
TEST_HOST_DEVICE_FUNC void runtime_test()
{
  test_discard<mwc64x32>();
  test_discard<mwc128x32>();
  test_seed_sequence<mwc64x32>();
  test_seed_sequence<mwc128x32>();

  const reference_values mwc64x32_vectors[] = {
    {0ull, 0ull, 3668825573u},
    {0ull, 1ull, 2594716740u},
    {0ull, 10000ull, 3744365330u},
    {0ull, 4294967296ull, 71135566u},
    {0ull, 9223372036854775808ull, 1050179472u},
    {0ull, 18446744073709551615ull, 923575008u},
    {1ull, 0ull, 3968550920u},
    {1ull, 1ull, 3682176977u},
    {1ull, 10000ull, 546496306u},
    {1ull, 4294967296ull, 1668101407u},
    {1ull, 9223372036854775808ull, 2222108681u},
    {1ull, 18446744073709551615ull, 2398742057u},
    {42ull, 0ull, 2646065960u},
    {42ull, 1ull, 929717238u},
    {42ull, 10000ull, 607205018u},
    {42ull, 4294967296ull, 3380939454u},
    {42ull, 9223372036854775808ull, 2054333958u},
    {42ull, 18446744073709551615ull, 1733534452u},
    {18446744073709551615ull, 0ull, 3230463829u},
    {18446744073709551615ull, 1ull, 261268165u},
    {18446744073709551615ull, 10000ull, 4152517651u},
    {18446744073709551615ull, 4294967296ull, 3392441641u},
    {18446744073709551615ull, 9223372036854775808ull, 4163917775u},
    {18446744073709551615ull, 18446744073709551615ull, 1230777632u},
  };
  test_reference<mwc64x32>(mwc64x32_vectors);
  const reference_values mwc128x32_vectors[] = {
    {0ull, 0ull, 4064076011u},
    {0ull, 1ull, 57942192u},
    {0ull, 10000ull, 3888026831u},
    {0ull, 4294967296ull, 2075148387u},
    {0ull, 9223372036854775808ull, 1539641713u},
    {0ull, 18446744073709551615ull, 3866290765u},
    {1ull, 0ull, 354514155u},
    {1ull, 1ull, 123672368u},
    {1ull, 10000ull, 459744823u},
    {1ull, 4294967296ull, 3457169204u},
    {1ull, 9223372036854775808ull, 3919537568u},
    {1ull, 18446744073709551615ull, 2230146024u},
    {42ull, 0ull, 491231386u},
    {42ull, 1ull, 1678838307u},
    {42ull, 10000ull, 4135714192u},
    {42ull, 4294967296ull, 882459584u},
    {42ull, 9223372036854775808ull, 1932495069u},
    {42ull, 18446744073709551615ull, 3455471807u},
    {18446744073709551615ull, 0ull, 2045921046u},
    {18446744073709551615ull, 1ull, 1160145612u},
    {18446744073709551615ull, 10000ull, 2315963632u},
    {18446744073709551615ull, 4294967296ull, 3140243374u},
    {18446744073709551615ull, 9223372036854775808ull, 2531678749u},
    {18446744073709551615ull, 18446744073709551615ull, 4254859935u},
  };
  test_reference<mwc128x32>(mwc128x32_vectors);
}

int main(int, char**)
{
  assert(test()); // NOLINT(misc-static-assert)
  static_assert(test());
  runtime_test();
  return 0;
}
