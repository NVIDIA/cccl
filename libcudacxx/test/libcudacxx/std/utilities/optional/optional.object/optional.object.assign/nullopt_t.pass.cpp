//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <cuda/std/optional>

// optional<T>& operator=(nullopt_t) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "archetypes.h"
#include "test_macros.h"

using cuda::std::nullopt;
using cuda::std::nullopt_t;
using cuda::std::optional;

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  enum class State
  {
    inactive,
    constructed,
    destroyed
  };
  State state = State::inactive;

  struct StateTracker
  {
    __host__ __device__ TEST_CONSTEXPR_CXX20 StateTracker(State& s)
        : state_(&s)
    {
      s = State::constructed;
    }
    __host__ __device__ TEST_CONSTEXPR_CXX20 ~StateTracker()
    {
      *state_ = State::destroyed;
    }

    State* state_;
  };
  {
    optional<int> opt;
    static_assert(noexcept(opt = nullopt) == true, "");
    opt = nullopt;
    assert(static_cast<bool>(opt) == false);
  }
  {
    optional<int> opt(3);
    opt = nullopt;
    assert(static_cast<bool>(opt) == false);
  }
  {
    optional<StateTracker> opt;
    opt = nullopt;
    assert(state == State::inactive);
    assert(static_cast<bool>(opt) == false);
  }
  {
    optional<StateTracker> opt(state);
    assert(state == State::constructed);
    opt = nullopt;
    assert(state == State::destroyed);
    assert(static_cast<bool>(opt) == false);
  }
  return true;
}

int main(int, char**)
{
#if TEST_STD_VER > 2017
  static_assert(test());
#endif
  test();
  using TT = TestTypes::TestType;
  TT::reset();
  {
    optional<TT> opt;
    static_assert(noexcept(opt = nullopt) == true, "");
    assert(TT::destroyed() == 0);
    opt = nullopt;
    assert(TT::constructed() == 0);
    assert(TT::alive() == 0);
    assert(TT::destroyed() == 0);
    assert(static_cast<bool>(opt) == false);
  }
  assert(TT::alive() == 0);
  assert(TT::destroyed() == 0);
  TT::reset();
  {
    optional<TT> opt(42);
    assert(TT::destroyed() == 0);
    TT::reset_constructors();
    opt = nullopt;
    assert(TT::constructed() == 0);
    assert(TT::alive() == 0);
    assert(TT::destroyed() == 1);
    assert(static_cast<bool>(opt) == false);
  }
  assert(TT::alive() == 0);
  assert(TT::destroyed() == 1);
  TT::reset();

  return 0;
}
