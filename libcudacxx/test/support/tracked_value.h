//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef SUPPORT_TRACKED_VALUE_H
#define SUPPORT_TRACKED_VALUE_H

#include <cuda/std/cassert>

#include "test_macros.h"

struct TrackedValue
{
  enum State
  {
    CONSTRUCTED,
    MOVED_FROM,
    DESTROYED
  };
  State state;

  TrackedValue()
      : state(State::CONSTRUCTED)
  {}

  TrackedValue(TrackedValue const& t)
      : state(State::CONSTRUCTED)
  {
    assert(t.state != State::MOVED_FROM && "copying a moved-from object");
    assert(t.state != State::DESTROYED && "copying a destroyed object");
  }

  TrackedValue(TrackedValue&& t)
      : state(State::CONSTRUCTED)
  {
    assert(t.state != State::MOVED_FROM && "double moving from an object");
    assert(t.state != State::DESTROYED && "moving from a destroyed object");
    t.state = State::MOVED_FROM;
  }

  TrackedValue& operator=(TrackedValue const& t)
  {
    assert(state != State::DESTROYED && "copy assigning into destroyed object");
    assert(t.state != State::MOVED_FROM && "copying a moved-from object");
    assert(t.state != State::DESTROYED && "copying a destroyed object");
    state = t.state;
    return *this;
  }

  TrackedValue& operator=(TrackedValue&& t)
  {
    assert(state != State::DESTROYED && "move assigning into destroyed object");
    assert(t.state != State::MOVED_FROM && "double moving from an object");
    assert(t.state != State::DESTROYED && "moving from a destroyed object");
    state   = t.state;
    t.state = State::MOVED_FROM;
    return *this;
  }

  ~TrackedValue()
  {
    assert(state != State::DESTROYED && "double-destroying an object");
    state = State::DESTROYED;
  }
};

#endif // SUPPORT_TRACKED_VALUE_H
