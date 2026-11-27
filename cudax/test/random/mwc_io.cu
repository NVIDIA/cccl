//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstdint>

#include <cuda/experimental/random.cuh>

#include <ios>
#include <sstream>
#include <string>

#include <c2h/catch2_test_helper.h>

template <class Engine, class CharT>
void test_roundtrip()
{
  std::basic_stringstream<CharT> stream;
  stream.setf(std::ios::hex, std::ios::basefield);
  stream.fill(CharT('#'));
  auto flags = stream.flags();

  Engine engine{42};
  engine.discard(1000);
  stream << engine;
  REQUIRE(stream.flags() == flags);
  REQUIRE(stream.fill() == CharT('#'));

  Engine copy;
  stream >> copy;
  REQUIRE(!stream.fail());
  REQUIRE(stream.flags() == flags);
  REQUIRE(copy == engine);
}

template <class Engine>
void test_invalid(const char* const text)
{
  std::istringstream stream{text};
  auto flags = stream.flags();

  Engine engine{42};
  auto saved = engine;
  stream >> engine;
  REQUIRE(stream.fail());
  REQUIRE(stream.flags() == flags);
  REQUIRE(engine == saved);
}

template <class Engine>
void test_reference(const char* const text, const cuda::std::uint32_t expected)
{
  Engine engine;
  std::istringstream stream{text};
  stream >> engine;
  REQUIRE(!stream.fail());
  REQUIRE(engine() == expected);
}

C2H_TEST("MWC stream input and output", "[random]")
{
  using cuda::experimental::mwc128x32;
  using cuda::experimental::mwc64x32;

  test_roundtrip<mwc64x32, char>();
  test_roundtrip<mwc128x32, char>();
  test_roundtrip<mwc64x32, wchar_t>();
  test_roundtrip<mwc128x32, wchar_t>();

  test_invalid<mwc64x32>("0 0");
  test_invalid<mwc64x32>("1");
  test_invalid<mwc64x32>("1 invalid");
  test_invalid<mwc64x32>("4294967296 1");
  test_invalid<mwc64x32>("4294967295 4294002167"); // m, an absorbing state
  test_invalid<mwc64x32>("0 4294002168"); // carry >= multiplier

  test_invalid<mwc128x32>("0 0 0 0");
  test_invalid<mwc128x32>("1 2 3");
  test_invalid<mwc128x32>("1 2 3 invalid");
  test_invalid<mwc128x32>("4294967295 4294967295 4294967295 4291611749");
  test_invalid<mwc128x32>("0 0 0 4291611750");

  test_reference<mwc64x32>("1 0", 1);
  test_reference<mwc64x32>("1 1", 0);
  test_reference<mwc128x32>("1 2 3 0", 3);
  test_reference<mwc128x32>("1 2 3 1", 2);
}
