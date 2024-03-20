/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include "catch2_test_helper.h"

#include <cub/detail/meta.cuh>

TEST_CASE("all_t works", "[meta][utils]")
{
  STATIC_REQUIRE(cub::detail::all_t<false, true, true>::value == false);
  STATIC_REQUIRE(cub::detail::all_t<true, false, true>::value == false);
  STATIC_REQUIRE(cub::detail::all_t<true, true, false>::value == false);
  STATIC_REQUIRE(cub::detail::all_t<true, true, true>::value == true);
  STATIC_REQUIRE(cub::detail::all_t<true, true>::value == true);
  STATIC_REQUIRE(cub::detail::all_t<true>::value == true);
  STATIC_REQUIRE(cub::detail::all_t<>::value == true);
}

TEST_CASE("none_t works", "[meta][utils]")
{
  STATIC_REQUIRE(cub::detail::none_t<false, true, true>::value == false);
  STATIC_REQUIRE(cub::detail::none_t<true, false, true>::value == false);
  STATIC_REQUIRE(cub::detail::none_t<true, true, false>::value == false);
  STATIC_REQUIRE(cub::detail::none_t<true, true, true>::value == false);
  STATIC_REQUIRE(cub::detail::none_t<false, false>::value == true);
  STATIC_REQUIRE(cub::detail::none_t<false>::value == true);
  STATIC_REQUIRE(cub::detail::none_t<>::value == true);
}

struct A{};
struct B{};
struct C{};
struct D{};

template <class T>
bool operator<(A, T) { return false; }
constexpr bool operator<(B, A) { return true; }
constexpr bool operator<(A, C) { return false; }

constexpr bool operator<(A, D) { return false; }
constexpr bool operator<(D, A) { return false; }

TEST_CASE("ordered works", "[meta][utils]")
{
  STATIC_REQUIRE(cub::detail::ordered<A, B>::value == true);
  STATIC_REQUIRE(cub::detail::ordered<B, A>::value == true);
  STATIC_REQUIRE(cub::detail::ordered<A, C>::value == false);
  STATIC_REQUIRE(cub::detail::ordered<B, C>::value == false);
}

TEST_CASE("staticly_ordered works", "[meta][utils]")
{
  STATIC_REQUIRE(cub::detail::statically_ordered<A, B>::value == false);
  STATIC_REQUIRE(cub::detail::statically_ordered<B, A>::value == true);
  STATIC_REQUIRE(cub::detail::statically_ordered<A, C>::value == false);
  STATIC_REQUIRE(cub::detail::statically_ordered<B, C>::value == false);
}

TEST_CASE("staticly_less works", "[meta][utils]")
{
  STATIC_REQUIRE(cub::detail::statically_less<A, B>::value == false);
  STATIC_REQUIRE(cub::detail::statically_less<B, A>::value == true);
  STATIC_REQUIRE(cub::detail::statically_less<A, C>::value == false);
  STATIC_REQUIRE(cub::detail::statically_less<B, C>::value == false);
}

TEST_CASE("staticly_equal works", "[meta][utils]")
{
  STATIC_REQUIRE(cub::detail::statically_equal<A, A>::value == false);
  STATIC_REQUIRE(cub::detail::statically_equal<A, D>::value == true);
  STATIC_REQUIRE(cub::detail::statically_equal<D, A>::value == true);
  STATIC_REQUIRE(cub::detail::statically_equal<A, C>::value == false);
  STATIC_REQUIRE(cub::detail::statically_equal<B, C>::value == false);
}
