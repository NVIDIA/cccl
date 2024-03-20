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

#include <cub/detail/requirements.cuh>

struct A{ int v; };
struct B{};
struct C{};
struct D{};

template <class T>
bool operator<(A, T) { return false; }
constexpr bool operator<(B, A) { return true; }

template <class T>
constexpr bool operator<(C, T) { return false; }

constexpr bool operator<(D, D) { return false; }

TEST_CASE("requirement works", "[requirements][meta]")
{
  STATIC_REQUIRE(cub::detail::requirements::requirement<A>::value == true);
  STATIC_REQUIRE(cub::detail::requirements::requirement<B>::value == false);
}

TEST_CASE("list_of_requirement works", "[requirements][meta]")
{
  STATIC_REQUIRE(cub::detail::requirements::list_of_requirements<A>::value == true);
  STATIC_REQUIRE(cub::detail::requirements::list_of_requirements<B>::value == false);

  STATIC_REQUIRE(cub::detail::requirements::list_of_requirements<A, A, A>::value == true);
  STATIC_REQUIRE(cub::detail::requirements::list_of_requirements<B, A, A>::value == false);
  STATIC_REQUIRE(cub::detail::requirements::list_of_requirements<A, B, A>::value == false);
  STATIC_REQUIRE(cub::detail::requirements::list_of_requirements<A, A, B>::value == false);
}

TEST_CASE("list_of_requirements_from_unique_categories works", "[requirements][meta]")
{
  STATIC_REQUIRE(cub::detail::requirements::list_of_requirements_from_unique_categories<>::value == true);
  STATIC_REQUIRE(cub::detail::requirements::list_of_requirements_from_unique_categories<A>::value == true);
  STATIC_REQUIRE(cub::detail::requirements::list_of_requirements_from_unique_categories<C>::value == true);
  STATIC_REQUIRE(cub::detail::requirements::list_of_requirements_from_unique_categories<A, C>::value == false);
  STATIC_REQUIRE(cub::detail::requirements::list_of_requirements_from_unique_categories<A, D>::value == true);
  STATIC_REQUIRE(cub::detail::requirements::list_of_requirements_from_unique_categories<D, C>::value == true);
  STATIC_REQUIRE(cub::detail::requirements::list_of_requirements_from_unique_categories<D, D>::value == false);
  STATIC_REQUIRE(cub::detail::requirements::list_of_requirements_from_unique_categories<A, D, C>::value == false);
  STATIC_REQUIRE(cub::detail::requirements::list_of_requirements_from_unique_categories<A, C, D>::value == false);
  STATIC_REQUIRE(cub::detail::requirements::list_of_requirements_from_unique_categories<C, A, D>::value == false);
  STATIC_REQUIRE(cub::detail::requirements::list_of_requirements_from_unique_categories<C, D, A>::value == false);
}

TEST_CASE("first_categorial_match_id works", "[requirements][meta]")
{
  STATIC_REQUIRE(cub::detail::requirements::first_categorial_match_id<A>::value == 0);
  STATIC_REQUIRE(cub::detail::requirements::first_categorial_match_id<A, A>::value == 0);
  STATIC_REQUIRE(cub::detail::requirements::first_categorial_match_id<A, C>::value == 0);
  STATIC_REQUIRE(cub::detail::requirements::first_categorial_match_id<A, C, D>::value == 0);
  STATIC_REQUIRE(cub::detail::requirements::first_categorial_match_id<A, D, C>::value == 1);
}

TEST_CASE("first_categorial_match works", "[requirements][meta]")
{
  STATIC_REQUIRE(std::is_same<cub::detail::requirements::first_categorial_match<A, A>, A>::value);
  STATIC_REQUIRE(std::is_same<cub::detail::requirements::first_categorial_match<A, D, C>, C>::value);
  STATIC_REQUIRE(std::is_same<cub::detail::requirements::first_categorial_match<A, D, A>, A>::value);
  STATIC_REQUIRE(std::is_same<cub::detail::requirements::first_categorial_match<A, A, D>, A>::value);
  STATIC_REQUIRE(std::is_same<cub::detail::requirements::first_categorial_match<A, C, D>, C>::value);
}

TEST_CASE("requirements_categorically_match_guarantees works", "[requirements][meta]")
{
  STATIC_REQUIRE(
    cub::detail::requirements::requirements_categorically_match_guarantees<::cuda::std::tuple<A>,
                                                                           ::cuda::std::tuple<A>>::value);
  STATIC_REQUIRE(
    cub::detail::requirements::requirements_categorically_match_guarantees<::cuda::std::tuple<A, D>,
                                                                           ::cuda::std::tuple<A>>::value);
  STATIC_REQUIRE(
    cub::detail::requirements::requirements_categorically_match_guarantees<::cuda::std::tuple<A, D>,
                                                                           ::cuda::std::tuple<D>>::value);
  STATIC_REQUIRE(
    cub::detail::requirements::requirements_categorically_match_guarantees<::cuda::std::tuple<A, D>,
                                                                           ::cuda::std::tuple<A, D>>::value);
  STATIC_REQUIRE(
    cub::detail::requirements::requirements_categorically_match_guarantees<::cuda::std::tuple<A, D>,
                                                                           ::cuda::std::tuple<A, D>>::value);
  STATIC_REQUIRE(
    cub::detail::requirements::requirements_categorically_match_guarantees<::cuda::std::tuple<A, D>,
                                                                           ::cuda::std::tuple<C>>::value);
  STATIC_REQUIRE(
    !cub::detail::requirements::requirements_categorically_match_guarantees<::cuda::std::tuple<D>,
                                                                            ::cuda::std::tuple<C>>::value);
}

TEST_CASE("get works", "[requirements][meta]")
{
  A a1{1};
  A a2{2};

  REQUIRE(cub::detail::requirements::match<C>(::cuda::std::make_tuple(a1)).v == a1.v);
  REQUIRE(cub::detail::requirements::match<C>(::cuda::std::make_tuple(a2)).v == a2.v);
  REQUIRE(cub::detail::requirements::match<C>(::cuda::std::make_tuple(D{}, a2)).v == a2.v);
  REQUIRE(cub::detail::requirements::match<C>(::cuda::std::make_tuple(a2, D{})).v == a2.v);
}

TEST_CASE("mask works", "[requirements][meta]")
{
  A a1{1};
  A a2{2};

  {
    auto defaults = ::cuda::std::make_tuple(a1);
    auto requirements = ::cuda::std::make_tuple();
    auto guarantees = cub::detail::requirements::mask(defaults, requirements);

    REQUIRE(::cuda::std::get<0>(guarantees).v == 1);
  }

  {
    auto defaults = ::cuda::std::make_tuple(a1);
    auto requirements = ::cuda::std::make_tuple(a2);
    auto guarantees = cub::detail::requirements::mask(defaults, requirements);

    REQUIRE(::cuda::std::get<0>(guarantees).v == 2);
  }

  {
    auto defaults = ::cuda::std::make_tuple(a1, D{});
    auto requirements = ::cuda::std::make_tuple(a2);
    auto guarantees = cub::detail::requirements::mask(defaults, requirements);

    REQUIRE(::cuda::std::get<0>(guarantees).v == 2);
  }

  {
    auto defaults = ::cuda::std::make_tuple(D{}, a1);
    auto requirements = ::cuda::std::make_tuple(a2);
    auto guarantees = cub::detail::requirements::mask(defaults, requirements);

    REQUIRE(::cuda::std::get<1>(guarantees).v == 2);
  }
}

TEST_CASE("require works", "[requirements][meta]")
{
  A a1{1};
  A a2{2};
  D d{};

  {
    auto guarantees = cub::require(a1);

    REQUIRE(::cuda::std::get<0>(guarantees).v == a1.v);
    REQUIRE(::cuda::std::tuple_size<decltype(guarantees)>::value == 1);
  }

  {
    auto guarantees = cub::require(a2, d);

    REQUIRE(::cuda::std::get<0>(guarantees).v == a2.v);
    REQUIRE(::cuda::std::tuple_size<decltype(guarantees)>::value == 2);
  }
}
