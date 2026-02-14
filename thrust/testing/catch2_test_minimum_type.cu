#include <thrust/detail/type_traits/minimum_type.h>
#include <thrust/iterator/iterator_categories.h>

#include <cuda/std/type_traits>

#include "catch2_test_helper.h"

template <typename... Ts>
using mt = thrust::detail::minimum_type<Ts...>;

template <typename, typename... Ts>
inline constexpr bool mt_fails_impl = true;

template <typename... Ts>
inline constexpr bool mt_fails_impl<::cuda::std::void_t<mt<Ts...>>, Ts...> = false;

template <typename... Ts>
inline constexpr bool mt_fails = mt_fails_impl<void, Ts...>;

struct A
{};
struct B : A
{};
struct C : B
{};
struct C2 : B
{};

TEST_CASE("MinimumType", "[minimum_type]")
{
  using ::cuda::std::is_same_v;

  STATIC_REQUIRE(is_same_v<mt<int>, int>);
  STATIC_REQUIRE(is_same_v<mt<int, int>, int>);
  STATIC_REQUIRE(is_same_v<mt<int, int, int, int>, int>);

  STATIC_REQUIRE(is_same_v<mt<char, short, int>, char>);
  STATIC_REQUIRE(is_same_v<mt<int, short, char>, int>);

  STATIC_REQUIRE(is_same_v<mt<A, B, C>, A>);
  STATIC_REQUIRE(is_same_v<mt<C, B, A>, A>);

  STATIC_REQUIRE(is_same_v<mt<A, B, C>, A>);
  STATIC_REQUIRE(is_same_v<mt<C, B, A>, A>);
  STATIC_REQUIRE(is_same_v<mt<C, B, A, B, C>, A>);

  STATIC_REQUIRE(
    is_same_v<
      mt<::cuda::std::random_access_iterator_tag, ::cuda::std::input_iterator_tag, ::cuda::std::forward_iterator_tag>,
      ::cuda::std::input_iterator_tag>);
  STATIC_REQUIRE(is_same_v<mt<::cuda::std::random_access_iterator_tag,
                              ::cuda::std::random_access_iterator_tag,
                              ::cuda::std::random_access_iterator_tag>,
                           ::cuda::std::random_access_iterator_tag>);

  STATIC_REQUIRE(mt_fails<C, C2>);
  STATIC_REQUIRE(mt_fails<int, A>);
  STATIC_REQUIRE(mt_fails<int, A, B, C>);
  STATIC_REQUIRE(mt_fails<A, B, C, int>);
}
