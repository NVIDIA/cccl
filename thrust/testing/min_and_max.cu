#include <thrust/extrema.h>

#include <unittest/unittest.h>

template <typename T>
struct TestMin
{
  void operator()()
  {
    // 2 < 3
    T two(2), three(3);
    REQUIRE(two == ::cuda::std::min(two, three));
    REQUIRE(two == ::cuda::std::min(two, three, ::cuda::std::less<T>()));

    REQUIRE(two == ::cuda::std::min(three, two));
    REQUIRE(two == ::cuda::std::min(three, two, ::cuda::std::less<T>()));

    REQUIRE(three == ::cuda::std::min(two, three, ::cuda::std::greater<T>()));
    REQUIRE(three == ::cuda::std::min(three, two, ::cuda::std::greater<T>()));

    using KV = key_value<T, T>;
    const KV two_and_two(two, two);
    const KV two_and_three(two, three);

    // the first element breaks ties
    ASSERT_EQUAL_QUIET(two_and_two, ::cuda::std::min(two_and_two, two_and_three));
    ASSERT_EQUAL_QUIET(two_and_three, ::cuda::std::min(two_and_three, two_and_two));

    ASSERT_EQUAL_QUIET(two_and_two, ::cuda::std::min(two_and_two, two_and_three, ::cuda::std::less<KV>()));
    ASSERT_EQUAL_QUIET(two_and_three, ::cuda::std::min(two_and_three, two_and_two, ::cuda::std::less<KV>()));

    ASSERT_EQUAL_QUIET(two_and_two, ::cuda::std::min(two_and_two, two_and_three, ::cuda::std::greater<KV>()));
    ASSERT_EQUAL_QUIET(two_and_three, ::cuda::std::min(two_and_three, two_and_two, ::cuda::std::greater<KV>()));
  }
};
DECLARE_GENERIC_UNITTEST_WITH_TYPES(TestMin, NumericTypes);

template <typename T>
struct TestMax
{
  void operator()()
  {
    // 2 < 3
    T two(2), three(3);
    REQUIRE(three == ::cuda::std::max(two, three));
    REQUIRE(three == ::cuda::std::max(two, three, ::cuda::std::less<T>()));

    REQUIRE(three == ::cuda::std::max(three, two));
    REQUIRE(three == ::cuda::std::max(three, two, ::cuda::std::less<T>()));

    REQUIRE(two == ::cuda::std::max(two, three, ::cuda::std::greater<T>()));
    REQUIRE(two == ::cuda::std::max(three, two, ::cuda::std::greater<T>()));

    using KV = key_value<T, T>;
    const KV two_and_two(two, two);
    const KV two_and_three(two, three);

    // the first element breaks ties
    ASSERT_EQUAL_QUIET(two_and_two, ::cuda::std::max(two_and_two, two_and_three));
    ASSERT_EQUAL_QUIET(two_and_three, ::cuda::std::max(two_and_three, two_and_two));

    ASSERT_EQUAL_QUIET(two_and_two, ::cuda::std::max(two_and_two, two_and_three, ::cuda::std::less<KV>()));
    ASSERT_EQUAL_QUIET(two_and_three, ::cuda::std::max(two_and_three, two_and_two, ::cuda::std::less<KV>()));

    ASSERT_EQUAL_QUIET(two_and_two, ::cuda::std::max(two_and_two, two_and_three, ::cuda::std::greater<KV>()));
    ASSERT_EQUAL_QUIET(two_and_three, ::cuda::std::max(two_and_three, two_and_two, ::cuda::std::greater<KV>()));
  }
};
DECLARE_GENERIC_UNITTEST_WITH_TYPES(TestMax, NumericTypes);
