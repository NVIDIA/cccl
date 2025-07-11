#include <thrust/extrema.h>

#include <unittest/unittest.h>

template <typename T>
struct TestMin
{
  void operator()(void)
  {
    // 2 < 3
    T two(2), three(3);
    ASSERT_EQUAL(two, ::cuda::std::min(two, three));
    ASSERT_EQUAL(two, ::cuda::std::min(two, three, ::cuda::std::less<T>()));

    ASSERT_EQUAL(two, ::cuda::std::min(three, two));
    ASSERT_EQUAL(two, ::cuda::std::min(three, two, ::cuda::std::less<T>()));

    ASSERT_EQUAL(three, ::cuda::std::min(two, three, ::cuda::std::greater<T>()));
    ASSERT_EQUAL(three, ::cuda::std::min(three, two, ::cuda::std::greater<T>()));

    using KV = key_value<T, T>;
    KV two_and_two(two, two);
    KV two_and_three(two, three);

    // the first element breaks ties
    ASSERT_EQUAL_QUIET(two_and_two, ::cuda::std::min(two_and_two, two_and_three));
    ASSERT_EQUAL_QUIET(two_and_three, ::cuda::std::min(two_and_three, two_and_two));

    ASSERT_EQUAL_QUIET(two_and_two, ::cuda::std::min(two_and_two, two_and_three, ::cuda::std::less<KV>()));
    ASSERT_EQUAL_QUIET(two_and_three, ::cuda::std::min(two_and_three, two_and_two, ::cuda::std::less<KV>()));

    ASSERT_EQUAL_QUIET(two_and_two, ::cuda::std::min(two_and_two, two_and_three, ::cuda::std::greater<KV>()));
    ASSERT_EQUAL_QUIET(two_and_three, ::cuda::std::min(two_and_three, two_and_two, ::cuda::std::greater<KV>()));
  }
};
SimpleUnitTest<TestMin, NumericTypes> TestMinInstance;

template <typename T>
struct TestMax
{
  void operator()(void)
  {
    // 2 < 3
    T two(2), three(3);
    ASSERT_EQUAL(three, ::cuda::std::max(two, three));
    ASSERT_EQUAL(three, ::cuda::std::max(two, three, ::cuda::std::less<T>()));

    ASSERT_EQUAL(three, ::cuda::std::max(three, two));
    ASSERT_EQUAL(three, ::cuda::std::max(three, two, ::cuda::std::less<T>()));

    ASSERT_EQUAL(two, ::cuda::std::max(two, three, ::cuda::std::greater<T>()));
    ASSERT_EQUAL(two, ::cuda::std::max(three, two, ::cuda::std::greater<T>()));

    using KV = key_value<T, T>;
    KV two_and_two(two, two);
    KV two_and_three(two, three);

    // the first element breaks ties
    ASSERT_EQUAL_QUIET(two_and_two, ::cuda::std::max(two_and_two, two_and_three));
    ASSERT_EQUAL_QUIET(two_and_three, ::cuda::std::max(two_and_three, two_and_two));

    ASSERT_EQUAL_QUIET(two_and_two, ::cuda::std::max(two_and_two, two_and_three, ::cuda::std::less<KV>()));
    ASSERT_EQUAL_QUIET(two_and_three, ::cuda::std::max(two_and_three, two_and_two, ::cuda::std::less<KV>()));

    ASSERT_EQUAL_QUIET(two_and_two, ::cuda::std::max(two_and_two, two_and_three, ::cuda::std::greater<KV>()));
    ASSERT_EQUAL_QUIET(two_and_three, ::cuda::std::max(two_and_three, two_and_two, ::cuda::std::greater<KV>()));
  }
};
SimpleUnitTest<TestMax, NumericTypes> TestMaxInstance;
