#include <thrust/detail/type_traits/minimum_type.h>
#include <thrust/iterator/iterator_categories.h>

#include <cuda/std/type_traits>

#include <unittest/unittest.h>

template <typename... Ts>
using mt = typename thrust::detail::minimum_type<Ts...>::type;

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

void TestMinimumType()
{
  using ::cuda::std::is_same_v;

  static_assert(is_same_v<mt<int>, int>);
  static_assert(is_same_v<mt<int, int>, int>);
  static_assert(is_same_v<mt<int, int, int, int>, int>);

  static_assert(is_same_v<mt<char, short, int>, char>);
  static_assert(is_same_v<mt<int, short, char>, int>);

  static_assert(is_same_v<mt<A, B, C>, A>);
  static_assert(is_same_v<mt<C, B, A>, A>);

  static_assert(is_same_v<mt<A, B, C>, A>);
  static_assert(is_same_v<mt<C, B, A>, A>);
  static_assert(is_same_v<mt<C, B, A, B, C>, A>);

  static_assert(
    is_same_v<
      mt<::cuda::std::random_access_iterator_tag, ::cuda::std::input_iterator_tag, ::cuda::std::forward_iterator_tag>,
      ::cuda::std::input_iterator_tag>);
  static_assert(is_same_v<mt<::cuda::std::random_access_iterator_tag,
                             ::cuda::std::random_access_iterator_tag,
                             ::cuda::std::random_access_iterator_tag>,
                          ::cuda::std::random_access_iterator_tag>);

  static_assert(mt_fails<C, C2>);
  static_assert(mt_fails<int, A>);
  // static_assert(mt_fails<int, A, B, C>);
  //  static_assert(mt_fails<A, B, C, int>);
}
DECLARE_UNITTEST(TestMinimumType);
