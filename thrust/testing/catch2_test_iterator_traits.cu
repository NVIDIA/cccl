#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <vector>

#include "catch2_test_helper.h"

template <typename Cat, typename Tag>
inline constexpr bool system_is = cuda::std::is_same_v<thrust::detail::iterator_category_to_system_t<Cat>, Tag>;

TEST_CASE("iterator_category_to_system", "[iterators]")
{
  STATIC_REQUIRE(system_is<thrust::input_device_iterator_tag, thrust::device_system_tag>);
  STATIC_REQUIRE(system_is<thrust::output_device_iterator_tag, thrust::device_system_tag>);
  STATIC_REQUIRE(system_is<thrust::forward_device_iterator_tag, thrust::device_system_tag>);
  STATIC_REQUIRE(system_is<thrust::bidirectional_device_iterator_tag, thrust::device_system_tag>);
  STATIC_REQUIRE(system_is<thrust::random_access_device_iterator_tag, thrust::device_system_tag>);

  STATIC_REQUIRE(system_is<thrust::input_host_iterator_tag, thrust::host_system_tag>);
  STATIC_REQUIRE(system_is<thrust::output_host_iterator_tag, thrust::host_system_tag>);
  STATIC_REQUIRE(system_is<thrust::forward_host_iterator_tag, thrust::host_system_tag>);
  STATIC_REQUIRE(system_is<thrust::bidirectional_host_iterator_tag, thrust::host_system_tag>);
  STATIC_REQUIRE(system_is<thrust::random_access_host_iterator_tag, thrust::host_system_tag>);
}

template <typename Cat, typename Traversal>
inline constexpr bool traversal_is =
  cuda::std::is_same_v<thrust::detail::iterator_category_to_traversal_t<Cat>, Traversal>;

TEST_CASE("iterator_category_to_traversal", "[iterators]")
{
  STATIC_REQUIRE(traversal_is<thrust::input_device_iterator_tag, thrust::single_pass_traversal_tag>);
  STATIC_REQUIRE(traversal_is<thrust::output_device_iterator_tag, thrust::incrementable_traversal_tag>);
  STATIC_REQUIRE(traversal_is<thrust::forward_device_iterator_tag, thrust::forward_traversal_tag>);
  STATIC_REQUIRE(traversal_is<thrust::bidirectional_device_iterator_tag, thrust::bidirectional_traversal_tag>);
  STATIC_REQUIRE(traversal_is<thrust::random_access_device_iterator_tag, thrust::random_access_traversal_tag>);

  STATIC_REQUIRE(traversal_is<thrust::input_host_iterator_tag, thrust::single_pass_traversal_tag>);
  STATIC_REQUIRE(traversal_is<thrust::output_host_iterator_tag, thrust::incrementable_traversal_tag>);
  STATIC_REQUIRE(traversal_is<thrust::forward_host_iterator_tag, thrust::forward_traversal_tag>);
  STATIC_REQUIRE(traversal_is<thrust::bidirectional_host_iterator_tag, thrust::bidirectional_traversal_tag>);
  STATIC_REQUIRE(traversal_is<thrust::random_access_host_iterator_tag, thrust::random_access_traversal_tag>);
}

struct cuda_make_counting_iterator
{
  template <typename... Args>
  auto operator()(Args&&... args) const
  {
    return cuda::make_counting_iterator(cuda::std::forward<Args>(args)...);
  }
};

struct cuda_make_transform_iterator
{
  template <typename... Args>
  auto operator()(Args&&... args) const
  {
    return cuda::make_transform_iterator(cuda::std::forward<Args>(args)...);
  }
};

struct cuda_make_zip_iterator
{
  template <typename... Args>
  auto operator()(Args&&... args) const
  {
    return cuda::make_zip_iterator(cuda::std::forward<Args>(args)...);
  }
};

struct thrust_make_counting_iterator
{
  template <typename... Args>
  auto operator()(Args&&... args) const
  {
    return thrust::make_counting_iterator(cuda::std::forward<Args>(args)...);
  }
};

struct thrust_make_transform_iterator
{
  template <typename... Args>
  auto operator()(Args&&... args) const
  {
    return thrust::make_transform_iterator(cuda::std::forward<Args>(args)...);
  }
};

struct thrust_make_zip_iterator
{
  template <typename... Args>
  auto operator()(Args&&... args) const
  {
    return thrust::make_zip_iterator(cuda::std::forward<Args>(args)...);
  }
};

template <typename Iterator>
inline constexpr bool has_random_access_traversal =
  cuda::std::is_same_v<thrust::iterator_traversal_t<Iterator>, thrust::random_access_traversal_tag>;

using cuda::std::__type_cartesian_product;
using cuda::std::__type_list;

using make_counting_t  = __type_list<cuda_make_counting_iterator, thrust_make_counting_iterator>;
using make_transform_t = __type_list<cuda_make_transform_iterator, thrust_make_transform_iterator>;
using make_zip_t       = __type_list<cuda_make_zip_iterator, thrust_make_zip_iterator>;
using make_it_t        = __type_cartesian_product<make_counting_t, make_transform_t, make_zip_t>;
TEMPLATE_LIST_TEST_CASE("iterator system and traversal propagation - any system", "[iterators]", make_it_t)
{
  using namespace cuda::std;
  __type_at_c<0, TestType> make_counting_iterator;
  __type_at_c<1, TestType> make_transform_iterator;
  __type_at_c<2, TestType> make_zip_iterator;

  auto counting_it = make_counting_iterator(0);
  STATIC_REQUIRE(is_same_v<thrust::iterator_system_t<decltype(counting_it)>, thrust::any_system_tag>);
  STATIC_REQUIRE(has_random_access_traversal<decltype(counting_it)>);

  auto transform_it = make_transform_iterator(counting_it, thrust::square<>{});
  STATIC_REQUIRE(is_same_v<thrust::iterator_system_t<decltype(transform_it)>, thrust::any_system_tag>);
  STATIC_REQUIRE(has_random_access_traversal<decltype(transform_it)>);

  [[maybe_unused]] auto zip_it = make_zip_iterator(counting_it, transform_it);
  STATIC_REQUIRE(is_same_v<thrust::iterator_system_t<decltype(zip_it)>, thrust::any_system_tag>);
  STATIC_REQUIRE(has_random_access_traversal<decltype(zip_it)>);
}

auto expected_tag(thrust::device_vector<int>) -> thrust::device_system_tag;
auto expected_tag(thrust::host_vector<int>) -> thrust::host_system_tag;
auto expected_tag(std::vector<int>) -> thrust::host_system_tag;

using vectors           = __type_list<thrust::device_vector<int>, thrust::host_vector<int>, std::vector<int>>;
using make_vec_and_it_t = __type_cartesian_product<vectors, make_transform_t, make_zip_t>;
TEMPLATE_LIST_TEST_CASE("iterator system and traversal propagation - vectors", "[iterators]", make_vec_and_it_t)
{
  using namespace cuda::std;
  __type_at_c<0, TestType> vec{};
  __type_at_c<1, TestType> make_transform_iterator;
  __type_at_c<2, TestType> make_zip_iterator;

  using tag = decltype(expected_tag(vec));

  auto vec_it = vec.begin();
  STATIC_REQUIRE(cuda::std::is_same_v<thrust::iterator_system_t<decltype(vec_it)>, tag>);
  STATIC_REQUIRE(has_random_access_traversal<decltype(vec_it)>);

  using thrust::placeholders::_1;
  auto transform_it = make_transform_iterator(vec_it, _1 + 1);
  STATIC_REQUIRE(is_same_v<thrust::iterator_system_t<decltype(transform_it)>, tag>);
  STATIC_REQUIRE(has_random_access_traversal<decltype(transform_it)>);

  [[maybe_unused]] auto zip_it = make_zip_iterator(vec_it, transform_it);
  STATIC_REQUIRE(is_same_v<thrust::iterator_system_t<decltype(zip_it)>, tag>);
  STATIC_REQUIRE(has_random_access_traversal<decltype(zip_it)>);
}

using make_vec_count_zip_it_t = __type_cartesian_product<vectors, make_counting_t, make_zip_t>;
TEMPLATE_LIST_TEST_CASE(
  "iterator system and traversal propagation - vectors and any system", "[iterators]", make_vec_count_zip_it_t)
{
  using namespace cuda::std;
  __type_at_c<0, TestType> vec{};
  __type_at_c<1, TestType> make_counting_iterator;
  __type_at_c<2, TestType> make_zip_iterator;

  using vec_tag = decltype(expected_tag(vec));

  auto vec_it = vec.begin();
  STATIC_REQUIRE(cuda::std::is_same_v<thrust::iterator_system_t<decltype(vec_it)>, vec_tag>);
  STATIC_REQUIRE(has_random_access_traversal<decltype(vec_it)>);

  auto counting_it = make_counting_iterator(0);
  STATIC_REQUIRE(cuda::std::is_same_v<thrust::iterator_system_t<decltype(counting_it)>, thrust::any_system_tag>);
  STATIC_REQUIRE(has_random_access_traversal<decltype(counting_it)>);

  [[maybe_unused]] auto zip_it = make_zip_iterator(vec_it, counting_it);
  STATIC_REQUIRE(cuda::std::is_same_v<thrust::iterator_system_t<decltype(zip_it)>, vec_tag>);
  STATIC_REQUIRE(has_random_access_traversal<decltype(zip_it)>);
}
