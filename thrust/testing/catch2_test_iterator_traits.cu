#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

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

TEST_CASE("iterator_category_to_system", "[iterators]")
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

TEST_CASE("iterator system propagation - any system", "[iterators]")
{
  [[maybe_unused]] auto counting_it = thrust::make_counting_iterator(0);
  STATIC_REQUIRE(cuda::std::is_same_v<thrust::iterator_system_t<decltype(counting_it)>, thrust::any_system_tag>);

  [[maybe_unused]] auto transform_it = thrust::make_transform_iterator(counting_it, thrust::square<>{});
  STATIC_REQUIRE(cuda::std::is_same_v<thrust::iterator_system_t<decltype(transform_it)>, thrust::any_system_tag>);

  [[maybe_unused]] auto zip_it = thrust::make_zip_iterator(counting_it, transform_it);
  STATIC_REQUIRE(cuda::std::is_same_v<thrust::iterator_system_t<decltype(zip_it)>, thrust::any_system_tag>);
}

TEST_CASE("iterator system propagation - host system", "[iterators]")
{
  [[maybe_unused]] auto h_vec    = thrust::host_vector<int>{};
  [[maybe_unused]] auto h_vec_it = h_vec.begin();
  STATIC_REQUIRE(cuda::std::is_same_v<thrust::iterator_system_t<decltype(h_vec_it)>, thrust::host_system_tag>);

  [[maybe_unused]] auto transform_it = thrust::make_transform_iterator(h_vec_it, thrust::square<>{});
  STATIC_REQUIRE(cuda::std::is_same_v<thrust::iterator_system_t<decltype(transform_it)>, thrust::host_system_tag>);

  [[maybe_unused]] auto zip_it = thrust::make_zip_iterator(h_vec_it, transform_it);
  STATIC_REQUIRE(cuda::std::is_same_v<thrust::iterator_system_t<decltype(zip_it)>, thrust::host_system_tag>);
}

TEST_CASE("iterator system propagation - device system", "[iterators]")
{
  [[maybe_unused]] auto d_vec    = thrust::device_vector<int>{};
  [[maybe_unused]] auto d_vec_it = d_vec.begin();
  STATIC_REQUIRE(cuda::std::is_same_v<thrust::iterator_system_t<decltype(d_vec_it)>, thrust::device_system_tag>);

  [[maybe_unused]] auto transform_it = thrust::make_transform_iterator(d_vec_it, thrust::square<>{});
  STATIC_REQUIRE(cuda::std::is_same_v<thrust::iterator_system_t<decltype(transform_it)>, thrust::device_system_tag>);

  [[maybe_unused]] auto zip_it = thrust::make_zip_iterator(d_vec_it, transform_it);
  STATIC_REQUIRE(cuda::std::is_same_v<thrust::iterator_system_t<decltype(zip_it)>, thrust::device_system_tag>);
}

TEST_CASE("iterator system propagation - any and device system", "[iterators]")
{
  [[maybe_unused]] auto d_vec    = thrust::device_vector<int>{};
  [[maybe_unused]] auto d_vec_it = d_vec.begin();
  STATIC_REQUIRE(cuda::std::is_same_v<thrust::iterator_system_t<decltype(d_vec_it)>, thrust::device_system_tag>);

  [[maybe_unused]] auto counting_it = thrust::make_counting_iterator(0);
  STATIC_REQUIRE(cuda::std::is_same_v<thrust::iterator_system_t<decltype(counting_it)>, thrust::any_system_tag>);

  [[maybe_unused]] auto zip_it = thrust::make_zip_iterator(d_vec_it, counting_it);
  STATIC_REQUIRE(cuda::std::is_same_v<thrust::iterator_system_t<decltype(zip_it)>, thrust::device_system_tag>);
}
