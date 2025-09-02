#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include "catch2_test_helper.h"
#include <unittest/unittest.h>

constexpr size_t num_samples = 10000;

template <typename Vector, typename U>
struct rebind_vector;

template <typename T, typename U, typename Allocator>
struct rebind_vector<thrust::host_vector<T, Allocator>, U>
{
  using type = thrust::host_vector<U, typename thrust::detail::allocator_traits<Allocator>::template rebind_alloc<U>>;
};

template <typename T, typename U, typename Allocator>
struct rebind_vector<thrust::device_vector<T, Allocator>, U>
{
  using type = thrust::device_vector<U, typename Allocator::template rebind<U>::other>;
};

template <typename T, typename U, typename Allocator>
struct rebind_vector<thrust::universal_vector<T, Allocator>, U>
{
  using type = thrust::universal_vector<U, typename Allocator::template rebind<U>::other>;
};

TEMPLATE_LIST_TEST_CASE("LogicalAndOr", "[functional]", vector_list)
{
  using Vector      = TestType;
  using T           = typename Vector::value_type;
  using bool_vector = typename rebind_vector<Vector, bool>::type;
  Vector lhs        = unittest::random_samples<T>(num_samples);
  Vector rhs        = unittest::random_samples<T>(num_samples);

  using namespace thrust::placeholders;
  bool_vector reference(lhs.size());
  bool_vector result(lhs.size());

  SECTION("and")
  {
    thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), reference.begin(), ::cuda::std::logical_and<T>{});
    thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), _1 && _2);
    CHECK(reference == result);
  }
  SECTION("or")
  {
    thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), reference.begin(), ::cuda::std::logical_or<T>{});
    thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), _1 || _2);
    CHECK(reference == result);
  }
}

TEMPLATE_LIST_TEST_CASE("LogicalNot", "[functional]", integral_vector_list)
{
  using Vector      = TestType;
  using T           = typename Vector::value_type;
  using bool_vector = typename rebind_vector<Vector, bool>::type;
  Vector input      = unittest::random_samples<T>(num_samples);

  if (input.size() > 0)
  {
    // produce at least one true in the output
    input[0] = T(0);
  }

  using namespace thrust::placeholders;
  bool_vector reference(input.size());
  bool_vector result(input.size());

  thrust::transform(input.begin(), input.end(), reference.begin(), ::cuda::std::logical_not<T>{});
  thrust::transform(input.begin(), input.end(), result.begin(), !_1);
  ASSERT_EQUAL(reference, result);
}
