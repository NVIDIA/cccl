#include <thrust/transform.h>

#include "catch2_test_helper.h"

TEST_CASE("Base", "[transform_n]")
{
  using namespace thrust::placeholders;

  thrust::device_vector vec{1, 2, 3, 4, 5};
  thrust::device_vector stencil{true, false, true};

  {
    thrust::device_vector result{0, 0, 0};
    thrust::transform_n(vec.begin(), 3, result.begin(), _1 * 2);
    CHECK(result == (thrust::device_vector{2, 4, 6}));
  }
  {
    thrust::device_vector result{0, 0, 0};
    thrust::transform_n(vec.begin(), 3, vec.begin(), result.begin(), _1 * _2);
    CHECK(result == (thrust::device_vector{1, 4, 9}));
  }
  {
    thrust::device_vector result{0, 0, 0};
    thrust::transform_if_n(vec.begin(), 3, result.begin(), _1 * 2, _1 > 1);
    CHECK(result == (thrust::device_vector{0, 4, 6}));
  }
  {
    thrust::device_vector result{0, 0, 0};
    thrust::transform_if_n(vec.begin(), 3, stencil.begin(), result.begin(), _1 * 2, _1);
    CHECK(result == (thrust::device_vector{2, 0, 6}));
  }
  {
    thrust::device_vector result{0, 0, 0};
    thrust::transform_if_n(vec.begin(), 3, vec.begin(), stencil.begin(), result.begin(), _1 * _2, _1);
    CHECK(result == (thrust::device_vector{1, 0, 9}));
  }
}
