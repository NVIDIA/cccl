#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/iterator>

#include <vector>

#include "catch2_test_helper.h"

TEST_CASE("discard_iterator", "[iterators]")
{
  auto discard = cuda::discard_iterator{};
  { // device system
    thrust::device_vector<int> vec{1, 2, 3, 4};
    thrust::copy(vec.begin(), vec.end(), discard);
  }

  { // host system
    thrust::host_vector<int> vec{1, 2, 3, 4};
    thrust::copy(vec.begin(), vec.end(), discard);
  }

  { // plain std::vector
    std::vector<int> vec{1, 2, 3, 4};
    thrust::copy(vec.begin(), vec.end(), discard);
  }
}

struct plus_one
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr int operator()(const int val) const noexcept
  {
    return val + 1;
  }
};

TEST_CASE("counting_iterator", "[iterators]")
{
  { // device system
    thrust::device_vector<int> vec{1, 2, 3, 4};
    thrust::copy(cuda::counting_iterator{0}, cuda::counting_iterator{4}, vec.begin());
  }

  { // host system
    thrust::host_vector<int> vec{1, 2, 3, 4};
    thrust::copy(cuda::counting_iterator{0}, cuda::counting_iterator{4}, vec.begin());
  }

  { // plain std::vector
    std::vector<int> vec{1, 2, 3, 4};
    thrust::copy(cuda::counting_iterator{0}, cuda::counting_iterator{4}, vec.begin());
  }
}

TEST_CASE("transform_iterator", "[iterators]")
{
  auto discard = cuda::discard_iterator{};
  { // device system
    thrust::device_vector<int> vec{1, 2, 3, 4};
    thrust::copy(cuda::make_transform_iterator(vec.begin(), plus_one{}),
                 cuda::make_transform_iterator(vec.end(), plus_one{}),
                 discard);
  }

  { // host system
    thrust::host_vector<int> vec{1, 2, 3, 4};
    thrust::copy(cuda::make_transform_iterator(vec.begin(), plus_one{}),
                 cuda::make_transform_iterator(vec.end(), plus_one{}),
                 discard);
  }

  { // plain std::vector
    std::vector<int> vec{1, 2, 3, 4};
    thrust::copy(cuda::make_transform_iterator(vec.begin(), plus_one{}),
                 cuda::make_transform_iterator(vec.end(), plus_one{}),
                 discard);
  }
}
