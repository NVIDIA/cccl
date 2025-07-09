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

TEST_CASE("constant_iterator", "[iterators]")
{
  { // device system
    thrust::device_vector<int> vec{1, 2, 3, 4};
    thrust::copy(cuda::constant_iterator{42, 0}, cuda::constant_iterator{42, 4}, vec.begin());
  }

  { // host system
    thrust::host_vector<int> vec{1, 2, 3, 4};
    thrust::copy(cuda::constant_iterator{42, 0}, cuda::constant_iterator{42, 4}, vec.begin());
  }

  { // plain std::vector
    std::vector<int> vec{1, 2, 3, 4};
    thrust::copy(cuda::constant_iterator{42, 0}, cuda::constant_iterator{42, 4}, vec.begin());
  }
}

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

TEST_CASE("permutation_iterator", "[iterators]")
{
  { // device system
    thrust::device_vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8, 9};
    thrust::device_vector<int> off{5, 2, 7, 0};
    thrust::device_vector<int> res{-1, -1, -1, -1, -1};
    thrust::copy(cuda::permutation_iterator{vec.begin(), off.begin()},
                 cuda::permutation_iterator{vec.begin(), off.end()},
                 res.begin());
    CHECK(res == thrust::device_vector<int>{6, 3, 8, 1, -1});
  }

  { // host system
    thrust::host_vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8, 9};
    thrust::host_vector<int> off{5, 2, 7, 0};
    thrust::host_vector<int> res{-1, -1, -1, -1, -1};
    thrust::copy(cuda::permutation_iterator{vec.begin(), off.begin()},
                 cuda::permutation_iterator{vec.begin(), off.end()},
                 res.begin());
    CHECK(res == thrust::host_vector<int>{6, 3, 8, 1, -1});
  }

  { // plain std::vector
    std::vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int> off{5, 2, 7, 0};
    std::vector<int> res{-1, -1, -1, -1, -1};
    thrust::copy(cuda::permutation_iterator{vec.begin(), off.begin()},
                 cuda::permutation_iterator{vec.begin(), off.end()},
                 res.begin());
    CHECK(res == std::vector<int>{6, 3, 8, 1, -1});
  }
}

TEST_CASE("strided_iterator", "[iterators]")
{
  auto discard = cuda::discard_iterator{};
  { // device system
    thrust::device_vector<int> vec{1, 2, 3, 4, 5, 6};
    thrust::copy(cuda::strided_iterator{vec.begin(), 2}, cuda::strided_iterator{vec.end(), 2}, discard);
  }

  { // host system
    thrust::host_vector<int> vec{1, 2, 3, 4, 5, 6};
    thrust::copy(cuda::strided_iterator{vec.begin(), 2}, cuda::strided_iterator{vec.end(), 2}, discard);
  }

  { // plain std::vector
    std::vector<int> vec{1, 2, 3, 4, 5, 6};
    thrust::copy(cuda::strided_iterator{vec.begin(), 2}, cuda::strided_iterator{vec.end(), 2}, discard);
  }
}

struct is_equal_index
{
  _CCCL_HOST_DEVICE constexpr void
  operator()([[maybe_unused]] const int index, [[maybe_unused]] const int expected) const noexcept
  {
    _CCCL_VERIFY(index == expected, "should have right value");
  }
};

TEST_CASE("tabulate_output_iterator", "[iterators]")
{
  { // device system
    thrust::device_vector<int> vec{5, 6, 7, 8, 9};
    thrust::copy(vec.begin(), vec.end(), cuda::make_tabulate_output_iterator(is_equal_index{}, 5));
  }

  { // host system
    thrust::host_vector<int> vec{5, 6, 7, 8, 9};
    thrust::copy(vec.begin(), vec.end(), cuda::make_tabulate_output_iterator(is_equal_index{}, 5));
  }

  { // plain std::vector
    std::vector<int> vec{5, 6, 7, 8, 9};
    thrust::copy(vec.begin(), vec.end(), cuda::make_tabulate_output_iterator(is_equal_index{}, 5));
  }
}

struct plus_one
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr int operator()(const int val) const noexcept
  {
    return val + 1;
  }
};

TEST_CASE("transform_output_iterator", "[iterators]")
{
  { // device system
    thrust::device_vector<int> vec{-1, -1, -1, -1, -1};
    thrust::copy(cuda::counting_iterator{0},
                 cuda::counting_iterator{5},
                 cuda::make_transform_output_iterator(vec.begin(), plus_one{}));
    thrust::device_vector<int> expected{1, 2, 3, 4, 5};
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }

  { // host system
    thrust::host_vector<int> vec{-1, -1, -1, -1, -1};
    thrust::copy(cuda::counting_iterator{0},
                 cuda::counting_iterator{5},
                 cuda::make_transform_output_iterator(vec.begin(), plus_one{}));
    thrust::host_vector<int> expected{1, 2, 3, 4, 5};
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }

  { // plain std::vector
    std::vector<int> vec{-1, -1, -1, -1, -1};
    thrust::copy(cuda::counting_iterator{0},
                 cuda::counting_iterator{5},
                 cuda::make_transform_output_iterator(vec.begin(), plus_one{}));
    std::vector<int> expected{1, 2, 3, 4, 5};
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));
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
