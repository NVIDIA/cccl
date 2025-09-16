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
    thrust::device_vector<int> expected{42, 42, 42, 42};
    thrust::copy(cuda::constant_iterator{42, 0}, cuda::constant_iterator{42, 4}, vec.begin());
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }

  { // host system
    thrust::host_vector<int> vec{1, 2, 3, 4};
    thrust::host_vector<int> expected{42, 42, 42, 42};
    thrust::copy(cuda::constant_iterator{42, 0}, cuda::constant_iterator{42, 4}, vec.begin());
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }

  { // plain std::vector
    std::vector<int> vec{1, 2, 3, 4};
    std::vector<int> expected{42, 42, 42, 42};
    thrust::copy(cuda::constant_iterator{42, 0}, cuda::constant_iterator{42, 4}, vec.begin());
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }
}

TEST_CASE("counting_iterator", "[iterators]")
{
  { // device system
    thrust::device_vector<int> vec{-1, -1, -1, -1};
    thrust::device_vector<int> expected{0, 1, 2, 3};
    thrust::copy(cuda::counting_iterator{0}, cuda::counting_iterator{4}, vec.begin());
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }

  { // host system
    thrust::host_vector<int> vec{-1, -1, -1, -1};
    thrust::host_vector<int> expected{0, 1, 2, 3};
    thrust::copy(cuda::counting_iterator{0}, cuda::counting_iterator{4}, vec.begin());
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }

  { // plain std::vector
    std::vector<int> vec{-1, -1, -1, -1};
    std::vector<int> expected{0, 1, 2, 3};
    thrust::copy(cuda::counting_iterator{0}, cuda::counting_iterator{4}, vec.begin());
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));
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

TEST_CASE("reverse_iterator", "[iterators]")
{
  { // device system
    thrust::device_vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8, 9};
    thrust::device_vector<int> expected{9, 8, 7, 6, 5, 4, 3, 2, 1};
    CHECK(thrust::equal(
      cuda::std::reverse_iterator{vec.end()}, cuda::std::reverse_iterator{vec.begin()}, expected.begin()));
  }

  { // host system
    thrust::host_vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8, 9};
    thrust::host_vector<int> expected{9, 8, 7, 6, 5, 4, 3, 2, 1};
    CHECK(thrust::equal(
      cuda::std::reverse_iterator{vec.end()}, cuda::std::reverse_iterator{vec.begin()}, expected.begin()));
  }

  { // plain std::vector
    std::vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int> expected{9, 8, 7, 6, 5, 4, 3, 2, 1};
    CHECK(thrust::equal(
      cuda::std::reverse_iterator{vec.end()}, cuda::std::reverse_iterator{vec.begin()}, expected.begin()));
  }
}

struct fake_bijection
{
  using index_type = ::cuda::std::uint32_t;

  constexpr fake_bijection() = default;

  [[nodiscard]] _CCCL_HOST_DEVICE constexpr index_type size() const noexcept
  {
    return 5;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE constexpr index_type operator()(index_type n) const noexcept
  {
    return __random_indices[n];
  }

  ::cuda::std::uint32_t __random_indices[5] = {4, 1, 2, 0, 3};
};

TEST_CASE("shuffle_iterator", "[iterators]")
{
  { // device system
    thrust::device_vector<int> vec{1, 2, 3, 4};
    thrust::device_vector<int> expected{4, 1, 2, 0};
    thrust::copy(cuda::shuffle_iterator{fake_bijection{}, 0}, cuda::shuffle_iterator{fake_bijection{}, 4}, vec.begin());
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }

  { // host system
    thrust::host_vector<int> vec{1, 2, 3, 4};
    thrust::host_vector<int> expected{4, 1, 2, 0};
    thrust::copy(cuda::shuffle_iterator{fake_bijection{}, 0}, cuda::shuffle_iterator{fake_bijection{}, 4}, vec.begin());
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }

  { // plain std::vector
    std::vector<int> vec{1, 2, 3, 4};
    std::vector<int> expected{4, 1, 2, 0};
    thrust::copy(cuda::shuffle_iterator{fake_bijection{}, 0}, cuda::shuffle_iterator{fake_bijection{}, 4}, vec.begin());
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }
}

TEST_CASE("strided_iterator", "[iterators]")
{
  { // device system
    thrust::device_vector<int> vec{-1, -1, -1, -1};
    thrust::device_vector<int> expected{0, 2, 4, 6};
    thrust::copy(cuda::strided_iterator{cuda::counting_iterator{0}, 2},
                 cuda::strided_iterator{cuda::counting_iterator{8}, 2},
                 vec.begin());
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }

  { // host system
    thrust::host_vector<int> vec{-1, -1, -1, -1};
    thrust::host_vector<int> expected{0, 2, 4, 6};
    thrust::copy(cuda::strided_iterator{cuda::counting_iterator{0}, 2},
                 cuda::strided_iterator{cuda::counting_iterator{8}, 2},
                 vec.begin());
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }

  { // plain std::vector
    std::vector<int> vec{-1, -1, -1, -1};
    std::vector<int> expected{0, 2, 4, 6};
    thrust::copy(cuda::strided_iterator{cuda::counting_iterator{0}, 2},
                 cuda::strided_iterator{cuda::counting_iterator{8}, 2},
                 vec.begin());
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));
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

TEST_CASE("transform_input_output_iterator", "[iterators]")
{
  { // device system
    thrust::device_vector<int> vec{-1, -1, -1, -1, -1};
    auto iter = cuda::transform_input_output_iterator(vec.begin(), plus_one{}, plus_one{});
    thrust::copy(cuda::counting_iterator{3}, cuda::counting_iterator{8}, iter);

    // Ensure we did write the right output, sequence starts at 3 + 1 == 4
    thrust::device_vector<int> expected{4, 5, 6, 7, 8};
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));

    // Ensure we did read the right input, output starts 4 + 1 == 5
    thrust::copy(iter, iter + 5, cuda::make_tabulate_output_iterator(is_equal_index{}, 5));
  }

  { // host system
    thrust::host_vector<int> vec{-1, -1, -1, -1, -1};
    auto iter = cuda::transform_input_output_iterator(vec.begin(), plus_one{}, plus_one{});
    thrust::copy(cuda::counting_iterator{3}, cuda::counting_iterator{8}, iter);

    // Ensure we did write the right output, sequence starts at 3 + 1 == 4
    thrust::host_vector<int> expected{4, 5, 6, 7, 8};
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));

    // Ensure we did read the right input, output starts 4 + 1 == 5
    thrust::copy(iter, iter + 5, cuda::make_tabulate_output_iterator(is_equal_index{}, 5));
  }

  { // plain std::vector
    std::vector<int> vec{-1, -1, -1, -1, -1};
    auto iter = cuda::transform_input_output_iterator(vec.begin(), plus_one{}, plus_one{});
    thrust::copy(cuda::counting_iterator{3}, cuda::counting_iterator{8}, iter);

    // Ensure we did write the right output, sequence starts at 3 + 1 == 4
    std::vector<int> expected{4, 5, 6, 7, 8};
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));

    // Ensure we did read the right input, output starts 4 + 1 == 5
    thrust::copy(iter, iter + 5, cuda::make_tabulate_output_iterator(is_equal_index{}, 5));
  }
}

TEST_CASE("transform_output_iterator", "[iterators]")
{
  { // device system
    thrust::device_vector<int> vec{-1, -1, -1, -1, -1};
    thrust::device_vector<int> expected{1, 2, 3, 4, 5};
    thrust::copy(cuda::counting_iterator{0},
                 cuda::counting_iterator{5},
                 cuda::make_transform_output_iterator(vec.begin(), plus_one{}));
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }

  { // host system
    thrust::host_vector<int> vec{-1, -1, -1, -1, -1};
    thrust::host_vector<int> expected{1, 2, 3, 4, 5};
    thrust::copy(cuda::counting_iterator{0},
                 cuda::counting_iterator{5},
                 cuda::make_transform_output_iterator(vec.begin(), plus_one{}));
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }

  { // plain std::vector
    std::vector<int> vec{-1, -1, -1, -1, -1};
    std::vector<int> expected{1, 2, 3, 4, 5};
    thrust::copy(cuda::counting_iterator{0},
                 cuda::counting_iterator{5},
                 cuda::make_transform_output_iterator(vec.begin(), plus_one{}));
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }
}

TEST_CASE("transform_iterator", "[iterators]")
{
  { // device system
    thrust::device_vector<int> vec{-1, -1, -1, -1};
    thrust::device_vector<int> expected{1, 2, 3, 4};
    thrust::copy(cuda::transform_iterator(cuda::counting_iterator{0}, plus_one{}),
                 cuda::transform_iterator(cuda::counting_iterator{4}, plus_one{}),
                 vec.begin());
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }

  { // host system
    thrust::host_vector<int> vec{-1, -1, -1, -1};
    thrust::host_vector<int> expected{1, 2, 3, 4};
    thrust::copy(cuda::transform_iterator(cuda::counting_iterator{0}, plus_one{}),
                 cuda::transform_iterator(cuda::counting_iterator{4}, plus_one{}),
                 vec.begin());
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }

  { // plain std::vector
    std::vector<int> vec{-1, -1, -1, -1};
    std::vector<int> expected{1, 2, 3, 4};
    thrust::copy(cuda::transform_iterator(cuda::counting_iterator{0}, plus_one{}),
                 cuda::transform_iterator(cuda::counting_iterator{4}, plus_one{}),
                 vec.begin());
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }
}

TEST_CASE("zip_iterator", "[iterators]")
{
  cuda::zip_function<cuda::std::plus<void>> fun{};
  { // device system
    thrust::device_vector<int> vec{-1, -1, -1, -1};
    auto iter = cuda::make_transform_iterator(cuda::make_zip_iterator(vec.begin(), cuda::counting_iterator{4}), fun);
    thrust::copy(iter, iter + 4, vec.begin());
    thrust::device_vector<int> expected{3, 4, 5, 6};
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }

  { // host system
    thrust::host_vector<int> vec{-1, -1, -1, -1};
    auto iter = cuda::make_transform_iterator(cuda::make_zip_iterator(vec.begin(), cuda::counting_iterator{4}), fun);
    thrust::copy(iter, iter + 4, vec.begin());
    thrust::host_vector<int> expected{3, 4, 5, 6};
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }

  { // plain std::vector
    std::vector<int> vec{-1, -1, -1, -1};
    auto iter = cuda::make_transform_iterator(cuda::make_zip_iterator(vec.begin(), cuda::counting_iterator{4}), fun);
    thrust::copy(iter, iter + 4, vec.begin());
    std::vector<int> expected{3, 4, 5, 6};
    CHECK(thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }
}
