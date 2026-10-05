#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>

#include <algorithm>

#include <unittest/unittest.h>

template <class Vector>
void test_scatter_simple()
{
  Vector map{6, 3, 1, 7, 2};
  Vector src{0, 1, 2, 3, 4};
  Vector dst(8, 0);

  thrust::scatter(src.begin(), src.end(), map.begin(), dst.begin());

  Vector ref{0, 2, 4, 1, 0, 0, 0, 3};
  REQUIRE(dst == ref);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(test_scatter_simple);

template <typename InputIterator1, typename InputIterator2, typename RandomAccessIterator>
void scatter(my_system& system, InputIterator1, InputIterator1, InputIterator2, RandomAccessIterator)
{
  system.validate_dispatch();
}

TEST_CASE("TestScatterDispatchExplicit", "[scatter]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::scatter(sys, vec.begin(), vec.begin(), vec.begin(), vec.begin());

  REQUIRE(sys.is_valid());
}

template <typename InputIterator1, typename InputIterator2, typename RandomAccessIterator>
void scatter(my_tag, InputIterator1, InputIterator1, InputIterator2, RandomAccessIterator output)
{
  *output = 13;
}

TEST_CASE("TestScatterDispatchImplicit", "[scatter]")
{
  thrust::device_vector<int> vec(1);

  thrust::scatter(thrust::retag<my_tag>(vec.begin()),
                  thrust::retag<my_tag>(vec.begin()),
                  thrust::retag<my_tag>(vec.begin()),
                  thrust::retag<my_tag>(vec.begin()));

  REQUIRE(13 == vec.front());
}

template <typename T>
void test_scatter(const size_t n)
{
  const size_t output_size = std::min((size_t) 10, 2 * n);

  thrust::host_vector<T> h_input(n, (T) 1);
  thrust::device_vector<T> d_input(n, (T) 1);

  thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);

  for (size_t i = 0; i < n; i++)
  {
    h_map[i] = h_map[i] % output_size;
  }

  thrust::device_vector<unsigned int> d_map = h_map;

  thrust::host_vector<T> h_output(output_size, (T) 0);
  thrust::device_vector<T> d_output(output_size, (T) 0);

  thrust::scatter(h_input.begin(), h_input.end(), h_map.begin(), h_output.begin());
  thrust::scatter(d_input.begin(), d_input.end(), d_map.begin(), d_output.begin());

  REQUIRE(h_output == d_output);
}
DECLARE_VARIABLE_UNITTEST(test_scatter);

template <typename T>
void test_scatter_to_discard_iterator(const size_t n)
{
  const size_t output_size = std::min((size_t) 10, 2 * n);

  thrust::host_vector<T> h_input(n, (T) 1);
  thrust::device_vector<T> d_input(n, (T) 1);

  thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);

  for (size_t i = 0; i < n; i++)
  {
    h_map[i] = h_map[i] % output_size;
  }

  thrust::device_vector<unsigned int> d_map = h_map;

  thrust::scatter(h_input.begin(), h_input.end(), h_map.begin(), thrust::make_discard_iterator());
  thrust::scatter(d_input.begin(), d_input.end(), d_map.begin(), thrust::make_discard_iterator());

  // there's nothing to check -- just make sure it compiles
}
DECLARE_VARIABLE_UNITTEST(test_scatter_to_discard_iterator);

template <class Vector>
void test_scatter_if_simple()
{
  Vector flg{0, 1, 0, 1, 0};
  Vector map{6, 3, 1, 7, 2};
  Vector src{0, 1, 2, 3, 4};
  Vector dst(8, 0);

  thrust::scatter_if(src.begin(), src.end(), map.begin(), flg.begin(), dst.begin());

  Vector ref{0, 0, 0, 1, 0, 0, 0, 3};
  REQUIRE(dst == ref);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(test_scatter_if_simple);

template <typename InputIterator1, typename InputIterator2, typename InputIterator3, typename RandomAccessIterator>
void scatter_if(my_system& system, InputIterator1, InputIterator1, InputIterator2, InputIterator3, RandomAccessIterator)
{
  system.validate_dispatch();
}

TEST_CASE("TestScatterIfDispatchExplicit", "[scatter]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::scatter_if(sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), vec.begin());

  REQUIRE(sys.is_valid());
}

template <typename InputIterator1, typename InputIterator2, typename InputIterator3, typename RandomAccessIterator>
void scatter_if(my_tag, InputIterator1, InputIterator1, InputIterator2, InputIterator3, RandomAccessIterator output)
{
  *output = 13;
}

TEST_CASE("TestScatterIfDispatchImplicit", "[scatter]")
{
  thrust::device_vector<int> vec(1);

  thrust::scatter_if(
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()));

  REQUIRE(13 == vec.front());
}

template <typename T>
class is_even_scatter_if
{
public:
  _CCCL_HOST_DEVICE bool operator()(const T i) const
  {
    return (i % 2) == 0;
  }
};

template <typename T>
void test_scatter_if(const size_t n)
{
  const size_t output_size = std::min((size_t) 10, 2 * n);

  thrust::host_vector<T> h_input(n, (T) 1);
  thrust::device_vector<T> d_input(n, (T) 1);

  thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);

  for (size_t i = 0; i < n; i++)
  {
    h_map[i] = h_map[i] % output_size;
  }

  thrust::device_vector<unsigned int> d_map = h_map;

  thrust::host_vector<T> h_output(output_size, (T) 0);
  thrust::device_vector<T> d_output(output_size, (T) 0);

  thrust::scatter_if(
    h_input.begin(), h_input.end(), h_map.begin(), h_map.begin(), h_output.begin(), is_even_scatter_if<unsigned int>());
  thrust::scatter_if(
    d_input.begin(), d_input.end(), d_map.begin(), d_map.begin(), d_output.begin(), is_even_scatter_if<unsigned int>());

  REQUIRE(h_output == d_output);
}
DECLARE_VARIABLE_UNITTEST(test_scatter_if);

template <typename T>
void test_scatter_if_to_discard_iterator(const size_t n)
{
  const size_t output_size = std::min((size_t) 10, 2 * n);

  thrust::host_vector<T> h_input(n, (T) 1);
  thrust::device_vector<T> d_input(n, (T) 1);

  thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);

  for (size_t i = 0; i < n; i++)
  {
    h_map[i] = h_map[i] % output_size;
  }

  thrust::device_vector<unsigned int> d_map = h_map;

  thrust::scatter_if(
    h_input.begin(),
    h_input.end(),
    h_map.begin(),
    h_map.begin(),
    thrust::make_discard_iterator(),
    is_even_scatter_if<unsigned int>());
  thrust::scatter_if(
    d_input.begin(),
    d_input.end(),
    d_map.begin(),
    d_map.begin(),
    thrust::make_discard_iterator(),
    is_even_scatter_if<unsigned int>());
}
DECLARE_VARIABLE_UNITTEST(test_scatter_if_to_discard_iterator);

template <typename Vector>
void test_scatter_counting_iterator()
{
  Vector source(10);
  thrust::sequence(source.begin(), source.end(), 0);

  Vector map(10);
  thrust::sequence(map.begin(), map.end(), 0);

  Vector output(10);

  // source has any_system_tag
  thrust::fill(output.begin(), output.end(), 0);
  thrust::scatter(thrust::make_counting_iterator(0), thrust::make_counting_iterator(10), map.begin(), output.begin());

  REQUIRE(output == map);

  // map has any_system_tag
  thrust::fill(output.begin(), output.end(), 0);
  thrust::scatter(source.begin(), source.end(), thrust::make_counting_iterator(0), output.begin());

  REQUIRE(output == map);

  // source and map have any_system_tag
  thrust::fill(output.begin(), output.end(), 0);
  thrust::scatter(thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(10),
                  thrust::make_counting_iterator(0),
                  output.begin());

  REQUIRE(output == map);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(test_scatter_counting_iterator);

template <typename Vector>
void test_scatter_if_counting_iterator()
{
  Vector source(10);
  thrust::sequence(source.begin(), source.end(), 0);

  Vector map(10);
  thrust::sequence(map.begin(), map.end(), 0);

  Vector stencil(10, 1);

  Vector output(10);

  // source has any_system_tag
  thrust::fill(output.begin(), output.end(), 0);
  thrust::scatter_if(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(10), map.begin(), stencil.begin(), output.begin());

  REQUIRE(output == map);

  // map has any_system_tag
  thrust::fill(output.begin(), output.end(), 0);
  thrust::scatter_if(source.begin(), source.end(), thrust::make_counting_iterator(0), stencil.begin(), output.begin());

  REQUIRE(output == map);

  // source and map have any_system_tag
  thrust::fill(output.begin(), output.end(), 0);
  thrust::scatter_if(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(10),
    thrust::make_counting_iterator(0),
    stencil.begin(),
    output.begin());

  REQUIRE(output == map);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(test_scatter_if_counting_iterator);
