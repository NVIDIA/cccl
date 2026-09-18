#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <unittest/unittest.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244 4267) // possible loss of data

template <typename Iterator1, typename Iterator2>
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
__global__
#endif
  void simple_copy_on_device(Iterator1 first1, Iterator1 last1, Iterator2 first2)
{
  while (first1 != last1)
  {
    *(first2++) = *(first1++);
  }
}

template <typename Iterator1, typename Iterator2>
void simple_copy(Iterator1 first1, Iterator1 last1, Iterator2 first2)
{
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  simple_copy_on_device<<<1, 1>>>(first1, last1, first2);
#else
  simple_copy_on_device(first1, last1, first2);
#endif
}

void TestDeviceDereferenceDeviceVectorIterator()
{
  thrust::device_vector<int> input = unittest::random_integers<int>(100);
  thrust::device_vector<int> output(input.size(), 0);

  simple_copy(input.begin(), input.end(), output.begin());

  REQUIRE(input == output);
}
TEST_CASE("TestDeviceDereferenceDeviceVectorIterator", "[dereference]")
{
  TestDeviceDereferenceDeviceVectorIterator();
}

void TestDeviceDereferenceDevicePtr()
{
  thrust::device_vector<int> input = unittest::random_integers<int>(100);
  thrust::device_vector<int> output(input.size(), 0);

  const thrust::device_ptr<int> _first1 = &input[0];
  const thrust::device_ptr<int> _last1  = _first1 + static_cast<std::ptrdiff_t>(input.size());
  const thrust::device_ptr<int> _first2 = &output[0];

  simple_copy(_first1, _last1, _first2);

  REQUIRE(input == output);
}
TEST_CASE("TestDeviceDereferenceDevicePtr", "[dereference]")
{
  TestDeviceDereferenceDevicePtr();
}

void TestDeviceDereferenceTransformIterator()
{
  thrust::device_vector<int> input = unittest::random_integers<int>(100);
  thrust::device_vector<int> output(input.size(), 0);

  simple_copy(thrust::make_transform_iterator(input.begin(), ::cuda::std::identity{}),
              thrust::make_transform_iterator(input.end(), ::cuda::std::identity{}),
              output.begin());

  REQUIRE(input == output);
}
TEST_CASE("TestDeviceDereferenceTransformIterator", "[dereference]")
{
  TestDeviceDereferenceTransformIterator();
}

void TestDeviceDereferenceTransformIteratorInputConversion()
{
  thrust::device_vector<int> input = unittest::random_integers<int>(100);
  thrust::device_vector<double> output(input.size(), 0);

  simple_copy(thrust::make_transform_iterator(input.begin(), ::cuda::std::identity{}),
              thrust::make_transform_iterator(input.end(), ::cuda::std::identity{}),
              output.begin());

  REQUIRE(input == output);
}
TEST_CASE("TestDeviceDereferenceTransformIteratorInputConversion", "[dereference]")
{
  TestDeviceDereferenceTransformIteratorInputConversion();
}

void TestDeviceDereferenceTransformIteratorOutputConversion()
{
  thrust::device_vector<int> input = unittest::random_integers<int>(100);
  thrust::device_vector<double> output(input.size(), 0);

  simple_copy(thrust::make_transform_iterator(input.begin(), ::cuda::std::identity{}),
              thrust::make_transform_iterator(input.end(), ::cuda::std::identity{}),
              output.begin());

  REQUIRE(input == output);
}
TEST_CASE("TestDeviceDereferenceTransformIteratorOutputConversion", "[dereference]")
{
  TestDeviceDereferenceTransformIteratorOutputConversion();
}

void TestDeviceDereferenceCountingIterator()
{
  const thrust::counting_iterator<int> first(1);
  const thrust::counting_iterator<int> last(6);

  thrust::device_vector<int> output(5);

  simple_copy(first, last, output.begin());

  const thrust::device_vector<int> ref{1, 2, 3, 4, 5};
  REQUIRE(output == ref);
}
TEST_CASE("TestDeviceDereferenceCountingIterator", "[dereference]")
{
  TestDeviceDereferenceCountingIterator();
}

void TestDeviceDereferenceTransformedCountingIterator()
{
  const thrust::counting_iterator<int> first(1);
  const thrust::counting_iterator<int> last(6);

  thrust::device_vector<int> output(5);

  simple_copy(thrust::make_transform_iterator(first, ::cuda::std::negate<int>()),
              thrust::make_transform_iterator(last, ::cuda::std::negate<int>()),
              output.begin());

  const thrust::device_vector<int> ref{-1, -2, -3, -4, -5};
  REQUIRE(output == ref);
}
TEST_CASE("TestDeviceDereferenceTransformedCountingIterator", "[dereference]")
{
  TestDeviceDereferenceTransformedCountingIterator();
}

_CCCL_DIAG_POP
