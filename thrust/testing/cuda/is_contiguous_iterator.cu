#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/unwrap_contiguous_iterator.h>

#include <cuda/experimental/container.cuh>
#include <cuda/experimental/memory_resource.cuh>
#include <cuda/experimental/stream.cuh>

#include <unittest/unittest.h>

namespace cudax = cuda::experimental;

_CCCL_HOST void test_is_contiguous_iterator_async_buffer()
{
  static_assert(thrust::is_contiguous_iterator<cudax::async_host_buffer<int>::iterator>::value);
  static_assert(thrust::is_contiguous_iterator<cudax::async_host_buffer<int>::const_iterator>::value);

  static_assert(thrust::is_contiguous_iterator<cudax::async_device_buffer<int>::iterator>::value);
  static_assert(thrust::is_contiguous_iterator<cudax::async_device_buffer<int>::const_iterator>::value);
}
DECLARE_UNITTEST(test_is_contiguous_iterator_async_buffer);

void test_try_unwrap_contiguous_iterator_async_buffer()
{
  using ::cuda::std::is_same_v;

  static_assert(is_same_v<thrust::try_unwrap_contiguous_iterator_t<cudax::async_host_buffer<int>::iterator>, int*>);
  static_assert(is_same_v<decltype(thrust::try_unwrap_contiguous_iterator(
                            ::cuda::std::declval<cudax::async_host_buffer<int>::iterator>())),
                          int*>);

  static_assert(is_same_v<thrust::try_unwrap_contiguous_iterator_t<cudax::async_device_buffer<int>::iterator>, int*>);
  static_assert(is_same_v<decltype(thrust::try_unwrap_contiguous_iterator(
                            ::cuda::std::declval<cudax::async_device_buffer<int>::iterator>())),
                          int*>);
}
DECLARE_UNITTEST(test_try_unwrap_contiguous_iterator_async_buffer);
