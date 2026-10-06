#define CCCL_IGNORE_DEPRECATED_API

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/shuffle_iterator.h>
#include <thrust/iterator/strided_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <cuda/__iterator/is_synthesizing_iterator.h>
#include <cuda/std/tuple>

#include <unittest/unittest.h>

struct Identity
{
  template <class T>
  _CCCL_HOST_DEVICE T operator()(T value) const
  {
    return value;
  }
};

template <class Iter, bool Expected>
void check_is_synthesizing()
{
  STATIC_REQUIRE(::cuda::__is_synthesizing_iterator_v<Iter> == Expected);
  STATIC_REQUIRE(::cuda::__is_synthesizing_iterator_v<const Iter> == Expected);
  STATIC_REQUIRE(::cuda::__is_synthesizing_iterator_v<volatile Iter> == Expected);
  STATIC_REQUIRE(::cuda::__is_synthesizing_iterator_v<const volatile Iter> == Expected);
  STATIC_REQUIRE(::cuda::__is_synthesizing_iterator_v<Iter&> == Expected);
  STATIC_REQUIRE(::cuda::__is_synthesizing_iterator_v<const Iter&> == Expected);
  STATIC_REQUIRE(::cuda::__is_synthesizing_iterator_v<Iter&&> == Expected);
}

TEST_CASE("is_synthesizing_iterator", "[iterators]")
{
  using ThrustCounting = thrust::counting_iterator<int>;
  using ThrustConstant = thrust::constant_iterator<int>;
  using CudaCounting   = cuda::counting_iterator<int>;
  using CudaConstant   = cuda::constant_iterator<int>;

  check_is_synthesizing<int*, false>();
  check_is_synthesizing<ThrustCounting, true>();
  check_is_synthesizing<ThrustConstant, true>();
  check_is_synthesizing<thrust::shuffle_iterator<int>, true>();
  check_is_synthesizing<thrust::permutation_iterator<int*, ThrustCounting>, false>();

  check_is_synthesizing<thrust::strided_iterator<ThrustCounting, thrust::runtime_value<int>>, true>();
  check_is_synthesizing<thrust::strided_iterator<int*, thrust::runtime_value<int>>, false>();
  check_is_synthesizing<thrust::transform_iterator<Identity, ThrustCounting>, true>();
  check_is_synthesizing<thrust::transform_iterator<Identity, ThrustCounting, int, int>, true>();
  check_is_synthesizing<thrust::transform_iterator<Identity, int*>, false>();
  check_is_synthesizing<thrust::transform_iterator<Identity, CudaCounting>, true>();

  check_is_synthesizing<thrust::zip_iterator<cuda::std::tuple<ThrustCounting, ThrustConstant>>, true>();
  check_is_synthesizing<thrust::zip_iterator<cuda::std::tuple<ThrustCounting, int*>>, false>();
  check_is_synthesizing<thrust::zip_iterator<cuda::std::tuple<CudaCounting, ThrustConstant>>, true>();
  check_is_synthesizing<
    thrust::strided_iterator<thrust::zip_iterator<cuda::std::tuple<ThrustCounting, int*>>, thrust::runtime_value<int>>,
    false>();

  check_is_synthesizing<cuda::strided_iterator<ThrustCounting, int>, true>();
  check_is_synthesizing<cuda::transform_iterator<Identity, ThrustCounting>, true>();
  check_is_synthesizing<cuda::zip_iterator<ThrustCounting, CudaConstant>, true>();
  check_is_synthesizing<cuda::zip_iterator<ThrustCounting, int*>, false>();
  check_is_synthesizing<cuda::zip_transform_iterator<Identity, ThrustCounting, CudaConstant>, true>();
  check_is_synthesizing<cuda::zip_transform_iterator<Identity, ThrustCounting, int*>, false>();
}
