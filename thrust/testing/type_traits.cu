#include <thrust/detail/type_traits.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

#include <cuda/__cccl_config>
#include <cuda/std/utility>

#if _CCCL_COMPILER(GCC, >=, 7)
// This header pulls in an unsuppressable warning on GCC 6
#  include <cuda/std/complex>
#endif // _CCCL_COMPILER(GCC, >=, 7)
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include <unittest/unittest.h>

void TestIsContiguousIterator()
{
  using HostVector   = thrust::host_vector<int>;
  using DeviceVector = thrust::device_vector<int>;

  ASSERT_EQUAL(thrust::is_contiguous_iterator_v<int*>, true);
  ASSERT_EQUAL(thrust::is_contiguous_iterator_v<thrust::device_ptr<int>>, true);

  ASSERT_EQUAL(thrust::is_contiguous_iterator_v<HostVector::iterator>, true);
  ASSERT_EQUAL(thrust::is_contiguous_iterator_v<HostVector::const_iterator>, true);

  ASSERT_EQUAL(thrust::is_contiguous_iterator_v<DeviceVector::iterator>, true);
  ASSERT_EQUAL(thrust::is_contiguous_iterator_v<DeviceVector::const_iterator>, true);

  ASSERT_EQUAL(thrust::is_contiguous_iterator_v<thrust::device_ptr<int>>, true);

  ASSERT_EQUAL(thrust::is_contiguous_iterator_v<const thrust::device_ptr<int>>, true);
  ASSERT_EQUAL(thrust::is_contiguous_iterator_v<volatile thrust::device_ptr<int>>, true);
  ASSERT_EQUAL(thrust::is_contiguous_iterator_v<thrust::device_ptr<int>&>, true);
  ASSERT_EQUAL(thrust::is_contiguous_iterator_v<const thrust::device_ptr<int>&>, true);
  ASSERT_EQUAL(thrust::is_contiguous_iterator_v<volatile thrust::device_ptr<int>&>, true);

  using HostIteratorTuple = cuda::std::tuple<HostVector::iterator, HostVector::iterator>;

  using ConstantIterator  = thrust::constant_iterator<int>;
  using CountingIterator  = thrust::counting_iterator<int>;
  using TransformIterator = thrust::transform_iterator<cuda::std::identity, HostVector::iterator>;
  using ZipIterator       = thrust::zip_iterator<HostIteratorTuple>;

  ASSERT_EQUAL(thrust::is_contiguous_iterator_v<ConstantIterator>, false);
  ASSERT_EQUAL(thrust::is_contiguous_iterator_v<CountingIterator>, false);
  ASSERT_EQUAL(thrust::is_contiguous_iterator_v<TransformIterator>, false);
  ASSERT_EQUAL(thrust::is_contiguous_iterator_v<ZipIterator>, false);
}
DECLARE_UNITTEST(TestIsContiguousIterator);

struct NonTriviallyCopyable
{
  NonTriviallyCopyable(const NonTriviallyCopyable&) {}
};
THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(NonTriviallyCopyable);

static_assert(!::cuda::std::is_trivially_copyable<NonTriviallyCopyable>::value, "");
static_assert(thrust::is_trivially_relocatable<NonTriviallyCopyable>::value, "");

void TestTriviallyRelocatable()
{
  static_assert(thrust::is_trivially_relocatable<int>::value, "");
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  static_assert(thrust::is_trivially_relocatable<__half>::value, "");
  static_assert(thrust::is_trivially_relocatable<int1>::value, "");
  static_assert(thrust::is_trivially_relocatable<int2>::value, "");
  static_assert(thrust::is_trivially_relocatable<int3>::value, "");
  static_assert(thrust::is_trivially_relocatable<int4>::value, "");
#  if _CCCL_HAS_INT128()
  static_assert(thrust::is_trivially_relocatable<__int128>::value, "");
#  endif // _CCCL_HAS_INT128()
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#if _CCCL_COMPILER(GCC, >=, 7)
  static_assert(thrust::is_trivially_relocatable<thrust::complex<float>>::value, "");
  static_assert(thrust::is_trivially_relocatable<::cuda::std::complex<float>>::value, "");
  static_assert(thrust::is_trivially_relocatable<cuda::std::pair<int, thrust::complex<float>>>::value, "");
  static_assert(thrust::is_trivially_relocatable<::cuda::std::pair<int, ::cuda::std::complex<float>>>::value, "");
  static_assert(thrust::is_trivially_relocatable<cuda::std::tuple<int, thrust::complex<float>, char>>::value, "");
  static_assert(thrust::is_trivially_relocatable<::cuda::std::tuple<int, ::cuda::std::complex<float>, char>>::value,
                "");
#endif // _CCCL_COMPILER(GCC, >=, 7)
  static_assert(thrust::is_trivially_relocatable<
                  ::cuda::std::tuple<cuda::std::pair<int, cuda::std::tuple<int, ::cuda::std::tuple<>>>,
                                     cuda::std::tuple<::cuda::std::pair<int, cuda::std::tuple<>>, int>>>::value,
                "");

  static_assert(!thrust::is_trivially_relocatable<cuda::std::pair<int, std::string>>::value, "");
  static_assert(!thrust::is_trivially_relocatable<::cuda::std::pair<int, std::string>>::value, "");
  static_assert(!thrust::is_trivially_relocatable<cuda::std::tuple<int, float, std::string>>::value, "");
  static_assert(!thrust::is_trivially_relocatable<::cuda::std::tuple<int, float, std::string>>::value, "");

  // test propagation of relocatability through pair and tuple
  static_assert(thrust::is_trivially_relocatable<NonTriviallyCopyable>::value, "");
  static_assert(thrust::is_trivially_relocatable<cuda::std::pair<NonTriviallyCopyable, int>>::value, "");
  static_assert(thrust::is_trivially_relocatable<::cuda::std::pair<NonTriviallyCopyable, int>>::value, "");
  static_assert(thrust::is_trivially_relocatable<cuda::std::tuple<NonTriviallyCopyable>>::value, "");
  static_assert(thrust::is_trivially_relocatable<::cuda::std::tuple<NonTriviallyCopyable>>::value, "");
};
DECLARE_UNITTEST(TestTriviallyRelocatable);
