#include <thrust/detail/type_traits.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

#include <cuda/__cccl_config>

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

  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<int*>::value, true);
  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<thrust::device_ptr<int>>::value, true);

  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<HostVector::iterator>::value, true);
  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<HostVector::const_iterator>::value, true);

  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<DeviceVector::iterator>::value, true);
  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<DeviceVector::const_iterator>::value, true);

  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<thrust::device_ptr<int>>::value, true);

  using HostIteratorTuple = thrust::tuple<HostVector::iterator, HostVector::iterator>;

  using ConstantIterator  = thrust::constant_iterator<int>;
  using CountingIterator  = thrust::counting_iterator<int>;
  using TransformIterator = thrust::transform_iterator<cuda::std::identity, HostVector::iterator>;
  using ZipIterator       = thrust::zip_iterator<HostIteratorTuple>;

  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<ConstantIterator>::value, false);
  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<CountingIterator>::value, false);
  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<TransformIterator>::value, false);
  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<ZipIterator>::value, false);
}
DECLARE_UNITTEST(TestIsContiguousIterator);

void TestIsCommutative()
{
  {
    using T  = int;
    using Op = ::cuda::std::plus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = ::cuda::std::multiplies<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = ::cuda::minimum<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = ::cuda::maximum<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = ::cuda::std::logical_or<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = ::cuda::std::logical_and<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = ::cuda::std::bit_or<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = ::cuda::std::bit_and<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = ::cuda::std::bit_xor<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }

  {
    using T  = char;
    using Op = ::cuda::std::plus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = short;
    using Op = ::cuda::std::plus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = long;
    using Op = ::cuda::std::plus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = long long;
    using Op = ::cuda::std::plus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = float;
    using Op = ::cuda::std::plus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = double;
    using Op = ::cuda::std::plus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }

  {
    using T  = int;
    using Op = ::cuda::std::minus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, false);
  }
  {
    using T  = int;
    using Op = ::cuda::std::divides<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, false);
  }
  {
    using T  = float;
    using Op = ::cuda::std::divides<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, false);
  }
  {
    using T  = float;
    using Op = ::cuda::std::minus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, false);
  }

  {
    using T  = thrust::tuple<int, int>;
    using Op = ::cuda::std::plus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, false);
  }
}
DECLARE_UNITTEST(TestIsCommutative);

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
  static_assert(thrust::is_trivially_relocatable<thrust::pair<int, thrust::complex<float>>>::value, "");
  static_assert(thrust::is_trivially_relocatable<::cuda::std::pair<int, ::cuda::std::complex<float>>>::value, "");
  static_assert(thrust::is_trivially_relocatable<thrust::tuple<int, thrust::complex<float>, char>>::value, "");
  static_assert(thrust::is_trivially_relocatable<::cuda::std::tuple<int, ::cuda::std::complex<float>, char>>::value,
                "");
#endif // _CCCL_COMPILER(GCC, >=, 7)
  static_assert(thrust::is_trivially_relocatable<
                  ::cuda::std::tuple<thrust::pair<int, thrust::tuple<int, ::cuda::std::tuple<>>>,
                                     thrust::tuple<::cuda::std::pair<int, thrust::tuple<>>, int>>>::value,
                "");

  static_assert(!thrust::is_trivially_relocatable<thrust::pair<int, std::string>>::value, "");
  static_assert(!thrust::is_trivially_relocatable<::cuda::std::pair<int, std::string>>::value, "");
  static_assert(!thrust::is_trivially_relocatable<thrust::tuple<int, float, std::string>>::value, "");
  static_assert(!thrust::is_trivially_relocatable<::cuda::std::tuple<int, float, std::string>>::value, "");

  // test propagation of relocatability through pair and tuple
  static_assert(thrust::is_trivially_relocatable<NonTriviallyCopyable>::value, "");
  static_assert(thrust::is_trivially_relocatable<thrust::pair<NonTriviallyCopyable, int>>::value, "");
  static_assert(thrust::is_trivially_relocatable<::cuda::std::pair<NonTriviallyCopyable, int>>::value, "");
  static_assert(thrust::is_trivially_relocatable<thrust::tuple<NonTriviallyCopyable>>::value, "");
  static_assert(thrust::is_trivially_relocatable<::cuda::std::tuple<NonTriviallyCopyable>>::value, "");
};
DECLARE_UNITTEST(TestTriviallyRelocatable);
