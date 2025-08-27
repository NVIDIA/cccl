// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_for.cuh>
#include <cub/device/device_transform.cuh>
#include <cub/iterator/cache_modified_output_iterator.cuh>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/zip_function.h>

#include <cuda/iterator>
#include <cuda/std/__functional/identity.h>

#include <sstream>

#include "catch2_large_problem_helper.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/custom_type.h>
#include <c2h/test_util_vec.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceTransform::Transform, transform_many);
DECLARE_LAUNCH_WRAPPER(cub::DeviceTransform::TransformStableArgumentAddresses, transform_many_stable);
DECLARE_LAUNCH_WRAPPER(cub::DeviceTransform::Fill, fill);

using offset_types = c2h::type_list<std::int32_t, std::int64_t>;

C2H_TEST("DeviceTransform::Transform BabelStream add",
         "[device][transform]",
         c2h::type_list<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t, uchar3>,
         offset_types)
{
  using type     = c2h::get<0, TestType>;
  using offset_t = c2h::get<1, TestType>;

  // test edge cases around 16, 128, page size, and full tile
  const offset_t num_items = GENERATE(0, 1, 15, 16, 17, 127, 128, 129, 4095, 4096, 4097, 100'000);
  CAPTURE(c2h::type_name<type>(), c2h::type_name<offset_t>(), num_items);

  c2h::device_vector<type> a(num_items, thrust::no_init);
  c2h::device_vector<type> b(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), a);
  c2h::gen(C2H_SEED(1), b);

  c2h::device_vector<type> result(num_items, thrust::no_init);
  transform_many(cuda::std::make_tuple(a.begin(), b.begin()), result.begin(), num_items, cuda::std::plus<type>{});

  // compute reference and verify
  c2h::host_vector<type> a_h = a;
  c2h::host_vector<type> b_h = b;
  c2h::host_vector<type> reference_h(num_items, thrust::no_init);
  std::transform(a_h.begin(), a_h.end(), b_h.begin(), reference_h.begin(), std::plus<type>{});
  REQUIRE(reference_h == result);
}

// note: because this uses a fancy iterator type, it will only test the fallback kernel
C2H_TEST("DeviceTransform::Transform works for large number of items",
         "[device][transform][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]",
         offset_types)
{
  using offset_t = c2h::get<0, TestType>;
  CAPTURE(c2h::type_name<offset_t>());
  const auto num_items = detail::make_large_offset<offset_t>();

  auto in_it              = thrust::make_counting_iterator(offset_t{0});
  auto expected_result_it = in_it;

  // Prepare helper to check results
  auto check_result_helper = detail::large_problem_test_helper(num_items);
  auto check_result_it     = check_result_helper.get_flagging_output_iterator(expected_result_it);

  transform_many(in_it, check_result_it, num_items, cuda::std::identity{});

  check_result_helper.check_all_results_correct();
}

C2H_TEST("DeviceTransform::Transform with multiple inputs works for large number of items",
         "[device][transform][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]",
         offset_types)
{
  using offset_t = c2h::get<0, TestType>;
  CAPTURE(c2h::type_name<offset_t>());
  const offset_t num_items = detail::make_large_offset<offset_t>();

  auto a_it               = thrust::make_counting_iterator(offset_t{0});
  auto b_it               = thrust::make_constant_iterator(offset_t{42});
  auto expected_result_it = thrust::make_counting_iterator(offset_t{42});

  // Prepare helper to check results
  auto check_result_helper = detail::large_problem_test_helper(num_items);
  auto check_result_it     = check_result_helper.get_flagging_output_iterator(expected_result_it);

  transform_many(cuda::std::make_tuple(a_it, b_it), check_result_it, num_items, cuda::std::plus<offset_t>{});

  check_result_helper.check_all_results_correct();
}

struct times_seven
{
  template <typename T>
  __host__ __device__ auto operator()(T v) const -> T
  {
    return static_cast<T>(v * 7);
  }
};

C2H_TEST("DeviceTransform::Transform with large input",
         "[device][transform][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]",
         offset_types)
try
{
  using type     = unsigned short;
  using offset_t = c2h::get<0, TestType>;

  // make size a few thread blocks below/beyond 4GiB. need to make sure I32 num_items stays below 2^31
  constexpr offset_t num_items = static_cast<offset_t>((1ll << 31) + (sizeof(offset_t) == 4 ? -123456 : 123456));
  REQUIRE(num_items > 0);

  c2h::device_vector<type> input(static_cast<size_t>(num_items), thrust::no_init);
  c2h::gen(C2H_SEED(1), input);

  c2h::device_vector<type> result(static_cast<size_t>(num_items), thrust::no_init);
  transform_many(cuda::std::make_tuple(input.begin()), result.begin(), num_items, times_seven{});

  // compute reference and verify
  c2h::host_vector<type> input_h = input;
  c2h::host_vector<type> reference_h(static_cast<size_t>(num_items), thrust::no_init);
  std::transform(input_h.begin(), input_h.end(), reference_h.begin(), times_seven{});
  REQUIRE((reference_h == result));
}
catch (const std::bad_alloc&)
{
  // allocation failure is not a test failure, so we can run tests on smaller GPUs
}

template <int Alignment>
struct overaligned_addable_and_equal_comparable_policy
{
  template <typename CustomType>
  struct alignas(Alignment) type
  {
    __host__ __device__ static void check(const CustomType& obj)
    {
      _CCCL_VERIFY(reinterpret_cast<uintptr_t>(&obj) % Alignment == 0,
                   "overaligned_addable_policy_t<Alignment> is not sufficiently aligned");
    }

    __host__ __device__ friend auto operator==(const CustomType& a, const CustomType& b) -> bool
    {
      check(a);
      check(b);
      return a.key == b.key;
    }

    __host__ __device__ friend auto operator+(char u, const CustomType& b) -> CustomType
    {
      check(b);
      CustomType result{};
      result.key = static_cast<size_t>(u) + b.key;
      result.val = b.val;
      return result;
    }
  };
};

template <int Alignment>
using overaligned_t = c2h::custom_type_t<overaligned_addable_and_equal_comparable_policy<Alignment>::template type>;

using huge_t = c2h::custom_type_t<c2h::equal_comparable_t, c2h::accumulateable_t, c2h::huge_data<666>::type>;
static_assert(alignof(huge_t) == 8, "Need a large type with alignment < 16");

using uncommon_types = c2h::type_list<
  // these vector types have a non-power-of-two size, and size and alignment are different
  char3,
  short3,
  int3,
  longlong3,
  overaligned_t<32>, // exceeds the memcpy_async (Hopper/Blackwell) and bulk copy alignments (only Blackwell)
#if !_CCCL_COMPILER(MSVC) // error C2719: [...] formal parameter with requested alignment of 256 won't be aligned
  overaligned_t<256>, // exceeds copy alignment on Hopper
                      // and exhausts guaranteed shared memory on Hopper (block_threads = 256, req. smem = 64KiB)
  overaligned_t<512>, // exhausts guaranteed shared memory on Blackwell (block_threads = 128, req. smem = 64KiB)
#endif // !_CCCL_COMPILER(MSVC)
  huge_t>;

struct uncommon_plus
{
  // vector types
  template <typename T>
  __host__ __device__ auto operator()(int8_t a, const T& b) const -> T
  {
    return T{a, 0, 0} + b;
  }

  __host__ __device__ auto operator()(int8_t a, const huge_t& b) const -> huge_t
  {
    huge_t r = b;
    r.key += static_cast<size_t>(a);
    return r;
  }

  __host__ __device__ auto operator()(int8_t a, const overaligned_t<32>& b) const -> overaligned_t<32>
  {
    return a + b;
  }

  __host__ __device__ auto operator()(int8_t a, const overaligned_t<256>& b) const -> overaligned_t<256>
  {
    return a + b;
  }

  __host__ __device__ auto operator()(int8_t a, const overaligned_t<512>& b) const -> overaligned_t<512>
  {
    return a + b;
  }
};

C2H_TEST("DeviceTransform::Transform uncommon types", "[device][transform]", uncommon_types)
{
  using type = c2h::get<0, TestType>;
  CAPTURE(c2h::type_name<type>());

  const int num_items = GENERATE(0, 1, 100, 1'000, 100'000); // try to hit the small and full tile code paths
  c2h::device_vector<int8_t> a(num_items, thrust::default_init); // put some bytes at the front, so SMEM has to handle
                                                                 // padding between tiles to align them
  c2h::device_vector<type> b(num_items, thrust::default_init);
  c2h::gen(C2H_SEED(1), a, int8_t{0}, int8_t{100});
  c2h::gen(C2H_SEED(1), b);

  c2h::device_vector<type> result(num_items, thrust::default_init);
  transform_many(cuda::std::make_tuple(a.begin(), b.begin()), result.begin(), num_items, uncommon_plus{});

  c2h::host_vector<int8_t> a_h = a;
  c2h::host_vector<type> b_h   = b;
  c2h::host_vector<type> reference_h(num_items);
  std::transform(a_h.begin(), a_h.end(), b_h.begin(), reference_h.begin(), uncommon_plus{});
  REQUIRE(c2h::host_vector<type>(result) == reference_h);
}

struct non_default_constructible
{
  int data;

  non_default_constructible()                                            = delete;
  non_default_constructible(const non_default_constructible&)            = default;
  non_default_constructible& operator=(const non_default_constructible&) = default;
  ~non_default_constructible()                                           = default;

  __host__ __device__ explicit non_default_constructible(int data)
      : data(data)
  {}

  friend __host__ __device__ auto operator==(non_default_constructible a, non_default_constructible b) -> bool
  {
    return a.data == b.data;
  }
};
static_assert(!cuda::std::is_trivially_default_constructible_v<non_default_constructible>);
static_assert(!cuda::std::is_default_constructible_v<non_default_constructible>);
static_assert(cuda::std::is_trivially_copyable_v<non_default_constructible>); // as required by the standard
static_assert(thrust::is_trivially_relocatable_v<non_default_constructible>); // CUB uses this check internally

C2H_TEST("DeviceTransform::Transform non-default constructible types", "[device][transform]")
{
  using type          = non_default_constructible;
  const int num_items = GENERATE(0, 1, 100, 1'000, 100'000); // try to hit the small and full tile code paths

  c2h::device_vector<type> input(num_items, non_default_constructible{42});
  c2h::device_vector<type> result(num_items, non_default_constructible{0});

  transform_many(cuda::std::make_tuple(input.begin()), result.begin(), num_items, cuda::std::identity{});

  c2h::host_vector<type> reference_h(num_items, non_default_constructible{42});
  REQUIRE(c2h::host_vector<type>(result) == reference_h);
}

template <typename T>
struct nstream_kernel
{
  static constexpr T scalar = 42;

  __host__ __device__ T operator()(const T& ai, const T& bi, const T& ci) const
  {
    return ai + bi + scalar * ci;
  }
};

// overwrites one input stream
C2H_TEST("DeviceTransform::Transform BabelStream nstream",
         "[device][transform]",
         c2h::type_list<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t>,
         offset_types)
{
  using type     = c2h::get<0, TestType>;
  using offset_t = c2h::get<1, TestType>;

  const offset_t num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  c2h::device_vector<type> a(num_items, thrust::no_init);
  c2h::device_vector<type> b(num_items, thrust::no_init);
  c2h::device_vector<type> c(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), a);
  c2h::gen(C2H_SEED(1), b);
  c2h::gen(C2H_SEED(1), c);

  // copy to host before changing
  c2h::host_vector<type> a_h = a;
  c2h::host_vector<type> b_h = b;
  c2h::host_vector<type> c_h = c;

  transform_many(cuda::std::make_tuple(a.begin(), b.begin(), c.begin()), a.begin(), num_items, nstream_kernel<type>{});

  // compute reference and verify
  auto z = thrust::make_zip_iterator(a_h.begin(), b_h.begin(), c_h.begin());
  std::transform(z, z + num_items, a_h.begin(), thrust::make_zip_function(nstream_kernel<type>{}));
  REQUIRE(a_h == a);
}

struct sum_five
{
  __host__ __device__ auto operator()(std::int8_t a, std::int16_t b, std::int32_t c, std::int64_t d, float e) const
    -> double
  {
    return a + b + c + d + e;
  }
};

C2H_TEST("DeviceTransform::Transform add five streams", "[device][transform]")
{
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  c2h::device_vector<std::int8_t> a(num_items, thrust::no_init);
  c2h::device_vector<std::int16_t> b(num_items, thrust::no_init);
  c2h::device_vector<std::int32_t> c(num_items, thrust::no_init);
  c2h::device_vector<std::int64_t> d(num_items, thrust::no_init);
  c2h::device_vector<float> e(num_items, thrust::no_init);

  c2h::gen(C2H_SEED(1), a, std::int8_t{10}, std::int8_t{100});
  c2h::gen(C2H_SEED(1), b, std::int16_t{10}, std::int16_t{100});
  c2h::gen(C2H_SEED(1), c, std::int32_t{10}, std::int32_t{100});
  c2h::gen(C2H_SEED(1), d, std::int64_t{10}, std::int64_t{100});
  c2h::gen(C2H_SEED(1), e, float{10}, float{100});

  c2h::device_vector<double> result(num_items, thrust::no_init);
  transform_many(
    cuda::std::make_tuple(a.begin(), b.begin(), c.begin(), d.begin(), e.begin()), result.begin(), num_items, sum_five{});

  // compute reference and verify
  c2h::host_vector<std::int8_t> a_h  = a;
  c2h::host_vector<std::int16_t> b_h = b;
  c2h::host_vector<std::int32_t> c_h = c;
  c2h::host_vector<std::int64_t> d_h = d;
  c2h::host_vector<float> e_h        = e;
  c2h::host_vector<double> reference_h(num_items, thrust::no_init);
  auto zip = thrust::zip_iterator{a_h.begin(), b_h.begin(), c_h.begin(), d_h.begin(), e_h.begin()};
  std::transform(zip, zip + num_items, reference_h.begin(), thrust::zip_function{sum_five{}});
  REQUIRE(reference_h == result);
}

struct give_me_five
{
  __device__ auto operator()() const -> int
  {
    return 5;
  }
};

C2H_TEST("DeviceTransform::Fill", "[device][transform]")
{
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  c2h::device_vector<int> result(num_items, thrust::no_init);
  fill(result.begin(), num_items, give_me_five{});

  // compute reference and verify
  c2h::device_vector<int> reference(num_items, 5);
  REQUIRE(reference == result);
}

C2H_TEST("DeviceTransform::Transform fancy input iterator types", "[device][transform]")
{
  using type          = int;
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  thrust::counting_iterator<type> a{0};
  thrust::counting_iterator<type> b{10};

  c2h::device_vector<type> result(num_items, thrust::no_init);
  transform_many(cuda::std::make_tuple(a, b), result.begin(), num_items, cuda::std::plus<type>{});

  // compute reference and verify
  c2h::host_vector<type> reference_h(num_items);
  std::transform(a, a + num_items, b, reference_h.begin(), std::plus<type>{});
  REQUIRE(reference_h == result);
}

C2H_TEST("DeviceTransform::Transform fancy output iterator type", "[device][transform]")
{
  using type          = int;
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  c2h::device_vector<type> a(num_items, 13);
  c2h::device_vector<type> b(num_items, 35);
  c2h::device_vector<type> result(num_items, thrust::no_init);

  using thrust::placeholders::_1;
  auto out = thrust::make_transform_output_iterator(result.begin(), _1 + 4);
  transform_many(cuda::std::make_tuple(a.begin(), b.begin()), out, num_items, cuda::std::plus<type>{});
  REQUIRE(result == c2h::device_vector<type>(num_items, (13 + 35) + 4));
}

C2H_TEST("DeviceTransform::Transform fancy output iterator type with void value type", "[device][transform]")
{
  using type          = int;
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  c2h::device_vector<type> a(num_items, 1);
  c2h::device_vector<type> b(num_items, 2);
  c2h::device_vector<type> result(num_items, thrust::no_init);

  using it_t = cub::CacheModifiedOutputIterator<cub::CacheStoreModifier::STORE_DEFAULT, int>;
  static_assert(cuda::std::is_void_v<it_t::value_type>);
  auto out = it_t{thrust::raw_pointer_cast(result.data())};
  transform_many(cuda::std::make_tuple(a.begin(), b.begin()), out, num_items, cuda::std::plus<type>{});
  REQUIRE(result == c2h::device_vector<type>(num_items, 3));
}

C2H_TEST("DeviceTransform::Transform mixed input iterator types", "[device][transform]")
{
  using type          = int;
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  thrust::counting_iterator<type> a{0};
  c2h::device_vector<type> b(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), b);

  c2h::device_vector<type> result(num_items, thrust::no_init);
  transform_many(cuda::std::make_tuple(a, b.begin()), result.begin(), num_items, cuda::std::plus<type>{});

  // compute reference and verify
  c2h::host_vector<type> b_h = b;
  c2h::host_vector<type> reference_h(num_items);
  std::transform(a, a + num_items, b_h.begin(), reference_h.begin(), std::plus<type>{});
  REQUIRE(reference_h == result);
}

struct plus_needs_stable_address
{
  int* a;
  int* b;

  __host__ __device__ int operator()(const int& v) const
  {
    const auto i = &v - a;
    return v + b[i];
  }
};

C2H_TEST("DeviceTransform::Transform address stability", "[device][transform]")
{
  using type          = int;
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  c2h::device_vector<type> a(num_items, thrust::no_init);
  c2h::device_vector<type> b(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), a);
  c2h::gen(C2H_SEED(1), b);

  c2h::device_vector<type> result(num_items, thrust::no_init);
  transform_many_stable(
    cuda::std::make_tuple(thrust::raw_pointer_cast(a.data())),
    result.begin(),
    num_items,
    plus_needs_stable_address{thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(b.data())});

  // compute reference and verify
  c2h::device_vector<type> a_h = a;
  c2h::device_vector<type> b_h = b;
  c2h::host_vector<type> reference_h(num_items);
  std::transform(a_h.begin(), a_h.end(), b_h.begin(), reference_h.begin(), std::plus<type>{});
  REQUIRE(reference_h == result);
}

// Non-trivially-copyable/relocatable type which cannot be copied using std::memcpy or cudaMemcpy
struct non_trivial
{
  int data;

  non_trivial() = default;

  __host__ __device__ explicit non_trivial(int data)
      : data(data)
  {}

  __host__ __device__ non_trivial(const non_trivial& nt)
      : data(nt.data)
  {}

  __host__ __device__ auto operator=(const non_trivial& nt) -> non_trivial&
  {
    data = nt.data;
    return *this;
  }

  __host__ __device__ auto operator-() const -> non_trivial
  {
    return non_trivial{-data};
  }

  friend __host__ __device__ auto operator==(non_trivial a, non_trivial b) -> bool
  {
    return a.data == b.data;
  }
};
static_assert(!cuda::std::is_trivially_copyable_v<non_trivial>); // as required by the standard
static_assert(!thrust::is_trivially_relocatable_v<non_trivial>); // CUB uses this check internally

// Note(bgruber): I gave up on writing a test that checks whether the copy ctor/assignment operator is actually called
// (e.g. by tracking/counting invocations of those), since C++ allows (but not guarantees) elision of these operations.
// Also thrust algorithms perform a lot of copies in-between, so the test needs to use only raw allocations and
// iteration for setup and checking.
C2H_TEST("DeviceTransform::Transform not trivially relocatable", "[device][transform]")
{
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  c2h::device_vector<non_trivial> input(num_items, non_trivial{42});
  c2h::device_vector<non_trivial> result(num_items, thrust::no_init);
  transform_many(
    cuda::std::make_tuple(thrust::raw_pointer_cast(input.data())), result.begin(), num_items, cuda::std::negate<>{});

  const auto reference = c2h::device_vector<non_trivial>(num_items, non_trivial{-42});
  REQUIRE((reference == result));
}

C2H_TEST("DeviceTransform::Transform buffer start alignment",
         "[device][transform]",
         c2h::type_list<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t>)
{
  using type          = c2h::get<0, TestType>;
  const int num_items = GENERATE(
    10, // try to hit sub-size-alignment, small and full tile code paths
    130,
    100'000,
    128 * 22 /* blackwell u8 tilesize */ + 2,
    128 * 11 /* blackwell u16 tilesize */ + 2,
    128 * 6 /* blackwell u32 tilesize */ + 2,
    128 * 3 /* blackwell u64 tilesize */ + 2);

  // we should hit byte offsets up until 64 (128 is the highest bulk_copy_alignment)
  const int offset_a = GENERATE(1, 2, 4, 8, 15, 16, 32, 64, 127);
  const int offset_b = GENERATE(1, 2, 4, 8, 15, 16, 32, 64, 127);
  if (num_items <= offset_a || num_items <= offset_b || offset_a * sizeof(type) > 64 || offset_b * sizeof(type) > 64)
  {
    return;
  }
  const int offset_r = offset_a;
  CAPTURE(c2h::type_name<type>(), num_items, offset_a, offset_b);

  c2h::device_vector<type> a(num_items + offset_a, thrust::no_init);
  c2h::device_vector<type> b(num_items + offset_b, thrust::no_init);
  thrust::sequence(a.begin(), a.end());
  thrust::sequence(b.begin(), b.end(), num_items + offset_a);
  c2h::device_vector<type> result(num_items + offset_r);
  transform_many(cuda::std::make_tuple(a.begin() + offset_a, b.begin() + offset_b),
                 result.begin() + offset_r,
                 num_items,
                 cuda::std::plus{});

  using thrust::placeholders::_1;
  c2h::device_vector<type> reference(num_items + offset_r);
  thrust::tabulate(reference.begin() + offset_r, reference.end(), (_1 + offset_a) * 2 + offset_b + num_items);
  REQUIRE(reference == result);
}

namespace Catch
{
template <typename T>
struct StringMaker<cub::detail::transform::aligned_base_ptr<T>>
{
  static auto convert(cub::detail::transform::aligned_base_ptr<T> abp) -> std::string
  {
    std::stringstream ss;
    ss << "{ptr: " << abp.ptr << ", head_padding: " << abp.head_padding << "}";
    return ss.str();
  }
};
} // namespace Catch

// TODO(bgruber): rewrite this example using int3
C2H_TEST("DeviceTransform::Transform aligned_base_ptr", "[device][transform]")
{
  alignas(128) int arr[256];
  using namespace cub::detail::transform;
  CHECK(make_aligned_base_ptr(&arr[0], 128) == aligned_base_ptr<int>{reinterpret_cast<char*>(&arr[0]), 0});
  CHECK(make_aligned_base_ptr(&arr[1], 128) == aligned_base_ptr<int>{reinterpret_cast<char*>(&arr[0]), 4});
  CHECK(make_aligned_base_ptr(&arr[5], 128) == aligned_base_ptr<int>{reinterpret_cast<char*>(&arr[0]), 20});
  CHECK(make_aligned_base_ptr(&arr[31], 128) == aligned_base_ptr<int>{reinterpret_cast<char*>(&arr[0]), 124});
  CHECK(make_aligned_base_ptr(&arr[32], 128) == aligned_base_ptr<int>{reinterpret_cast<char*>(&arr[32]), 0});
  CHECK(make_aligned_base_ptr(&arr[33], 128) == aligned_base_ptr<int>{reinterpret_cast<char*>(&arr[32]), 4});
  CHECK(make_aligned_base_ptr(&arr[127], 128) == aligned_base_ptr<int>{reinterpret_cast<char*>(&arr[96]), 124});
  CHECK(make_aligned_base_ptr(&arr[128], 128) == aligned_base_ptr<int>{reinterpret_cast<char*>(&arr[128]), 0});
  CHECK(make_aligned_base_ptr(&arr[129], 128) == aligned_base_ptr<int>{reinterpret_cast<char*>(&arr[128]), 4});
}

C2H_TEST("DeviceTransform::Transform aligned_base_ptr", "[device][transform]")
{
  using It         = cuda::std::reverse_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int>>>;
  using kernel_arg = cub::detail::transform::kernel_arg<It>;

  STATIC_REQUIRE(cuda::std::is_constructible_v<kernel_arg>);
  STATIC_REQUIRE(cuda::std::is_copy_constructible_v<kernel_arg>);
}

// See discussion on: https://github.com/NVIDIA/cccl/pull/4815
C2H_TEST("DeviceTransform::Transform vectorized output bug", "[device][transform]")
{
  using thrust::placeholders::_1;

  int num_items = std::numeric_limits<std::uint16_t>::max() - 1;
  c2h::device_vector<std::uint16_t> input(num_items);
  c2h::device_vector<std::uint16_t> output(num_items);
  thrust::sequence(input.begin(), input.end());

  auto out_it = thrust::make_transform_output_iterator(output.begin(), _1);
  transform_many(input.begin(), out_it, num_items, _1 + 1);

  c2h::host_vector<std::uint16_t> reference(num_items);
  thrust::generate(reference.begin(), reference.end(), [i = 0]() mutable {
    return static_cast<std::uint16_t>(++i);
  });
  CHECK(output == reference);
}

// See discussion on: https://github.com/NVIDIA/cccl/pull/4815
struct A
{
  int value;
};

struct B
{
  int value;
};

struct C
{
  int value;

  __host__ __device__ friend auto operator==(C a, C b) -> bool
  {
    return a.value == b.value;
  }

  friend auto operator<<(std::ostream& os, C c) -> std::ostream&
  {
    return os << "C{" << c.value << "}";
  }
};

struct AtoB
{
  __host__ __device__ B operator()(A a) const
  {
    return B{a.value + 1};
  }
};

struct BtoC
{
  __host__ __device__ C operator()(B b) const
  {
    return C{-b.value};
  }
};

C2H_TEST("DeviceTransform::Transform function/output_iter return type not convertible", "[device][transform]")
{
  using thrust::placeholders::_1;

  const int num_items = 10'000;
  c2h::device_vector<A> input(num_items, A{42});
  c2h::device_vector<C> output(num_items, thrust::no_init);

  auto out_it = thrust::make_transform_output_iterator(output.begin(), BtoC{});
  transform_many(input.begin(), out_it, num_items, AtoB{});

  c2h::device_vector<C> reference(num_items, C{-43});
  CHECK(output == reference);
}

__global__ void unrelated_kernel()
{
  __shared__ int ssmem; // 4 bytes
  extern __shared__ int dsmem[]; // aligned to 16 by default, so 12 bytes padding needed
  asm("" : "+r"(ssmem));
  asm("" : "+r"(dsmem[0]));
}

C2H_TEST("DeviceTransform::Transform does not effect unrelated kernel's static SMEM consumption", "[device][transform]")
{
  cudaFuncAttributes attrs;
  REQUIRE(cudaFuncGetAttributes(&attrs, unrelated_kernel) == cudaSuccess);
  REQUIRE(attrs.sharedSizeBytes == 4 + 12);
}

#if TEST_LAUNCH == 0

template <int BlockThreads, int ItemsPerPthread, typename T>
__global__ void fill_pdl_kernel(T* data, size_t n, T value)
{
  // we trigger the next kernel's launch very soon and wait a bit for it to spin up before starting to write. this way
  // we try to expose the next kernel reading uninitialized data, if it contains a bug.
  _CCCL_PDL_GRID_DEPENDENCY_SYNC();
  _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
  NV_IF_TARGET(NV_PROVIDES_SM_70, __nanosleep(100'000);); // must be enough to cover the next kernel's launch overhead

  const int tile_size = ItemsPerPthread * BlockThreads;
  const size_t offset = size_t{blockIdx.x} * tile_size;

  data += offset;
  n -= offset;

  for (int j = 0; j < ItemsPerPthread; j++)
  {
    const int i = threadIdx.x + j * BlockThreads;
    if (i < n)
    {
      data[i] = value;
    }
  }
}

template <typename T>
void fill_pdl(T* data, size_t n, T value)
{
  constexpr auto block_threads    = 256;
  constexpr auto items_per_thread = 4;
  const auto blocks               = static_cast<unsigned>(::cuda::ceil_div(n, block_threads * items_per_thread));

  THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
    blocks, block_threads, /* smem */ 0, /*stream*/ 0, /* pdl */ true)
    .doit(fill_pdl_kernel<block_threads, items_per_thread, T>, data, n, value);
}

C2H_TEST("DeviceTransform::Transform PDL overlap check", "[device][transform]")
{
  using type = int;
  // need a warmup run to lazy load kernels and perform some setup, then a problem size that occupies 1/2 of all SMs
  const int num_items = GENERATE(1, 50 * 128);

  c2h::device_vector<type> data(num_items, thrust::no_init);
  c2h::device_vector<bool> flags(num_items, thrust::no_init);
  c2h::device_vector<type> result(1, thrust::no_init);

  using thrust::placeholders::_1;

  // completely async work of filling, 2x transforming and 1x reduction. we also avoid using the launch wrapper, since
  // it would synchronize
  fill_pdl(thrust::raw_pointer_cast(data.data()), num_items, 42);
  cub::DeviceTransform::Transform(::cuda::std::make_tuple(data.begin()), data.begin(), num_items, cuda::std::negate{});
  cub::DeviceTransform::Transform(::cuda::std::make_tuple(data.begin()), flags.begin(), num_items, _1 == -42);
  thrust::reduce_into(
    thrust::cuda::par_nosync, flags.begin(), flags.end(), result.begin(), true, ::cuda::std::logical_and{});
  REQUIRE(result[0]); // access finally synchronize
}
#endif
