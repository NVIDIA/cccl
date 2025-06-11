// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_for.cuh>
#include <cub/device/device_transform.cuh>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/zip_function.h>

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

using offset_types = c2h::type_list<std::int32_t, std::int64_t>;

C2H_TEST("DeviceTransform::Transform BabelStream add",
         "[device][device_transform]",
         c2h::type_list<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t>,
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
  transform_many(::cuda::std::make_tuple(a.begin(), b.begin()), result.begin(), num_items, ::cuda::std::plus<type>{});

  // compute reference and verify
  c2h::host_vector<type> a_h = a;
  c2h::host_vector<type> b_h = b;
  c2h::host_vector<type> reference_h(num_items, thrust::no_init);
  std::transform(a_h.begin(), a_h.end(), b_h.begin(), reference_h.begin(), std::plus<type>{});
  REQUIRE(reference_h == result);
}

C2H_TEST("DeviceTransform::Transform works for large number of items",
         "[device][device_transform][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]",
         offset_types)
{
  using offset_t = c2h::get<0, TestType>;
  CAPTURE(c2h::type_name<offset_t>());

  // Clamp 64-bit offset type problem sizes to just slightly larger than 2^32 items
  const auto num_items_max_ull = ::cuda::std::clamp(
    static_cast<std::size_t>(::cuda::std::numeric_limits<offset_t>::max()),
    std::size_t{0},
    ::cuda::std::numeric_limits<std::uint32_t>::max() + static_cast<std::size_t>(2000000ULL));
  const offset_t num_items = static_cast<offset_t>(num_items_max_ull);

  auto in_it              = thrust::make_counting_iterator(offset_t{0});
  auto expected_result_it = in_it;

  // Prepare helper to check results
  auto check_result_helper = detail::large_problem_test_helper(num_items);
  auto check_result_it     = check_result_helper.get_flagging_output_iterator(expected_result_it);

  transform_many(in_it, check_result_it, num_items, ::cuda::std::identity{});

  check_result_helper.check_all_results_correct();
}

C2H_TEST("DeviceTransform::Transform with multiple inputs works for large number of items",
         "[device][device_transform][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]",
         offset_types)
{
  using offset_t = c2h::get<0, TestType>;
  CAPTURE(c2h::type_name<offset_t>());

  // Clamp 64-bit offset type problem sizes to just slightly larger than 2^32 items
  const auto num_items_max_ull = ::cuda::std::clamp(
    static_cast<std::size_t>(::cuda::std::numeric_limits<offset_t>::max()),
    std::size_t{0},
    ::cuda::std::numeric_limits<std::uint32_t>::max() + static_cast<std::size_t>(2000000ULL));
  const offset_t num_items = static_cast<offset_t>(num_items_max_ull);

  auto a_it               = thrust::make_counting_iterator(offset_t{0});
  auto b_it               = thrust::make_constant_iterator(offset_t{42});
  auto expected_result_it = thrust::make_counting_iterator(offset_t{42});

  // Prepare helper to check results
  auto check_result_helper = detail::large_problem_test_helper(num_items);
  auto check_result_it     = check_result_helper.get_flagging_output_iterator(expected_result_it);

  transform_many(::cuda::std::make_tuple(a_it, b_it), check_result_it, num_items, ::cuda::std::plus<offset_t>{});

  check_result_helper.check_all_results_correct();
}

struct times_seven
{
  _CCCL_HOST_DEVICE auto operator()(unsigned char v) const -> char
  {
    return static_cast<unsigned char>(v * 7);
  }
};

C2H_TEST("DeviceTransform::Transform with large input",
         "[device][device_transform][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]")
try
{
  using type     = unsigned char;
  using offset_t = cuda::std::int64_t;

  constexpr offset_t num_items = (offset_t{1} << 32) + 123456; // a few thread blocks beyond 4GiB
  c2h::device_vector<type> input(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), input);

  c2h::device_vector<type> result(num_items, thrust::no_init);
  transform_many(::cuda::std::make_tuple(input.begin()), result.begin(), num_items, times_seven{});

  // compute reference and verify
  c2h::host_vector<type> input_h = input;
  c2h::host_vector<type> reference_h(num_items, thrust::no_init);
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
    _CCCL_HOST_DEVICE static void check(const CustomType& obj)
    {
      _CCCL_VERIFY(reinterpret_cast<uintptr_t>(&obj) % Alignment == 0,
                   "overaligned_addable_policy_t<Alignment> is not sufficiently aligned");
    }

    _CCCL_HOST_DEVICE friend auto operator==(const CustomType& a, const CustomType& b) -> bool
    {
      check(a);
      check(b);
      return a.key == b.key;
    }

    _CCCL_HOST_DEVICE friend auto operator+(char u, const CustomType& b) -> CustomType
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
  // test with types exceeding the memcpy_async and bulk copy alignments (16 and 128 bytes respectively)
  overaligned_t<32>
#if !_CCCL_COMPILER(MSVC) // error C2719: [...] formal parameter with requested alignment of 256 won't be aligned
  ,
  overaligned_t<256>
#endif // !_CCCL_COMPILER(MSVC)
  // exhaust shared memory or registers
  ,
  huge_t>;

struct uncommon_plus
{
  // vector types
  template <typename T>
  _CCCL_HOST_DEVICE auto operator()(int8_t a, const T& b) const -> T
  {
    return T{a, 0, 0} + b;
  }

  _CCCL_HOST_DEVICE auto operator()(int8_t a, const huge_t& b) const -> huge_t
  {
    huge_t r = b;
    r.key += static_cast<size_t>(a);
    return r;
  }

  _CCCL_HOST_DEVICE auto operator()(int8_t a, const overaligned_t<32>& b) const -> overaligned_t<32>
  {
    return a + b;
  }

  _CCCL_HOST_DEVICE auto operator()(int8_t a, const overaligned_t<256>& b) const -> overaligned_t<256>
  {
    return a + b;
  }
};

C2H_TEST("DeviceTransform::Transform uncommon types", "[device][device_transform]", uncommon_types)
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
  transform_many(::cuda::std::make_tuple(a.begin(), b.begin()), result.begin(), num_items, uncommon_plus{});

  c2h::host_vector<int8_t> a_h = a;
  c2h::host_vector<type> b_h   = b;
  c2h::host_vector<type> reference_h(num_items);
  std::transform(a_h.begin(), a_h.end(), b_h.begin(), reference_h.begin(), uncommon_plus{});
  REQUIRE(c2h::host_vector<type>(result) == reference_h);
}

template <typename T>
struct nstream_kernel
{
  static constexpr T scalar = 42;

  _CCCL_HOST_DEVICE T operator()(const T& ai, const T& bi, const T& ci) const
  {
    return ai + bi + scalar * ci;
  }
};

// overwrites one input stream
C2H_TEST("DeviceTransform::Transform BabelStream nstream",
         "[device][device_transform]",
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

  transform_many(::cuda::std::make_tuple(a.begin(), b.begin(), c.begin()), a.begin(), num_items, nstream_kernel<type>{});

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

C2H_TEST("DeviceTransform::Transform add five streams", "[device][device_transform]")
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
  transform_many(::cuda::std::make_tuple(a.begin(), b.begin(), c.begin(), d.begin(), e.begin()),
                 result.begin(),
                 num_items,
                 sum_five{});

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

C2H_TEST("DeviceTransform::Transform no streams", "[device][device_transform]")
{
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  c2h::device_vector<int> result(num_items, thrust::no_init);
  transform_many(::cuda::std::tuple<>{}, result.begin(), num_items, give_me_five{});

  // compute reference and verify
  c2h::device_vector<int> reference(num_items, 5);
  REQUIRE(reference == result);
}

C2H_TEST("DeviceTransform::Transform fancy input iterator types", "[device][device_transform]")
{
  using type          = int;
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  thrust::counting_iterator<type> a{0};
  thrust::counting_iterator<type> b{10};

  c2h::device_vector<type> result(num_items, thrust::no_init);
  transform_many(::cuda::std::make_tuple(a, b), result.begin(), num_items, ::cuda::std::plus<type>{});

  // compute reference and verify
  c2h::host_vector<type> reference_h(num_items);
  std::transform(a, a + num_items, b, reference_h.begin(), std::plus<type>{});
  REQUIRE(reference_h == result);
}

C2H_TEST("DeviceTransform::Transform fancy output iterator type", "[device][device_transform]")
{
  using type          = int;
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  c2h::device_vector<type> a(num_items, 13);
  c2h::device_vector<type> b(num_items, 35);
  c2h::device_vector<type> result(num_items, thrust::no_init);

  using thrust::placeholders::_1;
  auto out = thrust::make_transform_output_iterator(result.begin(), _1 + 4);
  transform_many(::cuda::std::make_tuple(a.begin(), b.begin()), out, num_items, ::cuda::std::plus<type>{});
  REQUIRE(result == c2h::device_vector<type>(num_items, (13 + 35) + 4));
}

C2H_TEST("DeviceTransform::Transform mixed input iterator types", "[device][device_transform]")
{
  using type          = int;
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  thrust::counting_iterator<type> a{0};
  c2h::device_vector<type> b(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), b);

  c2h::device_vector<type> result(num_items, thrust::no_init);
  transform_many(::cuda::std::make_tuple(a, b.begin()), result.begin(), num_items, ::cuda::std::plus<type>{});

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

  _CCCL_HOST_DEVICE int operator()(const int& v) const
  {
    const auto i = &v - a;
    return v + b[i];
  }
};

C2H_TEST("DeviceTransform::Transform address stability", "[device][device_transform]")
{
  using type          = int;
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  c2h::device_vector<type> a(num_items, thrust::no_init);
  c2h::device_vector<type> b(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), a);
  c2h::gen(C2H_SEED(1), b);

  c2h::device_vector<type> result(num_items, thrust::no_init);
  transform_many_stable(
    ::cuda::std::make_tuple(thrust::raw_pointer_cast(a.data())),
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

  _CCCL_HOST_DEVICE explicit non_trivial(int data)
      : data(data)
  {}

  _CCCL_HOST_DEVICE non_trivial(const non_trivial& nt)
      : data(nt.data)
  {}

  _CCCL_HOST_DEVICE auto operator=(const non_trivial& nt) -> non_trivial&
  {
    data = nt.data;
    return *this;
  }

  _CCCL_HOST_DEVICE auto operator-() const -> non_trivial
  {
    return non_trivial{-data};
  }

  friend _CCCL_HOST_DEVICE auto operator==(non_trivial a, non_trivial b) -> bool
  {
    return a.data == b.data;
  }
};
static_assert(!::cuda::std::is_trivially_copyable_v<non_trivial>); // as required by the standard
static_assert(!thrust::is_trivially_relocatable_v<non_trivial>); // CUB uses this check internally

// Note(bgruber): I gave up on writing a test that checks whether the copy ctor/assignment operator is actually called
// (e.g. by tracking/counting invocations of those), since C++ allows (but not guarantees) elision of these operations.
// Also thrust algorithms perform a lot of copies in-between, so the test needs to use only raw allocations and
// iteration for setup and checking.
C2H_TEST("DeviceTransform::Transform not trivially relocatable", "[device][device_transform]")
{
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  c2h::device_vector<non_trivial> input(num_items, non_trivial{42});
  c2h::device_vector<non_trivial> result(num_items, thrust::no_init);
  transform_many(
    ::cuda::std::make_tuple(thrust::raw_pointer_cast(input.data())), result.begin(), num_items, ::cuda::std::negate<>{});

  const auto reference = c2h::device_vector<non_trivial>(num_items, non_trivial{-42});
  REQUIRE((reference == result));
}

C2H_TEST("DeviceTransform::Transform buffer start alignment",
         "[device][device_transform]",
         c2h::type_list<std::uint8_t, std::uint16_t, float, double>)
{
  using type          = c2h::get<0, TestType>;
  const int num_items = GENERATE(130, 100'000); // try to hit the small and full tile code paths
  const int offset    = GENERATE(1, 2, 4, 8, 16, 32, 64, 128); // global memory is always at least 256 byte aligned
  REQUIRE(num_items > offset);
  CAPTURE(c2h::type_name<type>(), num_items, offset);

  c2h::device_vector<type> input(num_items, thrust::no_init);
  thrust::sequence(input.begin(), input.end());
  c2h::device_vector<type> result(num_items);
  using thrust::placeholders::_1;
  transform_many(::cuda::std::make_tuple(input.begin() + offset),
                 result.begin() + offset,
                 num_items - offset,
                 _1 * 10); // FIXME(bgruber): does not work on negative

  c2h::device_vector<type> reference(num_items);
  thrust::tabulate(reference.begin() + offset, reference.end(), (_1 + offset) * 10);
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
C2H_TEST("DeviceTransform::Transform aligned_base_ptr", "[device][device_transform]")
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

C2H_TEST("DeviceTransform::Transform aligned_base_ptr", "[device][device_transform]")
{
  using It         = thrust::reverse_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int>>>;
  using kernel_arg = cub::detail::transform::kernel_arg<It>;

  STATIC_REQUIRE(::cuda::std::is_constructible_v<kernel_arg>);
  STATIC_REQUIRE(::cuda::std::is_copy_constructible_v<kernel_arg>);
}
