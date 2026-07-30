// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_transform.cuh>

#include <c2h/catch2_test_helper.h>

// The tile dispatch path only exists when nvcc is invoked with --enable-tile and the user opts in via
// CCCL_ENABLE_EXPERIMENTAL_TILE_TRANSFORM_DISPATCH. In any other build this file compiles to a single skipped test.
#if _CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED()
#  include <algorithm>

#  include <cuda_tile.h>

#  include "catch2_test_launch_helper.h"

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceTransform::Transform, transform_many);

namespace ct = ::cuda::tiles;

// Unary: v * v.
struct square_op
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class T>
  _CCCL_API T operator()(T v) const
  {
    return static_cast<T>(v * v);
  }
};

// Binary: a + b.
struct add_op
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class A, class B>
  _CCCL_API auto operator()(A a, B b) const
  {
    return static_cast<A>(a + b);
  }
};

CUB_NAMESPACE_BEGIN
namespace detail::transform::tile
{
template <class T>
inline constexpr bool tile_eligible_v<square_op, T, 1> = true;
template <>
struct tile_operator<square_op>
{
  using type = square_op;
};

template <class T>
inline constexpr bool tile_eligible_v<add_op, T, 2> = true;
template <>
struct tile_operator<add_op>
{
  using type = add_op;
};
} // namespace detail::transform::tile
CUB_NAMESPACE_END

// Unsigned types so arithmetic wraps deterministically and matches the host reference bit-for-bit.
using tile_types = c2h::type_list<::cuda::std::uint32_t, ::cuda::std::uint64_t>;

// Sizes span the runtime preconditions: multiples of 16 (with aligned c2h buffers) take the tile
// kernel; the others fall back to the standard CUB dispatch. Both must produce identical results.
#  define TILE_TRANSFORM_SIZES GENERATE(::cuda::std::int64_t{0}, 16, 32, 128, 1024, 4096, 65536, 17, 127, 1000)

C2H_TEST("DeviceTransform tile dispatch: unary scalar op routed through its tile_operator substitute",
         "[device][transform][tile]",
         tile_types)
{
  using type                           = c2h::get<0, TestType>;
  const ::cuda::std::int64_t num_items = TILE_TRANSFORM_SIZES;
  CAPTURE(c2h::type_name<type>(), num_items);

  c2h::device_vector<type> in(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(2), in);
  c2h::device_vector<type> result(num_items, thrust::no_init);

  transform_many(::cuda::std::make_tuple(in.begin()), result.begin(), num_items, square_op{});

  c2h::host_vector<type> in_h = in;
  c2h::host_vector<type> reference_h(num_items, thrust::no_init);
  std::transform(in_h.begin(), in_h.end(), reference_h.begin(), square_op{});
  REQUIRE(reference_h == result);
}

C2H_TEST("DeviceTransform tile dispatch: binary scalar op routed through its tile_operator substitute",
         "[device][transform][tile]",
         tile_types)
{
  using type                           = c2h::get<0, TestType>;
  const ::cuda::std::int64_t num_items = TILE_TRANSFORM_SIZES;
  CAPTURE(c2h::type_name<type>(), num_items);

  c2h::device_vector<type> a(num_items, thrust::no_init);
  c2h::device_vector<type> b(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(2), a);
  c2h::gen(C2H_SEED(2), b);
  c2h::device_vector<type> result(num_items, thrust::no_init);

  transform_many(::cuda::std::make_tuple(a.begin(), b.begin()), result.begin(), num_items, add_op{});

  c2h::host_vector<type> a_h = a;
  c2h::host_vector<type> b_h = b;
  c2h::host_vector<type> reference_h(num_items, thrust::no_init);
  std::transform(a_h.begin(), a_h.end(), b_h.begin(), reference_h.begin(), add_op{});
  REQUIRE(reference_h == result);
}

#else // !_CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED()

C2H_TEST("DeviceTransform tile dispatch requires --enable-tile", "[device][transform][tile]")
{
  SUCCEED("tile transform dispatch not enabled in this build");
}

#endif // _CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED()
