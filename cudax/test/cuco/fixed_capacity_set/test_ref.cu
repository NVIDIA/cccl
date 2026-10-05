// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/__cccl_config>
#include <cuda/buffer>
#include <cuda/functional>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/fixed_capacity_set_ref.cuh>
#include <cuda/experimental/__cuco/probing_scheme.cuh>

#include <cooperative_groups.h>
#include <testing.cuh>

#include "test_common.cuh"

using key_types = c2h::type_list<cuda::std::uint8_t, cuda::std::uint16_t, cuda::std::int32_t, cuda::std::int64_t>;
using cg_sizes  = c2h::type_list<int_c<1>, int_c<2>>;
using scopes    = c2h::type_list<cuda::std::integral_constant<cuda::thread_scope, cuda::thread_scope_device>,
                                 cuda::std::integral_constant<cuda::thread_scope, cuda::thread_scope_block>>;
using extents   = c2h::type_list<cuda::std::false_type, cuda::std::true_type>;

struct constant_hash
{
  template <class Key>
  [[nodiscard]] _CCCL_HOST_DEVICE_API cuda::std::uint32_t operator()(Key) const noexcept
  {
    return 0;
  }
};

struct exercise_ref
{
  template <class Ref>
  _CCCL_DEVICE_API void operator()(Ref ref, int* results) const
  {
    using key_type     = typename Ref::key_type;
    const auto group   = cooperative_groups::tiled_partition<Ref::cg_size>(cooperative_groups::this_thread_block());
    const int capacity = static_cast<int>(ref.capacity());
    bool success       = true;
    for (int i = 0; i < capacity; ++i)
    {
      const auto key = static_cast<key_type>(i);
      bool inserted;
      if constexpr (Ref::cg_size == 1)
      {
        success  = !ref.contains(key) && success;
        inserted = ref.insert(key);
      }
      else
      {
        success  = !ref.contains(group, key) && success;
        inserted = ref.insert(group, key);
      }
      group.sync();
      success = inserted && success;
      // Also exercise the cooperative overload when cg_size is one.
      success = ref.contains(group, key) && success;
      success = !ref.insert(group, key) && success;
      group.sync();
    }

    const auto absent = static_cast<key_type>(capacity);
    success           = !ref.insert(group, absent) && success;
    success           = !ref.contains(group, absent) && success;
    if constexpr (Ref::cg_size == 1)
    {
      success = !ref.insert(absent) && success;
      success = !ref.contains(absent) && success;
    }
    group.sync();
    if (group.thread_rank() == 0)
    {
      results[0] = static_cast<int>(success);
    }
  }
};

C2H_TEST("fixed_capacity_set external storage ref", "[container][ref]", key_types, cg_sizes, scopes, extents)
{
  using key_type                                = c2h::get<0, TestType>;
  [[maybe_unused]] constexpr int cg_size        = c2h::get<1, TestType>::value;
  [[maybe_unused]] constexpr auto scope         = c2h::get<2, TestType>::value;
  [[maybe_unused]] constexpr bool static_extent = c2h::get<3, TestType>::value;
  using probing_type                            = cudax::cuco::linear_probing<cg_size, constant_hash>;
  constexpr auto capacity                       = cudax::cuco::make_valid_capacity<probing_type, 1>(17);
  using ref_type                                = cudax::cuco::fixed_capacity_set_ref<
    key_type,
    scope,
    cuda::std::equal_to<key_type>,
    probing_type,
    1,
    static_extent ? capacity : cuda::std::dynamic_extent>;
  static_assert(cuda::std::is_same_v<typename ref_type::key_type, typename ref_type::value_type>);
  static_assert(ref_type::capacity_v == (static_extent ? capacity : cuda::std::dynamic_extent));
  static_assert(ref_type::thread_scope == scope);

  test_context context;
  const auto sentinel = static_cast<key_type>(-1);
  // Subword atomics touch a complete 32-bit word, including the final slot's padding.
  // The ref still sees only the logical capacity, not the padded backing allocation.
  constexpr auto allocation_bytes = (capacity * sizeof(key_type) + 3) / 4 * 4;
  constexpr auto allocation_size  = allocation_bytes / sizeof(key_type);
  constexpr auto alignment        = sizeof(key_type) < 4 ? 4 : sizeof(key_type);
  auto slots                      = cuda::make_buffer<key_type>(context.stream, context.mr, allocation_size, sentinel);
  REQUIRE(reinterpret_cast<cuda::std::uintptr_t>(slots.data()) % alignment == 0);
  auto results = cuda::make_buffer<int>(context.stream, context.mr, 1, 0);
  const ref_type ref{
    cudax::cuco::empty_key{sentinel}, {}, probing_type{}, typename ref_type::storage_span_type{slots.data(), capacity}};
  REQUIRE(ref.capacity() == capacity);
  REQUIRE(ref.data() == slots.data());
  REQUIRE(ref.begin() == slots.data());
  REQUIRE(ref.end() == slots.data() + capacity);
  REQUIRE(ref.empty_key_sentinel() == sentinel);

  cuda::launch(context.stream,
               cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<cg_size>()),
               exercise_ref{},
               ref,
               results.data());
  REQUIRE(context.all_equal(results.data(), 1, 1));
}
