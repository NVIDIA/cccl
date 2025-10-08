// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/config.cuh>

#include <cub/device/device_for.cuh>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <cuda/std/array>
#include <cuda/std/mdspan>
#include <cuda/std/span>

#include <c2h/catch2_test_helper.h>
#include <c2h/utility.h>
#include <catch2_test_launch_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceFor::ForEachInLayout, device_for_each_in_layout);

/***********************************************************************************************************************
 * Host reference
 **********************************************************************************************************************/

template <bool IsLayoutRight, int Position, typename T, typename ExtentType, typename... IndicesType>
static void fill_linear_impl(
  c2h::host_vector<T>& vector, [[maybe_unused]] const ExtentType& ext, size_t& pos, IndicesType... indices)
{
  if constexpr (sizeof...(IndicesType) == ExtentType::rank())
  {
    vector[pos++] = {indices...};
  }
  else
  {
    using IndexType = typename ExtentType::index_type;
    for (IndexType i = 0; i < ext.extent(Position); ++i)
    {
      if constexpr (IsLayoutRight)
      {
        fill_linear_impl<IsLayoutRight, Position + 1>(vector, ext, pos, indices..., i);
      }
      else
      {
        fill_linear_impl<IsLayoutRight, Position - 1>(vector, ext, pos, i, indices...);
      }
    }
  }
}

template <bool IsLayoutRight, typename T, typename IndexType, size_t... Extents>
static void fill_linear([[maybe_unused]] c2h::host_vector<T>& vector,
                        [[maybe_unused]] const cuda::std::extents<IndexType, Extents...>& ext)
{
  [[maybe_unused]] size_t pos = 0;
  if constexpr (sizeof...(Extents) == 0)
  {
    return;
  }
  else if constexpr (IsLayoutRight)
  {
    fill_linear_impl<IsLayoutRight, 0>(vector, ext, pos);
  }
  else
  {
    fill_linear_impl<IsLayoutRight, (sizeof...(Extents) - 1)>(vector, ext, pos);
  }
}

/***********************************************************************************************************************
 * Function Objects
 **********************************************************************************************************************/

template <typename IndexType, int Size>
struct LinearStore
{
  using data_t = cuda::std::array<IndexType, Size>;

  cuda::std::span<data_t> d_output_raw;

  template <typename... TArgs>
  __device__ void operator()(IndexType idx, TArgs... args)
  {
    static_assert(sizeof...(TArgs) == Size, "wrong number of arguments");
    d_output_raw[idx] = {args...};
  }
};

/***********************************************************************************************************************
 * TEST CASES
 **********************************************************************************************************************/

using index_types =
  c2h::type_list<int8_t,
                 uint8_t,
                 int16_t,
                 uint16_t,
                 int32_t,
                 uint32_t
#if _CCCL_HAS_INT128()
                 ,
                 int64_t,
                 uint64_t
#endif
                 >;

// int8_t/uint8_t are not enabled because they easily overflow
using index_types_dynamic =
  c2h::type_list<int16_t,
                 uint16_t,
                 int32_t,
                 uint32_t
#if _CCCL_HAS_INT128()
                 ,
                 int64_t,
                 uint64_t
#endif
                 >;

using dimensions =
  c2h::type_list<cuda::std::index_sequence<>,
                 cuda::std::index_sequence<5>,
                 cuda::std::index_sequence<5, 3>,
                 cuda::std::index_sequence<5, 3, 4>,
                 cuda::std::index_sequence<3, 2, 5, 4>>;

// TODO (fbusato): add padded layouts
using layouts = c2h::type_list<cuda::std::layout_left, cuda::std::layout_right>;

template <typename IndexType, size_t... Dimensions>
auto build_static_extents(IndexType, cuda::std::index_sequence<Dimensions...>)
  -> cuda::std::extents<IndexType, Dimensions...>
{
  return {};
}

C2H_TEST("DeviceFor::ForEachInLayout static", "[ForEachInLayout][static][device]", index_types, dimensions, layouts)
{
  using index_type    = c2h::get<0, TestType>;
  using dims          = c2h::get<1, TestType>;
  using layout_t      = c2h::get<2, TestType>;
  auto ext            = build_static_extents(index_type{}, dims{});
  using ext_t         = decltype(ext);
  using mapping_t     = typename layout_t::template mapping<ext_t>;
  constexpr auto rank = ext.rank();
  using data_t        = cuda::std::array<index_type, rank>;
  using store_op_t    = LinearStore<index_type, rank>;
  c2h::device_vector<data_t> d_output(cub::detail::size(ext), data_t{1});
  c2h::host_vector<data_t> h_output_expected(cub::detail::size(ext), data_t{2});
  auto d_output_raw = cuda::std::span<data_t>{thrust::raw_pointer_cast(d_output.data()), cub::detail::size(ext)};
  CAPTURE(c2h::type_name<index_type>(), c2h::type_name<dims>(), c2h::type_name<layout_t>());

  device_for_each_in_layout(mapping_t{ext}, store_op_t{d_output_raw});
  c2h::host_vector<data_t> h_output_gpu = d_output;
  constexpr bool is_layout_right        = cuda::std::is_same_v<layout_t, cuda::std::layout_right>;
  fill_linear<is_layout_right>(h_output_expected, ext);
// MSVC error: C3546: '...': there are no parameter packs available to expand in
//             make_tuple_types.h:__make_tuple_types_flat
#if !_CCCL_COMPILER(MSVC)
  REQUIRE(h_output_expected == h_output_gpu);
#endif // !_CCCL_COMPILER(MSVC)
}

C2H_TEST("DeviceFor::ForEachInLayout 3D dynamic", "[ForEachInLayout][dynamic][device]", index_types_dynamic, layouts)
{
  constexpr int rank = 3;
  using index_type   = c2h::get<0, TestType>;
  using layout_t     = c2h::get<1, TestType>;
  using ext_t        = cuda::std::dextents<index_type, 3>;
  using mapping_t    = typename layout_t::template mapping<ext_t>;
  using data_t       = cuda::std::array<index_type, rank>;
  using store_op_t   = LinearStore<index_type, rank>;
  auto X             = GENERATE_COPY(take(3, random(2, 10)));
  auto Y             = GENERATE_COPY(take(3, random(2, 10)));
  auto Z             = GENERATE_COPY(take(3, random(2, 10)));
  ext_t ext{X, Y, Z};
  c2h::device_vector<data_t> d_output(cub::detail::size(ext), data_t{1});
  c2h::host_vector<data_t> h_output_expected(cub::detail::size(ext), data_t{2});
  auto d_output_raw = cuda::std::span<data_t>{thrust::raw_pointer_cast(d_output.data()), cub::detail::size(ext)};
  CAPTURE(c2h::type_name<index_type>(), X, Y, Z);

  device_for_each_in_layout(mapping_t{ext}, store_op_t{d_output_raw});
  c2h::host_vector<data_t> h_output_gpu = d_output;
  constexpr bool is_layout_right        = cuda::std::is_same_v<layout_t, cuda::std::layout_right>;
  fill_linear<is_layout_right>(h_output_expected, ext);

#if !_CCCL_COMPILER(MSVC)
  REQUIRE(h_output_expected == h_output_gpu);
#endif // !_CCCL_COMPILER(MSVC)
}

//----------------------------------------------------------------------------------------------------------------------
// No duplicates

struct incrementer_t
{
  int* d_counts;

  template <typename IndexType, class OffsetT>
  __device__ void operator()(IndexType i, OffsetT)
  {
    atomicAdd(d_counts + i, 1); // Check if `i` was served more than once
  }
};

C2H_TEST("DeviceFor::ForEachInLayout no duplicates", "[ForEachInLayout][no_duplicates][device]", layouts)
{
  constexpr int min_items = 1;
  constexpr int max_items = 5000000;
  using layout_t          = c2h::get<0, TestType>;
  using ext_t             = cuda::std::dextents<int, 1>;
  using mapping_t         = typename layout_t::template mapping<ext_t>;
  const int num_items     = GENERATE_COPY(
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));
  c2h::device_vector<int> counts(num_items, 0);
  int* d_counts = thrust::raw_pointer_cast(counts.data());
  device_for_each_in_layout(mapping_t{ext_t{num_items}}, incrementer_t{d_counts});

  auto num_of_once_marked_items = thrust::count(c2h::device_policy, counts.begin(), counts.end(), 1);
  REQUIRE(num_of_once_marked_items == num_items);
}
