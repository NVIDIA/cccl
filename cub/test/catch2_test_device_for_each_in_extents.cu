/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
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

DECLARE_LAUNCH_WRAPPER(cub::DeviceFor::ForEachInExtents, device_for_each_in_extents);

/***********************************************************************************************************************
 * Host reference
 **********************************************************************************************************************/

template <int Rank = 0, typename T, typename ExtentType, typename... IndicesType>
static auto fill_linear_impl(c2h::host_vector<T>& vector, const ExtentType&, size_t& pos, IndicesType... indices)
  _CCCL_TRAILING_REQUIRES(void)((Rank == ExtentType::rank()))
{
  vector[pos++] = {indices...};
  return void(); // nvc++ requires a return statement
}

template <int Rank = 0, typename T, typename ExtentType, typename... IndicesType>
static auto fill_linear_impl(c2h::host_vector<T>& vector, const ExtentType& ext, size_t& pos, IndicesType... indices)
  _CCCL_TRAILING_REQUIRES(void)((Rank < ExtentType::rank()))
{
  using IndexType = typename ExtentType::index_type;
  for (IndexType i = 0; i < ext.extent(Rank); ++i)
  {
    fill_linear_impl<Rank + 1>(vector, ext, pos, indices..., i);
  }
  return void(); // nvc++ requires a return statement
}

template <typename T, typename IndexType, size_t... Extents>
static void fill_linear(c2h::host_vector<T>& vector, const cuda::std::extents<IndexType, Extents...>& ext)
{
  size_t pos = 0;
  fill_linear_impl(vector, ext, pos);
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

template <typename IndexType, size_t... Dimensions>
auto build_static_extents(IndexType, cuda::std::index_sequence<Dimensions...>)
  -> cuda::std::extents<IndexType, Dimensions...>
{
  return {};
}

C2H_TEST("DeviceFor::ForEachInExtents static", "[ForEachInExtents][static][device]", index_types, dimensions)
{
  using index_type    = c2h::get<0, TestType>;
  using dims          = c2h::get<1, TestType>;
  auto ext            = build_static_extents(index_type{}, dims{});
  constexpr auto rank = ext.rank();
  using data_t        = cuda::std::array<index_type, rank>;
  using store_op_t    = LinearStore<index_type, rank>;
  c2h::device_vector<data_t> d_output(cub::detail::size(ext), data_t{});
  c2h::host_vector<data_t> h_output(cub::detail::size(ext), data_t{});
  auto d_output_raw = cuda::std::span<data_t>{thrust::raw_pointer_cast(d_output.data()), cub::detail::size(ext)};
  CAPTURE(c2h::type_name<index_type>());

  device_for_each_in_extents(ext, store_op_t{d_output_raw});
  c2h::host_vector<data_t> h_output_gpu = d_output;
  fill_linear(h_output, ext);
// MSVC error: C3546: '...': there are no parameter packs available to expand in
//             make_tuple_types.h:__make_tuple_types_flat
#if !_CCCL_COMPILER(MSVC)
  REQUIRE(h_output == h_output_gpu);
#endif // !_CCCL_COMPILER(MSVC)
}

C2H_TEST("DeviceFor::ForEachInExtents 3D dynamic", "[ForEachInExtents][dynamic][device]", index_types_dynamic)
{
  constexpr int rank = 3;
  using index_type   = c2h::get<0, TestType>;
  using data_t       = cuda::std::array<index_type, rank>;
  using store_op_t   = LinearStore<index_type, rank>;
  auto X             = GENERATE_COPY(take(3, random(2, 10)));
  auto Y             = GENERATE_COPY(take(3, random(2, 10)));
  auto Z             = GENERATE_COPY(take(3, random(2, 10)));
  cuda::std::dextents<index_type, 3> ext{X, Y, Z};
  c2h::device_vector<data_t> d_output(cub::detail::size(ext), data_t{});
  c2h::host_vector<data_t> h_output(cub::detail::size(ext), data_t{});
  auto d_output_raw = cuda::std::span<data_t>{thrust::raw_pointer_cast(d_output.data()), cub::detail::size(ext)};
  CAPTURE(c2h::type_name<index_type>(), X, Y, Z);

  device_for_each_in_extents(ext, store_op_t{d_output_raw});
  c2h::host_vector<data_t> h_output_gpu = d_output;
  fill_linear(h_output, ext);
#if !_CCCL_COMPILER(MSVC)
  REQUIRE(h_output == h_output_gpu);
#endif // !_CCCL_COMPILER(MSVC)
}

//----------------------------------------------------------------------------------------------------------------------
//

struct incrementer_t
{
  int* d_counts;

  template <class OffsetT>
  __device__ void operator()(OffsetT i, OffsetT)
  {
    atomicAdd(d_counts + i, 1); // Check if `i` was served more than once
  }
};

C2H_TEST("DeviceFor::ForEachInExtents works", "[ForEachInExtents]")
{
  constexpr int max_items  = 5000000;
  constexpr int min_items  = 1;
  using offset_t           = int;
  using ext_t              = cuda::std::dextents<offset_t, 1>;
  const offset_t num_items = GENERATE_COPY(
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));
  c2h::device_vector<int> counts(num_items);
  int* d_counts = thrust::raw_pointer_cast(counts.data());
  device_for_each_in_extents(ext_t{num_items}, incrementer_t{d_counts});

  const auto num_of_once_marked_items =
    static_cast<offset_t>(thrust::count(c2h::device_policy, counts.begin(), counts.end(), 1));
  REQUIRE(num_of_once_marked_items == num_items);
}
