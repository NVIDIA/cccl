// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_load_to_shared.cuh>
#include <cub/util_allocator.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/type_traits>

#include <c2h/catch2_test_helper.h>

template <int ItemsPerThread, int ThreadsInBlock, bool SufficientResources, typename InputPointerT, typename OutputIteratorT>
__global__ void kernel(InputPointerT input, OutputIteratorT output, int num_items)
{
  if constexpr (SufficientResources)
  {
    using input_t         = cub::detail::it_value_t<InputPointerT>;
    using block_load2sh_t = cub::detail::BlockLoadToShared<ThreadsInBlock>;
    using storage_t       = typename block_load2sh_t::TempStorage;

    __shared__ storage_t storage;
    block_load2sh_t block_load2sh(storage);

    constexpr int tile_size    = ItemsPerThread * ThreadsInBlock;
    constexpr int buffer_align = block_load2sh_t::template SharedBufferAlignBytes<input_t>();
    constexpr int buffer_size  = block_load2sh_t::template SharedBufferSizeBytes<input_t>(tile_size);
    alignas(buffer_align) __shared__ char buffer[buffer_size];
    cuda::std::span<const input_t> src{input, static_cast<cuda::std::size_t>(num_items)};
    cuda::std::span<char> dst_buff{buffer};

    cuda::std::span<input_t> dst = block_load2sh.CopyAsync(dst_buff, src);
    block_load2sh.Commit();
    block_load2sh.Wait();

    for (int idx = threadIdx.x; idx < num_items; idx += ThreadsInBlock)
    {
      output[idx] = dst[idx];
    }
  }
  else
  {
    for (int idx = threadIdx.x; idx < num_items; idx += ThreadsInBlock)
    {
      output[idx] = input[idx];
    }
  }
}

template <int ItemsPerThread, int ThreadsInBlock, typename T, typename InputPointerT>
void test_block_load(const c2h::device_vector<T>& d_input, InputPointerT input)
{
  using block_load2sh_t               = cub::detail::BlockLoadToShared<ThreadsInBlock>;
  using storage_t                     = typename block_load2sh_t::TempStorage;
  constexpr int tile_size             = ItemsPerThread * ThreadsInBlock;
  constexpr int buffer_align          = block_load2sh_t::template SharedBufferAlignBytes<T>();
  constexpr int buffer_size           = block_load2sh_t::template SharedBufferSizeBytes<T>(tile_size);
  constexpr int total_smem            = ::cuda::round_up(sizeof(storage_t), buffer_align) + buffer_size;
  constexpr bool sufficient_resources = total_smem <= cub::detail::max_smem_per_block;
  CAPTURE(ThreadsInBlock, ItemsPerThread, sufficient_resources, c2h::type_name<T>());

  c2h::device_vector<T> d_output(d_input.size());
  kernel<ItemsPerThread, ThreadsInBlock, sufficient_resources>
    <<<1, ThreadsInBlock>>>(input, thrust::raw_pointer_cast(d_output.data()), static_cast<int>(d_input.size()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(d_input == d_output);
}

// %PARAM% IPT it 1:11

using types     = c2h::type_list<cuda::std::uint8_t, cuda::std::int32_t, cuda::std::int64_t>;
using vec_types = c2h::type_list<long2, double2>;

using threads_in_block = c2h::enum_type_list<int, 32, 128, 33, 65>;
using a_block_size     = c2h::enum_type_list<int, 256>;

using items_per_thread = c2h::enum_type_list<int, IPT>;

template <class TestType>
struct params_t
{
  using type = typename c2h::get<0, TestType>;

  static constexpr int items_per_thread = c2h::get<1, TestType>::value;
  static constexpr int threads_in_block = c2h::get<2, TestType>::value;
  static constexpr int tile_size        = items_per_thread * threads_in_block;
};

C2H_TEST("Block load to shared works", "[load][block]", types, items_per_thread, threads_in_block)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_input(GENERATE_COPY(take(10, random(0, params::tile_size))));
  c2h::gen(C2H_SEED(10), d_input);

  test_block_load<params::items_per_thread, params::threads_in_block>(d_input, thrust::raw_pointer_cast(d_input.data()));
}

C2H_TEST("Block load to shared works with even vector types", "[load][block]", vec_types, items_per_thread, a_block_size)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_input(GENERATE_COPY(take(10, random(0, params::tile_size))));
  c2h::gen(C2H_SEED(10), d_input);
  test_block_load<params::items_per_thread, params::threads_in_block>(d_input, thrust::raw_pointer_cast(d_input.data()));
}

C2H_TEST("Block load to shared works with custom types", "[load][block]", items_per_thread)
{
  using type                     = c2h::custom_type_t<c2h::equal_comparable_t>;
  constexpr int items_per_thread = c2h::get<0, TestType>::value;
  constexpr int threads_in_block = 64;
  constexpr int tile_size        = items_per_thread * threads_in_block;

  c2h::device_vector<type> d_input(GENERATE_COPY(take(10, random(0, tile_size))));
  c2h::gen(C2H_SEED(10), d_input);
  test_block_load<items_per_thread, threads_in_block>(d_input, thrust::raw_pointer_cast(d_input.data()));
}

#if IPT == 1
C2H_TEST("Block load to shared works with const and non-const datatype and different alignment cases",
         "[load][block]",
         c2h::type_list<const int*, int*>)
{
  using type           = int;
  using input_ptr_type = c2h::get<0, TestType>;

  const int offset_for_elements  = GENERATE_COPY(0, 1, 2, 3, 4);
  constexpr int items_per_thread = 4;
  constexpr int threads_in_block = 64;
  constexpr int tile_size        = items_per_thread * threads_in_block;

  c2h::device_vector<type> d_input_ref(tile_size);
  c2h::gen(C2H_SEED(10), d_input_ref);

  c2h::device_vector<type> d_input(tile_size + offset_for_elements);
  thrust::copy_n(d_input_ref.begin(), tile_size, d_input.begin() + offset_for_elements);

  test_block_load<items_per_thread, threads_in_block, type, input_ptr_type>(
    d_input_ref, thrust::raw_pointer_cast(d_input.data()) + offset_for_elements);
}
#endif
