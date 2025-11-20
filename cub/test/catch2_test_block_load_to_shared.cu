// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_load_to_shared.cuh>
#include <cub/util_allocator.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/cmath>
#include <cuda/std/type_traits>

#include <c2h/catch2_test_helper.h>

enum struct test_mode
{
  single_copy,
  multi_copy_single_phase,
  multi_copy_multi_phase,
  multi_copy_multi_barrier,
};

template <int ItemsPerThread, int ThreadsInBlock, test_mode Mode, typename InputPointerT, typename OutputIteratorT>
__global__ void kernel(InputPointerT input, OutputIteratorT output, int num_items)
{
  using input_t         = cub::detail::it_value_t<InputPointerT>;
  using block_load2sh_t = cub::detail::BlockLoadToShared<ThreadsInBlock>;
  using storage_t       = typename block_load2sh_t::TempStorage;

  __shared__ storage_t storage;
  block_load2sh_t block_load2sh{storage};

  constexpr int tile_size                = ItemsPerThread * ThreadsInBlock;
  constexpr int num_copies               = Mode == test_mode::single_copy ? 1 : 2;
  constexpr int num_buffers              = (Mode == test_mode::multi_copy_single_phase) ? 2 : 1;
  constexpr int max_num_items_first_copy = cuda::ceil_div(tile_size, num_copies);
  constexpr int buffer_align             = block_load2sh_t::template SharedBufferAlignBytes<input_t>();
  constexpr int buffer_size = block_load2sh_t::template SharedBufferSizeBytes<input_t>(max_num_items_first_copy);
  alignas(buffer_align) __shared__ char buffer[num_buffers][buffer_size];
  cuda::std::span<const input_t> src{input, static_cast<cuda::std::size_t>(num_items)};
  cuda::std::span<char> dst_buff{buffer[0]};
  const int num_items_first_copy = cuda::std::min(num_items, max_num_items_first_copy);

  cuda::std::span<input_t> dst = block_load2sh.CopyAsync(dst_buff, src.first(num_items_first_copy));
  if constexpr (Mode == test_mode::single_copy || Mode == test_mode::multi_copy_multi_phase
                || Mode == test_mode::multi_copy_multi_barrier)
  {
    block_load2sh.CommitAndWait();

    for (int idx = threadIdx.x; idx < num_items_first_copy; idx += ThreadsInBlock)
    {
      output[idx] = dst[idx];
    }
  }
  else if constexpr (Mode == test_mode::multi_copy_single_phase)
  {
    cuda::std::span<char> dst_buff2{buffer[1]};
    cuda::std::span<input_t> dst2 = block_load2sh.CopyAsync(dst_buff2, src.subspan(num_items_first_copy));

    block_load2sh.CommitAndWait();

    for (int idx = threadIdx.x; idx < num_items; idx += ThreadsInBlock)
    {
      output[idx] = idx < num_items_first_copy ? dst[idx] : dst2[idx - num_items_first_copy];
    }
  }

  if constexpr (Mode == test_mode::multi_copy_multi_phase)
  {
    // Make sure that everyone is done reading from dst
    __syncthreads();

    dst = block_load2sh.CopyAsync(dst_buff, src.subspan(num_items_first_copy));
    block_load2sh.CommitAndWait();

    for (int idx = static_cast<int>(threadIdx.x); idx < num_items - num_items_first_copy; idx += ThreadsInBlock)
    {
      output[num_items_first_copy + idx] = dst[idx];
    }
  }
  else if constexpr (Mode == test_mode::multi_copy_multi_barrier)
  {
    block_load2sh.Invalidate();

    // Reuse TempStorage
    block_load2sh_t second_block_load2sh{storage};

    cuda::std::span<input_t> dst = second_block_load2sh.CopyAsync(dst_buff, src.subspan(num_items_first_copy));

    second_block_load2sh.CommitAndWait();

    for (int idx = static_cast<int>(threadIdx.x); idx < num_items - num_items_first_copy; idx += ThreadsInBlock)
    {
      output[num_items_first_copy + idx] = dst[idx];
    }
  }
}

template <int ItemsPerThread, int ThreadsInBlock, test_mode Mode = test_mode::single_copy, typename T, typename InputPointerT>
void test_block_load(const c2h::device_vector<T>& d_input, InputPointerT input)
{
  using block_load2sh_t                  = cub::detail::BlockLoadToShared<ThreadsInBlock>;
  using storage_t                        = typename block_load2sh_t::TempStorage;
  constexpr int tile_size                = ItemsPerThread * ThreadsInBlock;
  constexpr int num_copies               = Mode == test_mode::single_copy ? 1 : 2;
  constexpr int num_buffers              = (Mode == test_mode::multi_copy_single_phase) ? 2 : 1;
  constexpr int max_num_items_first_copy = cuda::ceil_div(tile_size, num_copies);
  constexpr int buffer_align             = block_load2sh_t::template SharedBufferAlignBytes<T>();
  constexpr int buffer_size              = block_load2sh_t::template SharedBufferSizeBytes<T>(max_num_items_first_copy);
  constexpr int total_smem = ::cuda::round_up(int{sizeof(storage_t)}, buffer_align) + num_buffers * buffer_size;
  constexpr bool sufficient_resources = total_smem <= cub::detail::max_smem_per_block;
  if constexpr (sufficient_resources)
  {
    CAPTURE(ThreadsInBlock, ItemsPerThread, d_input.size(), sufficient_resources, c2h::type_name<T>(), Mode);

    c2h::device_vector<T> d_output(d_input.size());
    kernel<ItemsPerThread, ThreadsInBlock, Mode>
      <<<1, ThreadsInBlock>>>(input, thrust::raw_pointer_cast(d_output.data()), static_cast<int>(d_input.size()));
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    REQUIRE(d_input == d_output);
  }
}

template <int ThreadsInBlockX, int ThreadsInBlockY, int ThreadsInBlockZ, typename InputPointerT, typename OutputIteratorT>
__global__ void kernel_dyn_smem_dst(InputPointerT input, OutputIteratorT output, int num_items)
{
  using input_t         = cub::detail::it_value_t<InputPointerT>;
  using block_load2sh_t = cub::detail::BlockLoadToShared<ThreadsInBlockX, ThreadsInBlockY, ThreadsInBlockZ>;
  using storage_t       = typename block_load2sh_t::TempStorage;

  __shared__ storage_t storage;
  block_load2sh_t block_load2sh{storage};

  static_assert(alignof(input_t) <= block_load2sh_t::template SharedBufferAlignBytes<char>());
  extern __shared__ char smem_buff[];
  assert(cuda::is_aligned(smem_buff, block_load2sh_t::template SharedBufferAlignBytes<input_t>()));

  constexpr int ThreadsInBlock = ThreadsInBlockX * ThreadsInBlockY * ThreadsInBlockZ;

  cuda::std::span<input_t> src{input, static_cast<cuda::std::size_t>(num_items)};
  cuda::std::span<char> dst_buff{smem_buff, cuda::std::size_t{cuda::ptx::get_sreg_dynamic_smem_size()}};

  cuda::std::span<input_t> dst = block_load2sh.CopyAsync(dst_buff, src.first(num_items));

  // also test separate Commit and Wait calls with token passing here
  auto token = block_load2sh.Commit();
  block_load2sh.Wait(::cuda::std::move(token));

  for (int idx = cub::RowMajorTid(ThreadsInBlockX, ThreadsInBlockY, ThreadsInBlockZ); idx < num_items;
       idx += ThreadsInBlock)
  {
    output[idx] = dst[idx];
  }
}

template <int ItemsPerThread, int ThreadsInBlock, typename T, typename InputPointerT>
void test_block_load_dyn_smem_dst(const c2h::device_vector<T>& d_input, InputPointerT input)
{
  constexpr int block_dim_x = ThreadsInBlock / 4;
  constexpr int block_dim_y = 2;
  constexpr int block_dim_z = 2;
  using block_load2sh_t     = cub::detail::BlockLoadToShared<block_dim_x, block_dim_y, block_dim_z>;
  constexpr int tile_size   = ItemsPerThread * ThreadsInBlock;
  constexpr int buffer_size = block_load2sh_t::template SharedBufferSizeBytes<T>(tile_size);
  CAPTURE(ThreadsInBlock, ItemsPerThread, c2h::type_name<T>());

  c2h::device_vector<T> d_output(d_input.size());
  kernel_dyn_smem_dst<block_dim_x, block_dim_y, block_dim_z>
    <<<1, dim3{block_dim_x, block_dim_y, block_dim_z}, buffer_size>>>(
      input, thrust::raw_pointer_cast(d_output.data()), static_cast<int>(d_input.size()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(d_input == d_output);
}

// %PARAM% IPT it 1:11

using types     = c2h::type_list<cuda::std::uint8_t, cuda::std::int32_t, cuda::std::int64_t>;
using vec_types = c2h::type_list<long2, double2>;

using threads_in_block = c2h::enum_type_list<int, 32, 64, 96, 128>;
using a_block_size     = c2h::enum_type_list<int, 256>;

using items_per_thread = c2h::enum_type_list<int, IPT>;

using modes =
  c2h::enum_type_list<test_mode,
                      test_mode::single_copy,
                      test_mode::multi_copy_single_phase,
                      test_mode::multi_copy_multi_phase,
                      test_mode::multi_copy_multi_barrier>;

template <class TestType>
struct params_t
{
  using type = typename c2h::get<0, TestType>;

  static constexpr int items_per_thread = c2h::get<1, TestType>::value;
  static constexpr int threads_in_block = c2h::get<2, TestType>::value;
  static constexpr int tile_size        = items_per_thread * threads_in_block;
};

C2H_TEST("Block load to shared works", "[load][block]", types, items_per_thread, threads_in_block, modes)
{
  using params             = params_t<TestType>;
  using type               = typename params::type;
  constexpr test_mode mode = c2h::get<3, TestType>::value;

  c2h::device_vector<type> d_input(GENERATE_COPY(take(10, random(0, params::tile_size))));
  c2h::gen(C2H_SEED(10), d_input);

  test_block_load<params::items_per_thread, params::threads_in_block, mode>(
    d_input, thrust::raw_pointer_cast(d_input.data()));
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

C2H_TEST("Block load to shared works with dyn smem and internal TempStorage", "[load][block]", items_per_thread)
{
  using type                     = int;
  constexpr int items_per_thread = c2h::get<0, TestType>::value;
  constexpr int threads_in_block = 64;
  constexpr int tile_size        = items_per_thread * threads_in_block;

  c2h::device_vector<type> d_input(GENERATE_COPY(take(10, random(0, tile_size))));
  c2h::gen(C2H_SEED(10), d_input);
  test_block_load_dyn_smem_dst<items_per_thread, threads_in_block>(d_input, thrust::raw_pointer_cast(d_input.data()));
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

  test_block_load<items_per_thread, threads_in_block, test_mode::single_copy, type, input_ptr_type>(
    d_input_ref, thrust::raw_pointer_cast(d_input.data()) + offset_for_elements);
}
#endif
