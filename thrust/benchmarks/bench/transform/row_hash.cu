// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <cuda/std/cstdint>

#include <nvbench_helper.cuh>

// Regression benchmark for https://github.com/NVIDIA/cccl/issues/9070: thrust::tabulate was found to be up to ~4x
// slower than the equivalent thrust::transform (from a counting_iterator) or thrust::for_each_n for a row-wise
// functor that reads a wide row (many columns) and writes a single output value, such as a row hash used by
// libcudf. The three benchmarks below share the same workload so their timings can be compared directly.

namespace
{
using hash_type = cuda::std::uint32_t;

__host__ __device__ hash_type hash_combine(hash_type lhs, hash_type rhs)
{
  return lhs ^ (rhs + 0x9e3779b9u + (lhs << 6) + (lhs >> 2));
}

__host__ __device__ hash_type mix64_to_32(cuda::std::uint64_t key, hash_type seed)
{
  key ^= static_cast<cuda::std::uint64_t>(seed) + 0x9e3779b97f4a7c15ull;
  key ^= key >> 33;
  key *= 0xff51afd7ed558ccdull;
  key ^= key >> 33;
  key *= 0xc4ceb9fe1a85ec53ull;
  key ^= key >> 33;
  return static_cast<hash_type>(key ^ (key >> 32));
}

// columns are stored column-major: columns[col * num_rows + row]
struct row_hasher
{
  const cuda::std::uint64_t* columns{};
  int num_rows{};
  int num_cols{};
  hash_type seed{};

  __device__ hash_type operator()(int row_index) const
  {
    auto hash = mix64_to_32(columns[row_index], seed);
    for (int column_index = 1; column_index < num_cols; ++column_index)
    {
      const auto value = columns[static_cast<std::size_t>(column_index) * num_rows + row_index];
      hash             = hash_combine(hash, mix64_to_32(value, seed));
    }
    return hash;
  }
};

row_hasher make_row_hasher(nvbench::state& state, thrust::device_vector<cuda::std::uint64_t>& columns)
{
  const auto num_rows = static_cast<int>(state.get_int64("NumRows"));
  const auto num_cols = static_cast<int>(state.get_int64("NumCols"));

  columns.resize(static_cast<std::size_t>(num_rows) * num_cols);
  auto* d_columns = thrust::raw_pointer_cast(columns.data());
  thrust::for_each_n(
    thrust::device,
    thrust::make_counting_iterator(0),
    static_cast<int>(columns.size()),
    [d_columns, num_rows] __device__(int index) {
      const auto row = index % num_rows;
      const auto col = index / num_rows;
      d_columns[index] =
        (static_cast<cuda::std::uint64_t>(col) << 32) ^ static_cast<cuda::std::uint64_t>(row * 1315423911u + col);
    });

  state.add_global_memory_reads<cuda::std::uint64_t>(static_cast<std::size_t>(num_rows) * num_cols);
  state.add_global_memory_writes<hash_type>(num_rows);

  return row_hasher{d_columns, num_rows, num_cols, /* seed = */ 0};
}

void row_hash_tabulate(nvbench::state& state)
{
  thrust::device_vector<cuda::std::uint64_t> columns;
  const auto hasher = make_row_hasher(state, columns);
  thrust::device_vector<hash_type> output(hasher.num_rows, thrust::no_init);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               thrust::tabulate(policy(alloc, launch), output.begin(), output.end(), hasher);
             });
}

void row_hash_transform(nvbench::state& state)
{
  thrust::device_vector<cuda::std::uint64_t> columns;
  const auto hasher = make_row_hasher(state, columns);
  thrust::device_vector<hash_type> output(hasher.num_rows, thrust::no_init);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               thrust::transform(
                 policy(alloc, launch),
                 thrust::make_counting_iterator(0),
                 thrust::make_counting_iterator(hasher.num_rows),
                 output.begin(),
                 hasher);
             });
}

void row_hash_for_each(nvbench::state& state)
{
  thrust::device_vector<cuda::std::uint64_t> columns;
  const auto hasher = make_row_hasher(state, columns);
  thrust::device_vector<hash_type> output(hasher.num_rows, thrust::no_init);
  auto* output_ptr = thrust::raw_pointer_cast(output.data());

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               thrust::for_each_n(
                 policy(alloc, launch),
                 thrust::make_counting_iterator(0),
                 hasher.num_rows,
                 [hasher, output_ptr] __device__(int row_index) {
                   output_ptr[row_index] = hasher(row_index);
                 });
             });
}
} // namespace

NVBENCH_BENCH(row_hash_tabulate)
  .set_name("row_hash_tabulate")
  .add_int64_axis("NumRows", {1048576})
  .add_int64_axis("NumCols", {8, 32, 128, 256});

NVBENCH_BENCH(row_hash_transform)
  .set_name("row_hash_transform")
  .add_int64_axis("NumRows", {1048576})
  .add_int64_axis("NumCols", {8, 32, 128, 256});

NVBENCH_BENCH(row_hash_for_each)
  .set_name("row_hash_for_each")
  .add_int64_axis("NumRows", {1048576})
  .add_int64_axis("NumCols", {8, 32, 128, 256});
