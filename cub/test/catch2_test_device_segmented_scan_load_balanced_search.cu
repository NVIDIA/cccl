// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/dispatch/kernels/kernel_segmented_scan_load_balanced.cuh>

#include <cuda/std/cstdint>

#include <algorithm>
#include <tuple>
#include <vector>

#include "cub_test_macros.h"

template <typename OffsetT>
using work_t = cub::detail::segmented_scan::load_balanced_work_t<OffsetT>;

// One warp per range.
template <typename OffsetT>
__global__ void boundary_record_kernel(
  const work_t<OffsetT>* offsets,
  OffsetT num_segments,
  const work_t<OffsetT>* range_begins,
  const work_t<OffsetT>* range_ends,
  cub::detail::segmented_scan::boundary_record<OffsetT>* records)
{
  const auto record = cub::detail::segmented_scan::make_even_boundary_record<OffsetT>(
    offsets, num_segments, range_begins[blockIdx.x], range_ends[blockIdx.x]);
  if (threadIdx.x == 0)
  {
    records[blockIdx.x] = record;
  }
}

using offset_types = c2h::type_list<cuda::std::int32_t, cuda::std::int64_t>;

CUB_TEST("Load-balanced segmented scan finds the start and tail segments of a range",
         "[segmented_scan][device]",
         CUB_SMALL,
         offset_types)
{
  using offset_t = c2h::get<0, TestType>;
  using record_t = cub::detail::segmented_scan::boundary_record<offset_t>;
  using fields_t = std::tuple<work_t<offset_t>, work_t<offset_t>, offset_t, offset_t, bool>;

  // Offset counts on either side of one and two rounds of warp_threads candidates, and a count that needs three.
  const auto num_segments = static_cast<offset_t>(GENERATE(values({2, 30, 31, 32, 1023, 1024, 1056, 32768})));

  // The first segment is empty, runs of empty segments share a head, and the last three segments are empty.
  c2h::host_vector<work_t<offset_t>> h_offsets(num_segments + 1);
  h_offsets[0] = 0;
  for (offset_t segment = 0; segment < num_segments; ++segment)
  {
    const auto residue = segment % 11;
    const bool empty =
      residue == 0 || residue == 4 || residue == 5 || (num_segments > 3 && segment >= num_segments - 3);
    const auto size        = empty ? work_t<offset_t>{0} : static_cast<work_t<offset_t>>(segment * 13 % 9 + 1);
    h_offsets[segment + 1] = h_offsets[segment] + size;
  }
  const auto total = h_offsets[num_segments];

  // Ranges that begin at a head and inside a segment, and that end at the next head, inside a segment and at the end.
  std::vector<work_t<offset_t>> begins;
  std::vector<work_t<offset_t>> ends;
  const auto add = [&](work_t<offset_t> begin, work_t<offset_t> end) {
    begins.push_back(begin);
    ends.push_back(end);
  };
  for (offset_t segment = 0; segment < num_segments; ++segment)
  {
    const auto head = h_offsets[segment];
    const auto end  = h_offsets[segment + 1];
    if (head == end)
    {
      continue;
    }
    add(head, end);
    add(head, head + 1);
    add(head, total);
    if (end - head > 1)
    {
      add(head + 1, end);
    }
    if (end < total)
    {
      add(end - 1, end + 1);
    }
  }
  for (const int partitions : {1, 2, 3, 32, 33, 100})
  {
    if (static_cast<work_t<offset_t>>(partitions) <= total)
    {
      for (int partition = 0; partition < partitions; ++partition)
      {
        add(cub::detail::segmented_scan::even_split_boundary(total, partitions, partition),
            cub::detail::segmented_scan::even_split_boundary(total, partitions, partition + 1));
      }
    }
  }

  std::vector<fields_t> expected;
  for (size_t range = 0; range < begins.size(); ++range)
  {
    const auto first_segment = static_cast<offset_t>(
      std::upper_bound(h_offsets.begin(), h_offsets.end(), begins[range]) - h_offsets.begin() - 1);
    const auto tail_segment = static_cast<offset_t>(
      std::upper_bound(h_offsets.begin(), h_offsets.end(), ends[range] - 1) - h_offsets.begin() - 1);
    expected.emplace_back(
      begins[range],
      (std::max) (begins[range], h_offsets[tail_segment]),
      first_segment,
      tail_segment,
      h_offsets[first_segment] == begins[range]);
  }

  const c2h::device_vector<work_t<offset_t>> d_offsets = h_offsets;
  const c2h::device_vector<work_t<offset_t>> d_begins(begins.begin(), begins.end());
  const c2h::device_vector<work_t<offset_t>> d_ends(ends.begin(), ends.end());
  c2h::device_vector<record_t> d_records(begins.size());
  boundary_record_kernel<<<static_cast<unsigned int>(begins.size()), cub::detail::warp_threads>>>(
    thrust::raw_pointer_cast(d_offsets.data()),
    num_segments,
    thrust::raw_pointer_cast(d_begins.data()),
    thrust::raw_pointer_cast(d_ends.data()),
    thrust::raw_pointer_cast(d_records.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  const c2h::host_vector<record_t> h_records = d_records;
  std::vector<fields_t> actual;
  for (const auto& record : h_records)
  {
    actual.emplace_back(record.offset, record.tail_begin, record.first_segment, record.tail_segment, record.is_head);
  }
  REQUIRE(actual == expected);
}
