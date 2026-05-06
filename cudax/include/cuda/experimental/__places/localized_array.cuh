//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Implementation of the localized_array class which dispatches a VMM
 *        allocation over multiple data places using a partitioner function.
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__places/places.cuh>

#include <array>
#include <random>
#include <unordered_map>
#include <vector>

namespace cuda::experimental::places
{
/**
 * @brief Check if localized allocation statistics should be printed
 */
inline bool localized_alloc_stats_enabled()
{
  static bool enabled = [] {
    const char* env = ::std::getenv("CUDASTF_LOCALIZED_ALLOC_STATS");
    return env != nullptr && ::std::string(env) != "0";
  }();
  return enabled;
}

/**
 * @brief An allocator that takes a mapping function to dispatch an allocation over multiple data places.
 *
 * This is the mechanism used to implement the data_place of a grid of execution places.
 * Uses CUDA Virtual Memory Management (VMM) to create a contiguous virtual address range
 * backed by physical allocations on different devices according to the partitioner.
 */
class localized_array
{
  struct metadata
  {
    metadata(data_place place_, size_t size_, size_t offset_)
        : alloc_handle{}
        , place(mv(place_))
        , size(size_)
        , offset(offset_)
    {}

    CUmemGenericAllocationHandle alloc_handle;
    const data_place place;
    size_t size;
    size_t offset;
  };

public:
  template <typename F>
  localized_array(
    exec_place grid, partition_fn_t mapper, F&& delinearize, size_t total_size, size_t elemsize, dim4 data_dims)
      : grid(mv(grid))
      , mapper(mv(mapper))
      , total_size_bytes(total_size * elemsize)
      , data_dims(data_dims)
      , elemsize(elemsize)
  {
    cuda_try(cudaFree(nullptr));

    const int ndevs = cuda_try<cudaGetDeviceCount>();
    CUdevice dev    = cuda_try<cuCtxGetDevice>();

    int supportsVMM = cuda_try<cuDeviceGetAttribute>(CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, dev);
    EXPECT(supportsVMM == 1, "Cannot create a localized_array object on this machine because it does not support VMM.");

    CUmemAllocationProp prop = {};
    prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id         = dev;

    size_t alloc_granularity_bytes = cuda_try<cuMemGetAllocationGranularity>(&prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    block_size_bytes = alloc_granularity_bytes;

    vm_total_size_bytes =
      ((total_size_bytes + alloc_granularity_bytes - 1) / alloc_granularity_bytes) * alloc_granularity_bytes;

    size_t nblocks = vm_total_size_bytes / alloc_granularity_bytes;

    base_ptr = cuda_try<cuMemAddressReserve>(vm_total_size_bytes, 0ULL, 0ULL, 0ULL);

    ::std::vector<CUmemAccessDesc> accessDesc(ndevs);
    for (int d = 0; d < ndevs; d++)
    {
      accessDesc[d].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      accessDesc[d].location.id   = d;
      accessDesc[d].flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }

    block_stats stats;

    ::std::vector<pos4> owner;
    owner.reserve(nblocks);
    for (size_t i = 0; i < nblocks; i++)
    {
      owner.push_back(
        block_to_grid_pos(i * block_size_bytes / elemsize, alloc_granularity_bytes / elemsize, delinearize, stats));
    }

    meta.reserve(nblocks);

    ::std::unordered_map<::std::string, size_t> bytes_per_place;

    for (size_t i = 0; i < nblocks;)
    {
      pos4 p   = owner[i];
      size_t j = 0;
      while ((i + j < nblocks) && (owner[i + j] == p))
      {
        j++;
      }

      data_place place  = grid_pos_to_place(p);
      size_t alloc_size = j * alloc_granularity_bytes;
      bytes_per_place[place.to_string()] += alloc_size;

      meta.emplace_back(mv(place), alloc_size, i * block_size_bytes);

      i += j;
    }

    if (localized_alloc_stats_enabled())
    {
      fprintf(stderr, "\n=== Localized Array Allocation Statistics ===\n");
      fprintf(stderr, "Total size: %zu bytes (%.2f MB)\n", total_size_bytes, total_size_bytes / (1024.0 * 1024.0));
      fprintf(
        stderr, "VM reservation: %zu bytes (%.2f MB)\n", vm_total_size_bytes, vm_total_size_bytes / (1024.0 * 1024.0));
      fprintf(stderr, "Block size: %zu bytes (%.2f KB)\n", block_size_bytes, block_size_bytes / 1024.0);
      fprintf(stderr, "Number of blocks: %zu (merged into %zu allocations)\n", nblocks, meta.size());
      fprintf(stderr, "Number of places: %zu\n", bytes_per_place.size());

      fprintf(stderr, "\nAllocation distribution by place:\n");
      for (const auto& entry : bytes_per_place)
      {
        double pct = 100.0 * entry.second / vm_total_size_bytes;
        fprintf(stderr,
                "  %s: %zu bytes (%.2f MB, %.1f%%)\n",
                entry.first.c_str(),
                entry.second,
                entry.second / (1024.0 * 1024.0),
                pct);
      }

      if (stats.total_samples > 0)
      {
        double accuracy = 100.0 * stats.matching_samples / stats.total_samples;
        fprintf(stderr,
                "\nPlacement accuracy: %.1f%% (%zu/%zu samples matched chosen position)\n",
                accuracy,
                stats.matching_samples,
                stats.total_samples);
      }

      fprintf(stderr, "\nAllocation map (%zu allocations):\n", meta.size());
      fprintf(stderr, "  %-6s  %-12s  %-12s  %-10s  %s\n", "Index", "Offset", "Size", "Blocks", "Place");
      fprintf(stderr, "  %-6s  %-12s  %-12s  %-10s  %s\n", "-----", "------", "----", "------", "-----");
      for (size_t idx = 0; idx < meta.size(); idx++)
      {
        const auto& item   = meta[idx];
        size_t num_blocks  = item.size / alloc_granularity_bytes;
        size_t start_block = item.offset / alloc_granularity_bytes;
        fprintf(stderr,
                "  %-6zu  %-12zu  %-12zu  %-10zu  %s\n",
                idx,
                item.offset,
                item.size,
                num_blocks,
                item.place.to_string().c_str());
      }

      fprintf(stderr, "\nBlock ownership map (each char = 1 block, 0-9/a-z = place index):\n  ");
      ::std::unordered_map<::std::string, char> place_to_char;
      char next_char = '0';
      for (size_t i = 0; i < nblocks; i++)
      {
        ::std::string place_str = grid_pos_to_place(owner[i]).to_string();
        if (place_to_char.find(place_str) == place_to_char.end())
        {
          place_to_char[place_str] = next_char;
          if (next_char == '9')
          {
            next_char = 'a';
          }
          else
          {
            next_char++;
          }
        }
        fprintf(stderr, "%c", place_to_char[place_str]);
        if ((i + 1) % 80 == 0)
        {
          fprintf(stderr, "\n  ");
        }
      }
      fprintf(stderr, "\n");

      fprintf(stderr, "\n  Legend:\n");
      for (const auto& entry : place_to_char)
      {
        fprintf(stderr, "    %c = %s\n", entry.second, entry.first.c_str());
      }

      fprintf(stderr, "==============================================\n\n");
    }

    for (auto& item : meta)
    {
      int item_dev = device_ordinal(item.place);

      cuda_try(item.place.mem_create(&item.alloc_handle, item.size));

      _CCCL_ASSERT(item.offset + item.size <= vm_total_size_bytes, "Allocation offset out of bounds");
      cuda_try(cuMemMap(base_ptr + item.offset, item.size, 0ULL, item.alloc_handle, 0ULL));

      for (int d = 0; d < ndevs; d++)
      {
        int set_access = 1;
        if (item_dev != d)
        {
          cuda_try(cudaDeviceCanAccessPeer(&set_access, d, item_dev));

          if (!set_access)
          {
            fprintf(stderr, "Warning : Cannot enable peer access between devices %d and %d\n", d, item_dev);
          }
        }

        if (set_access == 1)
        {
          cuda_try(cuMemSetAccess(base_ptr + item.offset, item.size, &accessDesc[d], 1ULL));
        }
      }
    }
  }

  localized_array()                                  = delete;
  localized_array(const localized_array&)            = delete;
  localized_array(localized_array&&)                 = delete;
  localized_array& operator=(const localized_array&) = delete;
  localized_array& operator=(localized_array&&)      = delete;

  ~localized_array()
  {
    for (auto& item : meta)
    {
      size_t offset = item.offset;
      size_t sz     = item.size;
      cuda_try(cuMemUnmap(base_ptr + offset, sz));
      cuda_try(cuMemRelease(item.alloc_handle));
    }

    cuda_try(cuMemAddressFree(base_ptr, vm_total_size_bytes));
  }

  void* get_base_ptr() const
  {
    return reinterpret_cast<void*>(base_ptr);
  }

  /*
   * This equality operator is used to find entries in an allocation cache which match a specific request
   */
  template <typename... P>
  bool operator==(::std::tuple<P&...> t) const
  {
    // tuple arguments :
    // 0 : grid, 1 : mapper, 2 : delinearize function, 3 : total size, 4 elem_size, 5 : data_dims
    bool result = grid == ::std::get<0>(t) && mapper == ::std::get<1>(t)
               && this->total_size_bytes == ::std::get<3>(t) * ::std::get<4>(t) && elemsize == ::std::get<4>(t);
    if (result)
    {
      assert(this->total_size_bytes == ::std::get<3>(t) * ::std::get<4>(t));
      assert(data_dims == ::std::get<5>(t));
    }
    return result;
  }

private:
  data_place grid_pos_to_place(pos4 grid_pos)
  {
    return grid.get_place(grid_pos).affine_data_place();
  }

  struct block_stats
  {
    size_t total_samples    = 0;
    size_t matching_samples = 0;
  };

  template <typename F>
  pos4 block_to_grid_pos(size_t linearized_index, size_t allocation_granularity, F&& delinearize, block_stats& stats)
  {
#if 0
        return index_to_grid_pos(linearized_index, delinearize);
#else
    ::std::random_device rd;
    ::std::mt19937 gen(rd());
    ::std::uniform_int_distribution<> dis(0, static_cast<int>(allocation_granularity - 1));

    const size_t nsamples = 10;
    ::std::array<pos4, nsamples> sampled_pos;
    for (size_t sample = 0; sample < nsamples; sample++)
    {
      size_t index        = linearized_index + dis(gen);
      sampled_pos[sample] = index_to_grid_pos(index, delinearize);
    }

    ::std::unordered_map<pos4, size_t, ::cuda::experimental::stf::hash<pos4>> sample_cnt;
    for (auto& s : sampled_pos)
    {
      ++sample_cnt[s];
    }

    size_t max_cnt = 0;
    pos4 max_pos;
    for (auto& s : sample_cnt)
    {
      if (s.second > max_cnt)
      {
        max_pos = s.first;
        max_cnt = s.second;
      }
    }

    stats.total_samples += nsamples;
    stats.matching_samples += max_cnt;

    return max_pos;
#endif
  }

  template <typename F>
  pos4 index_to_grid_pos(size_t linearized_index, F&& delinearize)
  {
    pos4 coords        = delinearize(linearized_index);
    pos4 eplace_coords = mapper(coords, data_dims, grid.get_dims());
    return eplace_coords;
  }

  exec_place grid;
  partition_fn_t mapper = nullptr;
  ::std::vector<metadata> meta;

  size_t block_size_bytes = 0;
  size_t total_size_bytes = 0;

  size_t vm_total_size_bytes = 0;

  CUdeviceptr base_ptr = 0;

  dim4 data_dims;
  size_t elemsize = 0;
};

inline ::std::unordered_map<void*, ::std::unique_ptr<localized_array>>& get_composite_alloc_registry()
{
  static ::std::unordered_map<void*, ::std::unique_ptr<localized_array>> reg;
  return reg;
}

inline void* allocate_composite_data_place(const data_place_composite& p, ::std::ptrdiff_t size)
{
  const size_t size_u          = static_cast<size_t>(size);
  const exec_place& grid       = p.get_grid();
  const partition_fn_t& mapper = p.get_partitioner();
  auto delinearize_1d          = [](size_t i) {
    return pos4(static_cast<ssize_t>(i), 0, 0, 0);
  };
  auto arr  = ::std::make_unique<localized_array>(grid, mapper, delinearize_1d, size_u, 1, dim4(size_u));
  void* ptr = arr->get_base_ptr();
  get_composite_alloc_registry()[ptr] = ::std::move(arr);
  return ptr;
}

inline void deallocate_composite_data_place(void* ptr)
{
  get_composite_alloc_registry().erase(ptr);
}
} // namespace cuda::experimental::places
