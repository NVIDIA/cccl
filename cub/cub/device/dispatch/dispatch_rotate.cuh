// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_transform.cuh>
#include <cub/device/dispatch/kernels/kernel_rotate.cuh>
#include <cub/util_debug.cuh>

#include <cuda/std/functional>

#include <algorithm>
#include <climits>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <vector>

CUB_NAMESPACE_BEGIN
namespace detail
{
namespace rotate
{
// ============================================================================
// Error propagation macro (wraps CubDebug for consistent logging)
// ============================================================================

#define CUB_ROTATE_CHECK(call)         \
  do                                   \
  {                                    \
    cudaError_t _err = CubDebug(call); \
    if (_err != cudaSuccess)           \
    {                                  \
      return _err;                     \
    }                                  \
  } while (0)

// ============================================================================
// Runtime device queries
// ============================================================================

// Get number of SMs in stream's GPU.
inline cudaError_t get_num_sms(cudaStream_t stream, int& num_sms)
{
  int device;
  CUB_ROTATE_CHECK(cudaStreamGetDevice(stream, &device));
  CUB_ROTATE_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  return cudaSuccess;
}

inline cudaError_t get_launch_config(cudaStream_t stream, const int tile_bytes, int& block_size_out, int& grid_size_out)
{
  int device;
  CUB_ROTATE_CHECK(cudaStreamGetDevice(stream, &device));

  int num_sms;
  CUB_ROTATE_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  int max_threads_per_sm;
  CUB_ROTATE_CHECK(cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device));
  int shmem_per_sm;
  CUB_ROTATE_CHECK(cudaDeviceGetAttribute(&shmem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
  const auto [block_size, blocks_per_sm] = get_launch_bounds(tile_bytes, shmem_per_sm, max_threads_per_sm);
  block_size_out                         = block_size;
  grid_size_out                          = blocks_per_sm * num_sms;
  return cudaSuccess;
}

// ============================================================================
// Algorithm state (persisted between query and execution passes)
// ============================================================================

struct RotateState_t
{
  std::vector<int32_t> ordering_;
  uint32_t max_distance_ = 0;
};

// ============================================================================
// Algorithm selection
// ============================================================================

// Max dependency distance for which the long algorithm is faster than the naive one.
// Derived dynamically from the cooperative grid size.
inline size_t get_max_dependency_distance(cudaStream_t stream)
{
  int block_size, grid_size;
  get_launch_config(stream, rotate_long::TILE_BYTES, block_size, grid_size);
  return static_cast<size_t>(grid_size - 50); // TODO: could be a tuning parameter
}

constexpr int SHORT_TILE_BYTES = USE_SHORT_PIPELINE ? rotate_short::TILE_BYTES : rotate_short_no_pipeline::TILE_BYTES;

enum class RotateAlgo
{
  Short,
  Long,
  Naive
};

template <typename T>
RotateAlgo get_algorithm_to_use(size_t rotate_distance, size_t max_distance, cudaStream_t stream)
{
  if (rotate_distance <= SHORT_TILE_BYTES / sizeof(T))
  {
    return RotateAlgo::Short;
  }
  else if (max_distance < get_max_dependency_distance(stream))
  {
    return RotateAlgo::Long;
  }
  else
  {
    return RotateAlgo::Naive;
  }
}

template <typename T>
uint32_t compute_head_size(T const* d_array, size_t const size, size_t const rotate_distance)
{
  uint32_t const arr_misalignment = reinterpret_cast<uintptr_t>(d_array) % BYTES_PER_SECTOR;
  uint32_t head_size              = arr_misalignment == 0 ? 0U : (BYTES_PER_SECTOR - arr_misalignment) / sizeof(T);
  head_size                       = std::min(static_cast<size_t>(head_size), size - rotate_distance);
  assert(head_size < BYTES_PER_SECTOR);
  return head_size;
}

// ============================================================================
// BFS tile ordering (host-side)
// ============================================================================

// pos[node] = index in order
// TODO: could use vector instead
template <typename T>
uint32_t max_dependency_distance(
  const std::vector<Dependencies>& adj,
  const std::vector<int32_t>& order,
  size_t const arr_size,
  size_t const tile_size,
  size_t const rot_dist,
  uint32_t const head_size)
{
  std::unordered_map<int32_t, uint32_t> pos;
  for (uint32_t i = 0; i < order.size(); i++)
  {
    int32_t v = order[i];
    pos[v]    = i;
  }

  std::vector<int64_t> distances(order.size(), 0);

  auto const neg_head_size      = tile_detail::get_neg_head_size<T>(arr_size, rot_dist, head_size);
  auto const num_negative_tiles = tile_detail::get_num_negative_tiles(rot_dist, tile_size, neg_head_size);

  for (auto const node : order)
  {
    uint32_t node_ix  = pos[node];
    auto const adj_ix = tile_detail::tile_ix_to_arr_ix(node, num_negative_tiles);
    for (int i = 0; i < adj[adj_ix].num_dependencies_; ++i)
    {
      auto const dep    = adj[adj_ix].deps_[i];
      auto const dep_ix = pos[dep];
      int64_t distance  = static_cast<int64_t>(dep_ix) - static_cast<int64_t>(node_ix);
      distances[adj_ix] = std::max(distances[adj_ix], distance);
    }
  }

  auto max_elem = std::max_element(distances.begin(), distances.end());
  return *max_elem;
}

// TODO optimize, takes around the same time than the rotate itself
template <typename T>
RotateState_t
bfs_visit_order(size_t const arr_size, size_t const rot_dist, size_t const tile_size, uint32_t const head_size)
{
  auto const neg_head_size      = tile_detail::get_neg_head_size<T>(arr_size, rot_dist, head_size);
  const auto num_negative_tiles = tile_detail::get_num_negative_tiles(rot_dist, tile_size, neg_head_size);
  const auto num_positive_tiles = tile_detail::get_num_positive_tiles(arr_size, rot_dist, tile_size, head_size);
  const auto num_nodes          = num_negative_tiles + num_positive_tiles;
  std::vector<Dependencies> adj(num_nodes);

  for (uint32_t i = 0; i < num_negative_tiles; i++)
  {
    adj[i] = tile_detail::get_dependencies<T>(arr_size, rot_dist, tile_size, -(i + 1), num_nodes, head_size);
  }
  for (uint32_t i = 0; i < num_positive_tiles; i++)
  {
    adj[num_negative_tiles + i] =
      tile_detail::get_dependencies<T>(arr_size, rot_dist, tile_size, i, num_nodes, head_size);
  }

  std::vector<uint8_t> visited(num_nodes, 0);

  RotateState_t state;
  state.ordering_.reserve(num_nodes);

  std::vector<uint32_t> q;
  q.reserve(num_nodes);

  auto bfs_from = [&](uint32_t s) {
    assert(!visited[s]);
    visited[s] = 1;

    q.clear();
    q.push_back(s);
    size_t head = 0;

    while (head < q.size())
    {
      uint32_t u = q[head++];
      state.ordering_.push_back(tile_detail::arr_ix_to_tile_ix(u, num_negative_tiles));

      for (int i = 0; i < adj[u].num_dependencies_; ++i)
      {
        auto const v = tile_detail::tile_ix_to_arr_ix(adj[u].deps_[i], num_negative_tiles);
        if (!visited[v])
        {
          visited[v] = 1;
          q.push_back(v);
        }
      }
    }
  };

  // TODO: play with start position
  bfs_from(0);
  for (uint32_t s = 0; s < static_cast<uint32_t>(num_nodes); ++s)
  {
    if (!visited[s])
    {
      bfs_from(s);
    }
  }

  assert(state.ordering_.size() == num_nodes);

  state.max_distance_ = max_dependency_distance<T>(adj, state.ordering_, arr_size, tile_size, rot_dist, head_size);
  return state;
}

// ============================================================================
// Compute temp storage size and algorithm state for a given configuration.
// Pure host-side logic, no device memory access.
// ============================================================================

template <typename T>
void compute_temp_size_and_state(
  T* d_array, size_t size, size_t rotate_distance, cudaStream_t stream, size_t& temp_storage_bytes, RotateState_t& state)
{
  int num_sms;
  get_num_sms(stream, num_sms);

  if (rotate_distance <= SHORT_TILE_BYTES / sizeof(T))
  {
    if constexpr (USE_SHORT_PIPELINE)
    {
      temp_storage_bytes = sizeof(device_flag_t) * num_sms;
    }
    else
    {
      uint32_t const head_size_short = compute_head_size(d_array, size, rotate_distance);
      size_t const num_main_tiles =
        cuda::ceil_div((size - rotate_distance - head_size_short) * sizeof(T), static_cast<size_t>(SHORT_TILE_BYTES));
      temp_storage_bytes = sizeof(int) + sizeof(device_flag_t) * num_main_tiles;
    }
    state = RotateState_t{};
    return;
  }

  uint32_t const head_size = compute_head_size(d_array, size, rotate_distance);

  // Edge case where head_size consumes the entire positive region, so fall back to naive algorithm.
  if (size - rotate_distance <= static_cast<size_t>(head_size))
  {
    state.max_distance_ = std::numeric_limits<uint32_t>::max();
    temp_storage_bytes  = rotate_distance * sizeof(T);
    return;
  }

  state = bfs_visit_order<T>(size, rotate_distance, rotate_long::TILE_BYTES / sizeof(T), head_size);
  const auto algorithm_to_use = get_algorithm_to_use<T>(rotate_distance, state.max_distance_, stream);

  if (algorithm_to_use == RotateAlgo::Long)
  {
    auto const num_tiles = state.ordering_.size();
    // Tile counter + ordering + flags
    temp_storage_bytes = sizeof(uint32_t) + (sizeof(int32_t) + sizeof(device_flag_t)) * num_tiles;
  }
  else
  {
    // Naive algorithm: temp buffer to save the first rotate_distance elements,
    // plus whatever DeviceTransform requires for its own temp storage.
    size_t transform_temp_bytes = 0;
    T* dummy_out                = nullptr;
    cuda::std::identity id{};
    cub::DeviceTransform::Transform(
      nullptr, transform_temp_bytes, ::cuda::std::make_tuple(d_array), dummy_out, rotate_distance, id, stream);
    temp_storage_bytes = rotate_distance * sizeof(T) + transform_temp_bytes;
  }
}

// ============================================================================
// Unified dispatch: query pass (d_temp_storage == nullptr) or execution pass
//
// When d_temp_storage is nullptr (query pass):
//   - temp_storage_bytes is filled with the required allocation size
//   - state is filled with the precomputed BFS ordering
//   - No device work is performed
//
// When d_temp_storage is non-null (execution pass):
//   - state must contain the values written by a prior query pass
//   - The rotation is executed using the precomputed state
// ============================================================================

template <typename T>
CUB_RUNTIME_FUNCTION cudaError_t dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  RotateState_t& state,
  T* d_array,
  size_t size,
  size_t rotate_distance,
  cudaStream_t stream)
{
  if constexpr (sizeof(T) != alignof(T))
  {
    return dispatch(
      d_temp_storage,
      temp_storage_bytes,
      state,
      reinterpret_cast<uint8_t*>(d_array),
      size * sizeof(T),
      rotate_distance * sizeof(T),
      stream);
  }
  else
  {
    if (size <= 1 || rotate_distance == 0)
    {
      temp_storage_bytes = 0;
      return cudaSuccess;
    }
    if (rotate_distance >= size)
    {
      rotate_distance %= size;
    }
    if (rotate_distance == 0)
    {
      temp_storage_bytes = 0;
      return cudaSuccess;
    }

    // Query pass: compute temp storage requirements and fill state
    if (d_temp_storage == nullptr)
    {
      compute_temp_size_and_state(d_array, size, rotate_distance, stream, temp_storage_bytes, state);
      return cudaSuccess;
    }

    int num_sms;
    CUB_ROTATE_CHECK(get_num_sms(stream, num_sms));

    // Execution pass: use the precomputed state
    const auto algo_to_use = get_algorithm_to_use<T>(rotate_distance, state.max_distance_, stream);

    if (algo_to_use == RotateAlgo::Short)
    {
      constexpr size_t TILE_SIZE = SHORT_TILE_BYTES / sizeof(T);
      bool const is_tiny         = (2 * rotate_distance > size) || (size <= TILE_SIZE);

      if (is_tiny)
      {
        const int shmem = static_cast<int>(size * sizeof(T));
        CUB_ROTATE_CHECK(cudaFuncSetAttribute(
          rotate_short_no_pipeline::rotate_tiny_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
        rotate_short_no_pipeline::rotate_tiny_kernel<T><<<1, 512, shmem, stream>>>(d_array, size, rotate_distance);
        CUB_ROTATE_CHECK(cudaGetLastError());
      }
      else
      {
        if constexpr (USE_SHORT_PIPELINE)
        {
          auto const head_size = compute_head_size(d_array, size, rotate_distance);
          size_t const num_main_tiles =
            cuda::ceil_div((size - rotate_distance - head_size) * sizeof(T), static_cast<size_t>(SHORT_TILE_BYTES));
          assert(reinterpret_cast<uintptr_t>(d_array + head_size) % BYTES_PER_SECTOR == 0 || num_main_tiles == 0);

          rotate_short::setup_kernel<<<1, 512, 0, stream>>>(d_temp_storage, temp_storage_bytes, num_sms);
          CUB_ROTATE_CHECK(cudaGetLastError());

          const auto dynamic_shmem = rotate_short::get_shmem_usage<T>(stream);
          CUB_ROTATE_CHECK(cudaFuncSetAttribute(
            rotate_short::rotate_short_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shmem));
          rotate_short::rotate_short_kernel<T><<<num_sms, rotate_short::BLOCK_SIZE, dynamic_shmem, stream>>>(
            d_array, size, d_temp_storage, temp_storage_bytes, rotate_distance, num_main_tiles, head_size);
          CUB_ROTATE_CHECK(cudaGetLastError());
        }
        else
        {
          auto const head_size = compute_head_size(d_array, size, rotate_distance);
          size_t const num_main_tiles =
            cuda::ceil_div((size - rotate_distance - head_size) * sizeof(T), static_cast<size_t>(SHORT_TILE_BYTES));

          int block_size, grid_size;
          CUB_ROTATE_CHECK(get_launch_config(stream, rotate_short_no_pipeline::TILE_BYTES, block_size, grid_size));

          rotate_short_no_pipeline::setup_kernel<<<1, 512, 0, stream>>>(
            d_temp_storage, temp_storage_bytes, num_main_tiles);
          CUB_ROTATE_CHECK(cudaGetLastError());

          rotate_short_no_pipeline::rotate_short_kernel_no_pipeline<T><<<grid_size, block_size, 0, stream>>>(
            d_array, size, d_temp_storage, temp_storage_bytes, rotate_distance, num_main_tiles, head_size);
          CUB_ROTATE_CHECK(cudaGetLastError());
        }
      }
    }
    else if (algo_to_use == RotateAlgo::Long)
    {
      size_t const num_tiles = state.ordering_.size();
      if (num_tiles == 0)
      {
        return cudaErrorInvalidValue;
      }
      uint32_t const head_size = compute_head_size(d_array, size, rotate_distance);

      rotate_long::setup_kernel<<<num_sms, 512, 0, stream>>>(d_temp_storage, temp_storage_bytes, num_tiles);
      CUB_ROTATE_CHECK(cudaGetLastError());
      CUB_ROTATE_CHECK(cudaMemcpyAsync(
        reinterpret_cast<uint32_t*>(d_temp_storage) + 1,
        state.ordering_.data(),
        num_tiles * sizeof(int32_t),
        cudaMemcpyHostToDevice,
        stream));

      // Launch rotate_long_kernel using cooperative kernel launch
      int block_size, grid_size;
      CUB_ROTATE_CHECK(get_launch_config(stream, rotate_long::TILE_BYTES, block_size, grid_size));
      dim3 grid_dim(grid_size);
      dim3 block_dim(block_size);
      //! Make sure the variable types are exactly the same that the kernel takes
      void* kernelArgs[] = {
        (void*) &d_array,
        (void*) &size,
        (void*) &d_temp_storage,
        (void*) &temp_storage_bytes,
        (void*) &rotate_distance,
        const_cast<void*>(static_cast<const void*>(&num_tiles)),
        const_cast<void*>(static_cast<const void*>(&head_size))};
      CUB_ROTATE_CHECK(cudaLaunchCooperativeKernel(
        (void*) rotate_long::rotate_long_kernel<T>, grid_dim, block_dim, kernelArgs, 0, stream));
    }
    else
    {
      T* d_save_buf = reinterpret_cast<T*>(d_temp_storage);
      cuda::std::identity id{};
      ::cuda::stream_ref env{stream};

      // Copy first elems into tmp buffer
      CUB_ROTATE_CHECK(cub::DeviceTransform::__transform_internal(
        ::cuda::std::make_tuple(d_array), d_save_buf, rotate_distance, ::cuda::always_true{}, id, env));

      // Copy last elems chunk-wise into the start of the array
      for (size_t offset = rotate_distance; offset < size; offset += rotate_distance)
      {
        size_t const elems_to_copy = std::min(rotate_distance, size - offset);
        CUB_ROTATE_CHECK(cub::DeviceTransform::__transform_internal(
          ::cuda::std::make_tuple(d_array + offset),
          d_array + offset - rotate_distance,
          elems_to_copy,
          ::cuda::always_true{},
          id,
          env));
      }

      // Copy saved elems into end of the array
      CUB_ROTATE_CHECK(cub::DeviceTransform::__transform_internal(
        ::cuda::std::make_tuple(d_save_buf),
        d_array + size - rotate_distance,
        rotate_distance,
        ::cuda::always_true{},
        id,
        env));
    }

    return cudaSuccess;
  } // else (aligned type)
}

#undef CUB_ROTATE_CHECK
} // namespace rotate
} // namespace detail
CUB_NAMESPACE_END
