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
#include <memory>
#include <new>
#include <utility>
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

// Avoid initializing all members of std::vector
template <typename T>
struct UninitializedAllocator : std::allocator<T>
{
  using value_type = T;

  template <typename U>
  struct rebind
  {
    using other = UninitializedAllocator<U>;
  };

  template <typename U>
  UninitializedAllocator(UninitializedAllocator<U> const&) noexcept
  {}

  UninitializedAllocator() noexcept = default;

  template <typename U>
  void construct(U* ptr)
  {
    ::new (static_cast<void*>(ptr)) U;
  }

  template <typename U, typename Arg, typename... Args>
  void construct(U* ptr, Arg&& arg, Args&&... args)
  {
    ::new (static_cast<void*>(ptr)) U(std::forward<Arg>(arg), std::forward<Args>(args)...);
  }
};

template <typename T, typename U>
bool operator==(UninitializedAllocator<T> const&, UninitializedAllocator<U> const&)
{
  return true;
}

template <typename T, typename U>
bool operator!=(UninitializedAllocator<T> const&, UninitializedAllocator<U> const&)
{
  return false;
}

struct RotateState_t
{
  // Dense tile indices in [0, ordering_.size()). The long kernel maps each
  // claimed index to the signed tile coordinate used by the copy helpers.
  std::vector<uint32_t, UninitializedAllocator<uint32_t>> ordering_;
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

constexpr int SHORT_TILE_BYTES = rotate_short::TILE_BYTES;

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

template <RotDir Dir, typename T>
uint32_t compute_head_size(T const* d_array, size_t const size, size_t const rotate_distance)
{
  uintptr_t const boundary_address     = reinterpret_cast<uintptr_t>(d_array + (Dir == RotDir::Right ? size : 0));
  uint32_t const boundary_misalignment = boundary_address % BYTES_PER_SECTOR;
  uint32_t head_size = Dir == RotDir::Left ? ((BYTES_PER_SECTOR - boundary_misalignment) % BYTES_PER_SECTOR) / sizeof(T)
                                           : boundary_misalignment / sizeof(T);
  head_size          = std::min(static_cast<size_t>(head_size), size - rotate_distance);
  assert(head_size < BYTES_PER_SECTOR);
  return head_size;
}

// ============================================================================
// BFS tile ordering (host-side)
// ============================================================================
// Clang 14 rejects templating the function
#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((target_clones("avx2", "default"))) inline void
fill_arithmetic_sequence(uint32_t* output, uint32_t const count, uint32_t const first)
{
#  if defined(__clang__)
#    pragma clang loop vectorize_width(8) interleave_count(1)
  for (uint32_t i = 0; i < count; ++i)
  {
    output[i] = first + i;
  }
#  else
#    pragma GCC ivdep
  for (uint32_t i = 0; i < count; ++i)
  {
    output[i] = first + i;
  }
#  endif
}

__attribute__((target_clones("avx2", "default"))) inline void
fill_arithmetic_sequence(int32_t* output, uint32_t const count, int32_t const first)
{
#  if defined(__clang__)
#    pragma clang loop vectorize_width(8) interleave_count(1)
  for (uint32_t i = 0; i < count; ++i)
  {
    output[i] = first + static_cast<int32_t>(i);
  }
#  else
#    pragma GCC ivdep
  for (uint32_t i = 0; i < count; ++i)
  {
    output[i] = first + static_cast<int32_t>(i);
  }
#  endif
}
#endif

enum class SmallSide
{
  Negative,
  Positive
};

// Half-open interval of tile indices that a given tile depends on.
struct DependencySegment
{
  uint32_t begin_;
  uint32_t end_;
  uint8_t offsets_;
};

struct DependencySegments
{
  DependencySegment segments_[19]; // Since there can be at most 20 boundaries. See `get_dependency_segments` for an
                                   // explanation.
  uint32_t size_ = 0;
};

// Given a certain rotate configuration, compute the offset of the dependency of a tile of a given candidate.
uint32_t dependency_offset(uint32_t const num_positive_tiles, uint32_t const num_tiles, uint32_t const candidate)
{
  return (num_positive_tiles + num_tiles - 2 + candidate) % num_tiles;
}

// For the dependencies of a given tile, compute the bitmask of which of the five possible offsets are present: {P-2,
// P-1, P, P+1, P+2}.
template <typename T>
uint8_t dependency_offsets_at(
  uint32_t const index,
  size_t const arr_size,
  size_t const rot_dist,
  uint32_t const head_size,
  uint32_t const neg_head_size,
  uint32_t const num_negative_tiles,
  uint32_t const num_positive_tiles)
{
  constexpr uint32_t TILE_SIZE = rotate_long::TILE_BYTES / sizeof(T);
  uint32_t const num_nodes     = num_negative_tiles + num_positive_tiles;
  uint8_t mask                 = 0;
  int32_t const tile           = tile_detail::arr_ix_to_tile_ix(index, num_negative_tiles);
  auto const dependencies =
    tile_detail::get_dependencies(arr_size, rot_dist, TILE_SIZE, tile, head_size, neg_head_size, num_negative_tiles);
  for (uint32_t dependency = dependencies.begin_; dependency < dependencies.end_; ++dependency)
  {
    uint32_t const offset = dependency >= index ? dependency - index : dependency + num_nodes - index;
    bool matched          = false;
    for (uint32_t candidate = 0; candidate < 5; ++candidate)
    {
      if (offset == dependency_offset(num_positive_tiles, num_nodes, candidate))
      {
        mask |= static_cast<uint8_t>(1u << candidate);
        matched = true;
      }
    }
    (void) matched;
    assert(matched);
  }
  assert(mask != 0);
  return mask;
}

template <typename T>
DependencySegments get_dependency_segments(
  size_t const arr_size,
  size_t const rot_dist,
  uint32_t const head_size,
  uint32_t const neg_head_size,
  uint32_t const num_negative_tiles,
  uint32_t const num_positive_tiles)
{
  uint32_t const num_nodes = num_negative_tiles + num_positive_tiles;
  // Places in the array where the dependency pattern changes.
  // Inside of a boundary all nodes have the dependency pattern: (i + {X}) % N, where X is a subset of {P-2, P-1, P,
  // P+1, P+2}.
  // There are at most 20 boundaries, since the dependency pattern can change only at 0, M, 2M mod N, or N, and each of
  // those can be +/- 2. So 4 * 5 = 20. M = num_negative_tiles, N = num_nodes.
  // 0 -> Possible partial tile at the start of the array.
  // M -> Possible partial tile at the end of the negative region and partial tile at the start of the positive region.
  //      Also where the tiles that depend on the first possible partial negative tile are.
  // 2M mod N -> where the tiles that depend on the last possible partial negative tile and the first possible partial
  // positive tile are.
  // N -> Possible partial tile at the end of the array.
  uint32_t boundaries[20];
  uint32_t num_boundaries = 0;
  auto add_boundary       = [&](int64_t const boundary) {
    if (boundary >= 0 && boundary <= num_nodes)
    {
      boundaries[num_boundaries++] = static_cast<uint32_t>(boundary);
    }
  };

  // Add all possible boundaries
  for (uint32_t const center : {0u, num_negative_tiles, (2u * num_negative_tiles) % num_nodes, num_nodes})
  {
    for (int64_t delta = -2; delta <= 2; ++delta)
    {
      add_boundary(static_cast<int64_t>(center) + delta);
    }
  }

  std::sort(boundaries, boundaries + num_boundaries);
  auto const unique_end = std::unique(boundaries, boundaries + num_boundaries);
  num_boundaries        = static_cast<uint32_t>(unique_end - boundaries);

  DependencySegments result;
  for (uint32_t i = 0; i + 1 < num_boundaries; ++i)
  {
    uint32_t const begin = boundaries[i];
    uint32_t const end   = boundaries[i + 1];
    if (begin == end)
    {
      continue;
    }
    uint8_t const offsets = dependency_offsets_at<T>(
      begin, arr_size, rot_dist, head_size, neg_head_size, num_negative_tiles, num_positive_tiles);
    // If the last segment has the same offsets, merge them into one segment.
    if (result.size_ > 0 && result.segments_[result.size_ - 1].offsets_ == offsets)
    {
      result.segments_[result.size_ - 1].end_ = end;
    }
    else
    {
      result.segments_[result.size_++] = {begin, end, offsets};
    }
  }
  assert(result.size_ > 0 && result.segments_[0].begin_ == 0 && result.segments_[result.size_ - 1].end_ == num_nodes);
  return result;
}

// Take advantage of the extreme rotate distance to simplify the dependency ordering creation.
template <SmallSide Side>
RotateState_t small_side_visit_order(
  uint32_t const num_negative_tiles, uint32_t const num_positive_tiles, DependencySegments const& dependency_segments)
{
  uint32_t const num_nodes = num_negative_tiles + num_positive_tiles;

  RotateState_t state;
  state.ordering_.resize(num_nodes);
  uint32_t* output = state.ordering_.data();

  if constexpr (Side == SmallSide::Negative)
  {
    // In dense node coordinates, every real dependency advances by one of the
    // five offsets from -(num_negative_tiles + 2) through
    // -(num_negative_tiles - 2). Visiting node zero followed by all other nodes
    // in descending order therefore gives a tight dependency bound.
    output[0] = 0;
  }

  constexpr uint32_t first = Side == SmallSide::Negative ? 1 : 0;
  for (uint32_t i = first; i < num_nodes; ++i)
  {
    if constexpr (Side == SmallSide::Negative)
    {
      output[i] = num_nodes - i;
    }
    else
    {
      // Ascending dense coordinates make each P-2/.../P+2 dependency advance by
      // at most P+2 positions; modular wraparound dependencies point backward.
      output[i] = i;
    }
  }

  for (uint32_t segment_index = 0; segment_index < dependency_segments.size_; ++segment_index)
  {
    auto const segment = dependency_segments.segments_[segment_index];
    for (uint32_t candidate = 0; candidate < 5; ++candidate)
    {
      if ((segment.offsets_ & (1u << candidate)) == 0)
      {
        continue;
      }
      uint32_t const offset = dependency_offset(num_positive_tiles, num_nodes, candidate);
      if constexpr (Side == SmallSide::Positive)
      {
        if (offset > 0 && segment.begin_ < std::min(segment.end_, num_nodes - offset))
        {
          state.max_distance_ = std::max(state.max_distance_, offset);
        }
      }
      else if (offset > 0)
      {
        bool const contains_zero        = segment.begin_ == 0;
        bool const contains_wrap_source = segment.end_ > num_nodes - offset + 1;
        if (contains_zero || contains_wrap_source)
        {
          state.max_distance_ = std::max(state.max_distance_, num_nodes - offset);
        }
      }
    }
  }
  return state;
}

struct VisitInterval
{
  uint32_t begin_;
  uint32_t end_;
};

#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((target_clones("avx2", "default")))
#endif
uint32_t max_position_distance(int32_t const* source, int32_t const* target, uint32_t const count)
{
  int32_t result = 0;
#if defined(__clang__)
#  pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUC__)
#  pragma GCC ivdep
#endif
  for (uint32_t i = 0; i < count; ++i)
  {
    result = std::max(result, target[i] - source[i]);
  }
  return static_cast<uint32_t>(result);
}

void set_exact_max_dependency_distance(
  RotateState_t& state,
  uint32_t const num_positive_tiles,
  DependencySegments const& dependency_segments,
  int32_t const* positions)
{
  uint32_t const num_nodes = static_cast<uint32_t>(state.ordering_.size());
  state.max_distance_      = 0;
  for (uint32_t segment_index = 0; segment_index < dependency_segments.size_; ++segment_index)
  {
    auto const segment = dependency_segments.segments_[segment_index];
    for (uint32_t candidate = 0; candidate < 5; ++candidate)
    {
      if ((segment.offsets_ & (1u << candidate)) == 0)
      {
        continue;
      }
      uint32_t const offset     = dependency_offset(num_positive_tiles, num_nodes, candidate);
      uint32_t const wrap       = num_nodes - offset;
      uint32_t const linear_end = std::min(segment.end_, wrap);
      if (segment.begin_ < linear_end)
      {
        state.max_distance_ =
          std::max(state.max_distance_,
                   max_position_distance(
                     positions + segment.begin_, positions + segment.begin_ + offset, linear_end - segment.begin_));
      }
      uint32_t const wrap_begin = std::max(segment.begin_, wrap);
      if (wrap_begin < segment.end_)
      {
        state.max_distance_ =
          std::max(state.max_distance_,
                   max_position_distance(
                     positions + wrap_begin, positions + wrap_begin - (num_nodes - offset), segment.end_ - wrap_begin));
      }
    }
  }
}

// Given a range of nodes, append the subranges that have not yet been visited to the output ordering and mark them as
// visited.
template <typename AppendRange>
_CCCL_FORCEINLINE void append_unvisited_linear(
  std::vector<VisitInterval>& visited, uint32_t const begin, uint32_t const end, AppendRange& append_range)
{
  if (begin == end)
  {
    return;
  }

  // TODO: Clang 14 lowers this search to a serial cmov chain that is slower than GCC's branchy implementation on x86.
  auto first =
    std::lower_bound(visited.begin(), visited.end(), begin, [](VisitInterval const interval, uint32_t point) {
      return interval.end_ < point;
    });
  auto last             = first;
  uint32_t cursor       = begin;
  uint32_t merged_begin = begin;
  uint32_t merged_end   = end;
  while (last != visited.end() && last->begin_ <= end)
  {
    uint32_t const gap_end = std::min(end, last->begin_);
    if (cursor < gap_end)
    {
      append_range(cursor, gap_end - cursor);
    }
    cursor       = std::min(end, std::max(cursor, last->end_));
    merged_begin = std::min(merged_begin, last->begin_);
    merged_end   = std::max(merged_end, last->end_);
    ++last;
  }
  if (cursor < end)
  {
    append_range(cursor, end - cursor);
  }

  if (first == last)
  {
    visited.insert(first, {merged_begin, merged_end});
  }
  else
  {
    first->begin_ = merged_begin;
    first->end_   = merged_end;
    visited.erase(first + 1, last);
  }
}

// Create the dependency order by using dependency ranges. Starting from tile 0, recursively get the dependency range of
// a range of nodes, and add ones that are not yet in the ordering. That dependency range becomes the range of nodes in
// the next iteration.
// TODO: performance can be improved for certain degenerate cases.
RotateState_t circulant_interval_visit_order(
  uint32_t const num_negative_tiles, uint32_t const num_positive_tiles, DependencySegments const& dependency_segments)
{
  uint32_t const num_nodes = num_negative_tiles + num_positive_tiles;

  RotateState_t state;
  state.ordering_.resize(num_nodes);
  uint32_t* output     = state.ordering_.data();
  uint32_t output_size = 0;

  std::vector<int32_t, UninitializedAllocator<int32_t>> position_storage(num_nodes);
  int32_t* positions = position_storage.data();

  std::vector<VisitInterval> visited;
  // Reserve a small amount of space to avoid repeated reallocations in the common case of a small number of nodes.
  visited.reserve(std::min(num_nodes, 512u));

  // This function is in the hot path, so the specific compiler and arch optimizations make a difference.
  auto append_range = [&](uint32_t start, uint32_t length) {
    while (length > 0)
    {
      uint32_t const count = std::min(length, num_nodes - start);
      uint32_t const first = start;
#if defined(__x86_64__) && defined(__GNUC__)
      fill_arithmetic_sequence(output + output_size, count, first);
      fill_arithmetic_sequence(positions + start, count, static_cast<int32_t>(output_size));
      output_size += count;
#else
      uint32_t* const range_output = output + output_size;
#  if defined(__GNUC__)
#    pragma GCC ivdep
#  endif
      for (uint32_t i = 0; i < count; ++i)
      {
        range_output[i]      = first + i;
        positions[start + i] = static_cast<int32_t>(output_size + i);
      }
      output_size += count;
#endif
      start += count;
      length -= count;
    }
  };
  auto append_unvisited = [&](uint32_t const start, uint32_t const length) {
    uint32_t const first_length = std::min(length, num_nodes - start);
    append_unvisited_linear(visited, start, start + first_length, append_range);
    append_unvisited_linear(visited, 0, length - first_length, append_range);
  };

  uint32_t layer_start = 0;
  uint32_t layer       = 0;
  while (output_size < num_nodes)
  {
    uint32_t const layer_length = layer >= num_nodes / 2 ? num_nodes : 2 * layer + 1;
    append_unvisited(layer_start, layer_length);
    ++layer;
    layer_start += num_positive_tiles - 1;
    if (layer_start >= num_nodes)
    {
      layer_start -= num_nodes;
    }
  }

  set_exact_max_dependency_distance(state, num_positive_tiles, dependency_segments, positions);

  return state;
}

// Build a dependency-safe tile order and its maximum forward dependency distance.
template <typename T>
RotateState_t bfs_visit_order(size_t const arr_size, size_t const rot_dist, uint32_t const head_size)
{
  constexpr size_t TILE_SIZE    = rotate_long::TILE_BYTES / sizeof(T);
  auto const neg_head_size      = tile_detail::get_neg_head_size<T>(arr_size, rot_dist, head_size);
  const auto num_negative_tiles = tile_detail::get_num_negative_tiles(rot_dist, TILE_SIZE, neg_head_size);
  const auto num_positive_tiles = tile_detail::get_num_positive_tiles(arr_size, rot_dist, TILE_SIZE, head_size);
  auto const dependency_segments =
    get_dependency_segments<T>(arr_size, rot_dist, head_size, neg_head_size, num_negative_tiles, num_positive_tiles);

  // A descending order is both cheap to generate and has a tight dependency
  // bound while the negative side is small.
  constexpr uint32_t max_direct_dependency_distance = 256;
  if (num_negative_tiles + 1 <= max_direct_dependency_distance)
  {
    return small_side_visit_order<SmallSide::Negative>(num_negative_tiles, num_positive_tiles, dependency_segments);
  }
  // At an exact-half rotation, alignment rounding can make the positive side one tile smaller than the negative side.
  if (num_positive_tiles + 1 <= max_direct_dependency_distance)
  {
    return small_side_visit_order<SmallSide::Positive>(num_negative_tiles, num_positive_tiles, dependency_segments);
  }

  return circulant_interval_visit_order(num_negative_tiles, num_positive_tiles, dependency_segments);
}

// ============================================================================
// Compute temp storage size and algorithm state for a given configuration.
// Pure host-side logic, no device memory access.
// ============================================================================

template <RotDir Dir, typename T>
void compute_temp_size_and_state(
  T* d_array, size_t size, size_t rotate_distance, cudaStream_t stream, size_t& temp_storage_bytes, RotateState_t& state)
{
  assert(rotate_distance > 0 && rotate_distance <= size / 2);
  uint32_t const head_size = compute_head_size<Dir>(d_array, size, rotate_distance);
  if (rotate_distance <= SHORT_TILE_BYTES / sizeof(T))
  {
    size_t const num_main_tiles =
      cuda::ceil_div((size - rotate_distance - head_size) * sizeof(T), static_cast<size_t>(SHORT_TILE_BYTES));
    temp_storage_bytes = sizeof(int) + sizeof(device_flag_t) * num_main_tiles;
    state              = RotateState_t{};
    return;
  }

  assert(static_cast<size_t>(head_size) < size - rotate_distance);

  state                       = bfs_visit_order<T>(size, rotate_distance, head_size);
  const auto algorithm_to_use = get_algorithm_to_use<T>(rotate_distance, state.max_distance_, stream);

  if (algorithm_to_use == RotateAlgo::Long)
  {
    auto const num_tiles = state.ordering_.size();
    // Tile counter + ordering + flags
    temp_storage_bytes = sizeof(uint32_t) + (sizeof(uint32_t) + sizeof(device_flag_t)) * num_tiles;
  }
  else
  {
    // Naive algorithm: temp buffer to save the wrapped rotate_distance elements,
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

template <typename T, RotDir Dir = RotDir::Left>
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
    return dispatch<uint8_t, Dir>(
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
    if (rotate_distance > size / 2 && Dir == RotDir::Left)
    {
      return dispatch<T, RotDir::Right>(
        d_temp_storage, temp_storage_bytes, state, d_array, size, size - rotate_distance, stream);
    }
    assert(rotate_distance <= size / 2);

    // Query pass: compute temp storage requirements and fill state
    if (d_temp_storage == nullptr)
    {
      compute_temp_size_and_state<Dir>(d_array, size, rotate_distance, stream, temp_storage_bytes, state);
      return cudaSuccess;
    }

    // Execution pass: use the precomputed state
    const auto algo_to_use = get_algorithm_to_use<T>(rotate_distance, state.max_distance_, stream);

    if (algo_to_use == RotateAlgo::Short)
    {
      constexpr size_t TILE_SIZE = SHORT_TILE_BYTES / sizeof(T);
      bool const is_tiny         = size <= TILE_SIZE;

      if (is_tiny)
      {
        const int shmem = static_cast<int>(size * sizeof(T));
        CUB_ROTATE_CHECK(cudaFuncSetAttribute(
          rotate_short::rotate_tiny_kernel<Dir, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
        rotate_short::rotate_tiny_kernel<Dir, T><<<1, 512, shmem, stream>>>(d_array, size, rotate_distance);
        CUB_ROTATE_CHECK(cudaGetLastError());
      }
      else
      {
        auto const head_size = compute_head_size<Dir>(d_array, size, rotate_distance);
        size_t const num_main_tiles =
          cuda::ceil_div((size - rotate_distance - head_size) * sizeof(T), static_cast<size_t>(SHORT_TILE_BYTES));

        int block_size, grid_size;
        // Reuse get_launch_config by scaling the tile footprint by the stage count: each block
        // stages PIPELINE_STAGES shmem tiles, so its per-block shmem cost is PIPELINE_STAGES *
        // TILE_BYTES (reproduces the device-side rotate_short::LAUNCH_BOUNDS on the real device).
        CUB_ROTATE_CHECK(
          get_launch_config(stream, rotate_short::TILE_BYTES * rotate_short::PIPELINE_STAGES, block_size, grid_size));

        CUB_ROTATE_CHECK(cudaMemsetAsync(d_temp_storage, 0, temp_storage_bytes, stream));

        // Multi-stage pipeline holds PIPELINE_STAGES tile buffers in dynamic shared memory.
        constexpr int dynamic_shmem = rotate_short::get_shmem_usage<T>();
        CUB_ROTATE_CHECK(cudaFuncSetAttribute(
          rotate_short::rotate_short_kernel<Dir, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shmem));
        rotate_short::rotate_short_kernel<Dir, T><<<grid_size, block_size, dynamic_shmem, stream>>>(
          d_array, size, d_temp_storage, rotate_distance, num_main_tiles, head_size);
        CUB_ROTATE_CHECK(cudaGetLastError());
      }
    }
    else if (algo_to_use == RotateAlgo::Long)
    {
      size_t const num_tiles = state.ordering_.size();
      if (num_tiles == 0)
      {
        return cudaErrorInvalidValue;
      }
      uint32_t const head_size = compute_head_size<Dir>(d_array, size, rotate_distance);
      int num_sms;
      CUB_ROTATE_CHECK(get_num_sms(stream, num_sms));

      rotate_long::setup_kernel<<<num_sms, 512, 0, stream>>>(d_temp_storage, num_tiles);
      CUB_ROTATE_CHECK(cudaGetLastError());
      CUB_ROTATE_CHECK(cudaMemcpyAsync(
        reinterpret_cast<uint32_t*>(d_temp_storage) + 1,
        state.ordering_.data(),
        num_tiles * sizeof(uint32_t),
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
        (void*) &rotate_distance,
        const_cast<void*>(static_cast<const void*>(&num_tiles)),
        const_cast<void*>(static_cast<const void*>(&head_size))};
      CUB_ROTATE_CHECK(cudaLaunchCooperativeKernel(
        (void*) rotate_long::rotate_long_kernel<Dir, T>, grid_dim, block_dim, kernelArgs, 0, stream));
    }
    else
    {
      T* d_save_buf = reinterpret_cast<T*>(d_temp_storage);
      cuda::std::identity id{};
      ::cuda::stream_ref env{stream};

      // Save the wrapped elements
      auto* src = Dir == RotDir::Left ? d_array : d_array + size - rotate_distance;
      CUB_ROTATE_CHECK(cub::DeviceTransform::__transform_internal(
        ::cuda::std::make_tuple(src), d_save_buf, rotate_distance, ::cuda::always_true{}, id, env));

      // Shift the remaining region in chunks
      size_t remaining = size - rotate_distance;
      while (remaining > 0)
      {
        size_t const elems_to_copy = std::min(rotate_distance, remaining);
        size_t const src_ix        = Dir == RotDir::Left ? size - remaining : remaining - elems_to_copy;
        size_t const dst_ix        = Dir == RotDir::Left ? src_ix - rotate_distance : src_ix + rotate_distance;
        CUB_ROTATE_CHECK(cub::DeviceTransform::__transform_internal(
          ::cuda::std::make_tuple(d_array + src_ix), d_array + dst_ix, elems_to_copy, ::cuda::always_true{}, id, env));
        remaining -= elems_to_copy;
      }

      // Restore the saved elements
      auto* dst = Dir == RotDir::Left ? d_array + size - rotate_distance : d_array;
      CUB_ROTATE_CHECK(cub::DeviceTransform::__transform_internal(
        ::cuda::std::make_tuple(d_save_buf), dst, rotate_distance, ::cuda::always_true{}, id, env));
    }

    return cudaSuccess;
  } // else (aligned type)
}

#undef CUB_ROTATE_CHECK
} // namespace rotate
} // namespace detail
CUB_NAMESPACE_END
