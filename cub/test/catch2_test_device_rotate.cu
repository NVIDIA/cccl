// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_rotate.cuh>

#include <thrust/equal.h>
#include <thrust/execution_policy.h>

#include <cuda/cmath>

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <curand.h>

#include <c2h/catch2_test_helper.h>

using namespace cub::detail::rotate;

// ============================================================================
// Device-side test harness
// ============================================================================

template <typename T>
class RotateTestHarness
{
public:
  static constexpr unsigned long long kNoMismatch = ULLONG_MAX;

  RotateTestHarness()
  {
    cudaStreamCreate(&stream_);
    curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen_, 12345ull);
    curandSetStream(gen_, stream_);
  }

  ~RotateTestHarness()
  {
    if (d_reference_)
    {
      cudaFreeAsync(d_reference_, stream_);
    }
    if (d_workspace_)
    {
      cudaFreeAsync(d_workspace_, stream_);
    }
    if (d_tmp_storage_)
    {
      cudaFreeAsync(d_tmp_storage_, stream_);
    }
    curandDestroyGenerator(gen_);
    cudaStreamDestroy(stream_);
  }

  RotateTestHarness(RotateTestHarness const&)            = delete;
  RotateTestHarness& operator=(RotateTestHarness const&) = delete;

  void prepare(size_t size, size_t rot_dist, int unaligned_elems)
  {
    if (rot_dist >= size)
    {
      rot_dist %= size;
    }
    size_            = size;
    rot_dist_        = rot_dist;
    unaligned_elems_ = unaligned_elems;

    constexpr size_t kAlignT =
      sizeof(T) >= sizeof(uint32_t)
        ? 1
        : (sizeof(uint32_t) % sizeof(T) == 0 ? sizeof(uint32_t) / sizeof(T) : sizeof(uint32_t));
    size_t const buf_count = cuda::round_up(size + BYTES_PER_SECTOR, kAlignT);
    realloc_async(reinterpret_cast<void**>(&d_reference_), buf_count * sizeof(T));
    realloc_async(reinterpret_cast<void**>(&d_workspace_), buf_count * sizeof(T));

    // Query pass: determine temp storage size and fill state
    tmp_storage_bytes_ = 0;
    cub::DeviceRotate::Rotate(nullptr, tmp_storage_bytes_, state_, workspace_ptr(), size, rot_dist, stream_);
    realloc_async(reinterpret_cast<void**>(&d_tmp_storage_), tmp_storage_bytes_);

    size_t const count_u32 = buf_count * sizeof(T) / sizeof(uint32_t);
    curandGenerate(gen_, reinterpret_cast<uint32_t*>(d_reference_), count_u32);
  }

  T* reference_ptr() const
  {
    return d_reference_ + unaligned_elems_;
  }
  T* workspace_ptr() const
  {
    return d_workspace_ + unaligned_elems_;
  }

  void reset_workspace()
  {
    cudaMemcpyAsync(workspace_ptr(), reference_ptr(), size_ * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
  }

  void rotate(cudaStream_t stream)
  {
    cub::DeviceRotate::Rotate(d_tmp_storage_, tmp_storage_bytes_, state_, workspace_ptr(), size_, rot_dist_, stream);
  }

  unsigned long long verify()
  {
    auto policy        = thrust::cuda::par.on(stream_);
    T* ws              = workspace_ptr();
    T* ref             = reference_ptr();
    size_t const tail  = size_ - rot_dist_;
    bool const eq_head = thrust::equal(policy, ws, ws + tail, ref + rot_dist_);
    bool const eq_tail = thrust::equal(policy, ws + tail, ws + size_, ref);
    if (eq_head && eq_tail)
    {
      return kNoMismatch;
    }
    else
    {
      std::vector<T> h_ws(size_), h_ref(size_);
      cudaMemcpyAsync(h_ws.data(), ws, size_ * sizeof(T), cudaMemcpyDeviceToHost, stream_);
      cudaMemcpyAsync(h_ref.data(), ref, size_ * sizeof(T), cudaMemcpyDeviceToHost, stream_);
      cudaStreamSynchronize(stream_);
      for (size_t i = 0; i < size_; ++i)
      {
        size_t const expected = (i + rot_dist_) % size_;
        if (h_ws[i] != h_ref[expected])
        {
          return i;
        }
      }
      throw std::runtime_error("Thrust found a mismatch, but manual verification did not!");
    }
  }

  unsigned long long run_and_verify()
  {
    reset_workspace();
    rotate(stream_);
    return verify();
  }

private:
  void realloc_async(void** ptr, size_t bytes)
  {
    if (*ptr)
    {
      cudaFreeAsync(*ptr, stream_);
      cudaStreamSynchronize(stream_);
      cudaMemPool_t pool;
      cudaDeviceGetDefaultMemPool(&pool, 0);
      cudaMemPoolTrimTo(pool, 0);
    }
    cudaMallocAsync(ptr, bytes, stream_);
  }

  cudaStream_t stream_;
  curandGenerator_t gen_;
  T* d_reference_           = nullptr;
  T* d_workspace_           = nullptr;
  void* d_tmp_storage_      = nullptr;
  int unaligned_elems_      = 0;
  size_t size_              = 0;
  size_t rot_dist_          = 0;
  size_t tmp_storage_bytes_ = 0;
  cub::RotateState_t state_;
};

// ============================================================================
// Helpers
// ============================================================================

struct EdgeCase
{
  size_t arr_size;
  size_t rot_dist;
  int unaligned_elems;
};

template <typename T>
void check_edge_case(RotateTestHarness<T>& harness, EdgeCase const& tc)
{
  harness.prepare(tc.arr_size, tc.rot_dist, tc.unaligned_elems);
  auto const mismatch = harness.run_and_verify();
  REQUIRE(mismatch == harness.kNoMismatch);
}

template <typename T>
void run_random_tests(
  RotateTestHarness<T>& harness, size_t min_size, size_t max_size, size_t min_rot, size_t max_rot, size_t num_tests)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> size_dist(min_size, max_size);
  std::uniform_int_distribution<size_t> rot_dist(min_rot, max_rot);
  std::uniform_int_distribution<int> unaligned_dist(0, static_cast<int>(BYTES_PER_SECTOR / sizeof(T)) - 1);

  for (size_t i = 0; i < num_tests; ++i)
  {
    harness.prepare(size_dist(gen), rot_dist(gen), unaligned_dist(gen));
    auto const mismatch = harness.run_and_verify();
    REQUIRE(mismatch == harness.kNoMismatch);
  }
}

// ============================================================================
// Edge case tests
// ============================================================================

using rotate_types = c2h::type_list<uint8_t, uint16_t, uint32_t, uint64_t, uchar3>;

C2H_TEST("DeviceRotate trivial and degenerate cases", "[device][rotate]", rotate_types)
{
  using T            = c2h::get<0, TestType>;
  constexpr size_t S = BYTES_PER_SECTOR / sizeof(T);

  RotateTestHarness<T> harness;

  std::vector<EdgeCase> cases = {
    {2, 1, 0},
    {2, 1, 1},
    {3, 1, 0},
    {3, 2, 0},
    {S, 1, 0},
    {S, S - 1, 0},
    {S + 1, 1, 0},
    {S + 1, S, 0},
  };

  for (auto const& tc : cases)
  {
    check_edge_case(harness, tc);
  }
}

C2H_TEST("DeviceRotate no-op and modular reduction", "[device][rotate]", rotate_types)
{
  using T = c2h::get<0, TestType>;

  RotateTestHarness<T> harness;

  std::vector<EdgeCase> cases = {
    {100, 0, 0},
    {100, 100, 0},
    {100, 250, 0},
    {100, 200, 0},
    {100, 300, 0},
    {100, 101, 0},
    {100, 99, 0},
  };

  for (auto const& tc : cases)
  {
    check_edge_case(harness, tc);
  }
}

C2H_TEST("DeviceRotate short algorithm boundary", "[device][rotate]", rotate_types)
{
  using T             = c2h::get<0, TestType>;
  constexpr size_t TS = rotate_short::TILE_BYTES / sizeof(T);

  RotateTestHarness<T> harness;

  std::vector<EdgeCase> cases = {
    {TS + 1, TS, 0},
    {TS + 1, TS, 1},
    {TS + 1, TS, 31},
    {2 * TS, TS, 0},
    {2 * TS, TS - 1, 0},
    {2 * TS, TS + 1, 0},
    {2 * TS, TS + 1, 1},
    {2 * TS, TS + 1, 16},
    {2 * TS, TS + 1, 31},
  };

  for (auto const& tc : cases)
  {
    check_edge_case(harness, tc);
  }
}

C2H_TEST("DeviceRotate short rotate_tiny path", "[device][rotate]", rotate_types)
{
  using T             = c2h::get<0, TestType>;
  constexpr size_t TS = rotate_short::TILE_BYTES / sizeof(T);

  RotateTestHarness<T> harness;

  std::vector<EdgeCase> cases = {
    {TS, 1, 0},
    {TS, TS - 1, 0},
    {TS, TS / 2, 0},
    {2 * TS - 1, TS, 0},
    {2 * TS, TS, 0},
    {2 * TS + 1, TS, 0},
    {TS + 1, TS / 2 + 1, 0},
    {TS + 1, TS / 2, 0},
  };

  for (auto const& tc : cases)
  {
    check_edge_case(harness, tc);
  }
}

C2H_TEST("DeviceRotate short head_size alignments", "[device][rotate]", rotate_types)
{
  using T             = c2h::get<0, TestType>;
  constexpr size_t TS = rotate_short::TILE_BYTES / sizeof(T);

  RotateTestHarness<T> harness;

  std::vector<EdgeCase> cases = {
    {2 * TS, 100, 0},
    {2 * TS, 100, 1},
    {2 * TS, 100, 8},
    {2 * TS, 100, 15},
    {2 * TS, 100, 16},
    {2 * TS, 100, 24},
    {2 * TS, 100, 31},
  };

  for (auto const& tc : cases)
  {
    check_edge_case(harness, tc);
  }
}

C2H_TEST("DeviceRotate short overcopy and tile distribution", "[device][rotate]", rotate_types)
{
  using T             = c2h::get<0, TestType>;
  constexpr size_t S  = BYTES_PER_SECTOR / sizeof(T);
  constexpr size_t TS = rotate_short::TILE_BYTES / sizeof(T);

  RotateTestHarness<T> harness;

  std::vector<EdgeCase> cases = {
    // overcopy alignment
    {2 * TS, S, 0},
    {2 * TS, S + 1, 0},
    {2 * TS, 2 * S - 1, 0},
    {2 * TS, S / 2, 0},
    // get_bytes_to_load remainder
    {TS + 1, 1, 0},
    {2 * TS + 1, 1, 0},
    {3 * TS + 1, 1, 0},
    {2 * TS + 100, 100, 0},
    {TS + 32, 1, 1},
    {2 * TS + 32, 1, 1},
    // tile distribution across SMs
    {TS + S + 1, 1, 0},
    {2 * TS + S + 1, 1, 0},
    // last tile remainder sizes
    {TS + S + 2, 1, 0},
    {TS + S + S, 1, 0},
    {2 * TS, 1, 0},
    {2 * TS + 1, 1, 0},
  };

  for (auto const& tc : cases)
  {
    check_edge_case(harness, tc);
  }
}

C2H_TEST("DeviceRotate long algorithm cases", "[device][rotate]", rotate_types)
{
  using T             = c2h::get<0, TestType>;
  constexpr size_t TS = rotate_short::TILE_BYTES / sizeof(T);

  RotateTestHarness<T> harness;

  std::vector<EdgeCase> cases = {
    // minimum cases
    {2 * TS + 1, TS + 1, 0},
    {2 * TS + 1, TS + 1, 1},
    {2 * TS + 1, TS + 1, 16},
    {2 * TS + 1, TS + 1, 31},
    {3 * TS, TS + 1, 0},
    {3 * TS, 2 * TS - 1, 0},
    // exactly 1 positive tile
    {2 * TS + 1, TS + 1, 0},
    {2 * TS + 2, TS + 1, 0},
    // tile remainder cases
    {3 * TS + 1, TS + 1, 0},
    {3 * TS + 2, TS + 1, 0},
    {3 * TS, 2 * TS + 1, 0},
    {3 * TS, 2 * TS, 0},
    // neg_head_size variations
    {10 * TS, 5 * TS + 1, 0},
    {10 * TS, 5 * TS + 16, 0},
    {10 * TS + 1, 5 * TS + 1, 0},
    {10 * TS + 17, 5 * TS + 1, 1},
    // exact multiples of TILE_SIZE
    {2 * TS, TS + 1, 0},
    {3 * TS, TS, 0},
    {3 * TS, 2 * TS, 0},
    {4 * TS, 2 * TS + 1, 0},
    {4 * TS, 2 * TS + 1, 1},
  };

  for (auto const& tc : cases)
  {
    check_edge_case(harness, tc);
  }
}

C2H_TEST("DeviceRotate long all alignments at medium scale", "[device][rotate]", rotate_types)
{
  using T = c2h::get<0, TestType>;

  RotateTestHarness<T> harness;

  std::vector<EdgeCase> cases = {
    {100'000, 50'000, 0},
    {100'000, 50'000, 1},
    {100'000, 50'000, 8},
    {100'000, 50'000, 15},
    {100'000, 50'000, 16},
    {100'000, 50'000, 24},
    {100'000, 50'000, 31},
  };

  for (auto const& tc : cases)
  {
    check_edge_case(harness, tc);
  }
}

C2H_TEST("DeviceRotate head consumes positive region fallback", "[device][rotate]", rotate_types)
{
  using T = c2h::get<0, TestType>;

  RotateTestHarness<T> harness;

  std::vector<EdgeCase> cases = {
    {100'000, 99'999, 1},
    {100'000, 99'999, 15},
    {100'000, 99'999, 31},
    {100'000, 99'970, 1},
    {100'000, 99'969, 1},
    {100'000, 99'968, 1},
    {100'000, 99'999, 0},
  };

  for (auto const& tc : cases)
  {
    check_edge_case(harness, tc);
  }
}

C2H_TEST("DeviceRotate maximal and near-boundary rotations", "[device][rotate]", rotate_types)
{
  using T             = c2h::get<0, TestType>;
  constexpr size_t S  = BYTES_PER_SECTOR / sizeof(T);
  constexpr size_t TS = rotate_short::TILE_BYTES / sizeof(T);

  RotateTestHarness<T> harness;

  std::vector<EdgeCase> cases = {
    {1'000, 999, 0},
    {1'000, 999, 1},
    {10 * TS, 10 * TS - 1, 0},
    {10 * TS, 10 * TS - 1, 1},
    {100'000, 60'000, 0},
    {100'000, 60'000, 1},
    {100'000, 60'000, 31},
    {200'000, 200'000 - 1, 1},
    {200'000, 200'000 - S, 1},
    {200'000, 200'000 - S + 1, 1},
    {200'000, 200'000 - S - 1, 1},
    // naive algorithm path
    {100'000, 99'000, 0},
    {100'000, 99'000, 1},
    // sizes near BYTES_PER_SECTOR
    {S - 1, 1, 0},
    {S - 1, S - 2, 0},
    {S, S / 2, 0},
    {S + 1, S / 2, 0},
    {2 * S, S, 0},
    {2 * S + 1, S, 0},
    {2 * S, 1, 0},
  };

  for (auto const& tc : cases)
  {
    check_edge_case(harness, tc);
  }
}

C2H_TEST("DeviceRotate large-scale tests", "[device][rotate]", rotate_types)
{
  using T             = c2h::get<0, TestType>;
  constexpr size_t TS = rotate_short::TILE_BYTES / sizeof(T);

  RotateTestHarness<T> harness;

  std::vector<EdgeCase> cases = {
    {1'000'000, TS + 1, 0},
    {1'000'000, TS + 1, 1},
    {1'000'000, 500'000, 0},
    {1'000'000, 500'000, 1},
    {1'000'000, 999'999, 0},
    {1'000'000, 999'999, 1},
  };

  for (auto const& tc : cases)
  {
    check_edge_case(harness, tc);
  }
}

// ============================================================================
// size == 0 and size == 1 (bypass harness which has its own % size)
// ============================================================================

C2H_TEST("DeviceRotate size zero and one", "[device][rotate]", rotate_types)
{
  using T = c2h::get<0, TestType>;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  size_t temp_bytes = 42;
  cub::RotateState_t state{};

  // size == 0, rotate_distance == 0
  REQUIRE(cub::DeviceRotate::Rotate(nullptr, temp_bytes, state, static_cast<T*>(nullptr), 0, 0, stream) == cudaSuccess);
  REQUIRE(temp_bytes == 0);

  // size == 0, rotate_distance > 0 (was UB before fix)
  temp_bytes = 42;
  REQUIRE(cub::DeviceRotate::Rotate(nullptr, temp_bytes, state, static_cast<T*>(nullptr), 0, 5, stream) == cudaSuccess);
  REQUIRE(temp_bytes == 0);

  // size == 0, rotate_distance very large
  temp_bytes = 42;
  REQUIRE(
    cub::DeviceRotate::Rotate(nullptr, temp_bytes, state, static_cast<T*>(nullptr), 0, 999, stream) == cudaSuccess);
  REQUIRE(temp_bytes == 0);

  // size == 1 — rotation of a single element is always a no-op
  T* d_buf = nullptr;
  cudaMallocAsync(&d_buf, sizeof(T), stream);
  T val = {};
  cudaMemcpyAsync(d_buf, &val, sizeof(T), cudaMemcpyHostToDevice, stream);

  temp_bytes = 42;
  REQUIRE(cub::DeviceRotate::Rotate(nullptr, temp_bytes, state, d_buf, 1, 0, stream) == cudaSuccess);
  REQUIRE(temp_bytes == 0);

  temp_bytes = 42;
  REQUIRE(cub::DeviceRotate::Rotate(nullptr, temp_bytes, state, d_buf, 1, 1, stream) == cudaSuccess);
  REQUIRE(temp_bytes == 0);

  temp_bytes = 42;
  REQUIRE(cub::DeviceRotate::Rotate(nullptr, temp_bytes, state, d_buf, 1, 100, stream) == cudaSuccess);
  REQUIRE(temp_bytes == 0);

  cudaFreeAsync(d_buf, stream);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
}

// ============================================================================
// Long algorithm: 1-element positive region (no naive fallback)
// ============================================================================

C2H_TEST("DeviceRotate long single-element positive region", "[device][rotate]", rotate_types)
{
  using T             = c2h::get<0, TestType>;
  constexpr size_t S  = BYTES_PER_SECTOR / sizeof(T);
  constexpr size_t TS = rotate_short::TILE_BYTES / sizeof(T);

  RotateTestHarness<T> harness;

  std::vector<EdgeCase> cases = {
    // pos region = 1 element, head_size = 0 (aligned) — 1 pos tile with 1 elem
    {TS + 2, TS + 1, 0},
    // pos region = 2 elements
    {TS + 3, TS + 1, 0},
    // pos region = S elements (one sector worth)
    {TS + 1 + S, TS + 1, 0},
    // with alignment offsets
    {TS + 2, TS + 1, 1},
    {TS + 3, TS + 1, 1},
  };

  for (auto const& tc : cases)
  {
    check_edge_case(harness, tc);
  }
}

// ============================================================================
// Short: zero overcopy combined with non-zero head_size
// ============================================================================

C2H_TEST("DeviceRotate short zero overcopy with head", "[device][rotate]", rotate_types)
{
  using T             = c2h::get<0, TestType>;
  constexpr size_t S  = BYTES_PER_SECTOR / sizeof(T);
  constexpr size_t TS = rotate_short::TILE_BYTES / sizeof(T);

  RotateTestHarness<T> harness;

  // rot_dist is an exact multiple of ELEMS_PER_SECTOR → overcopy = 0
  // combined with various non-zero head_size values
  std::vector<EdgeCase> cases = {
    {2 * TS, S, 1},
    {2 * TS, S, 8},
    {2 * TS, S, 15},
    {2 * TS, S, 16},
    {2 * TS, S, 31},
    {2 * TS, 2 * S, 1},
    {2 * TS, 2 * S, 31},
    {3 * TS, S, 1},
    {3 * TS, S, 31},
  };

  for (auto const& tc : cases)
  {
    check_edge_case(harness, tc);
  }
}

// ============================================================================
// Long: both head_size > 0 and neg_head_size > 0 simultaneously
// ============================================================================

C2H_TEST("DeviceRotate long dual head alignment", "[device][rotate]", rotate_types)
{
  using T             = c2h::get<0, TestType>;
  constexpr size_t S  = BYTES_PER_SECTOR / sizeof(T);
  constexpr size_t TS = rotate_short::TILE_BYTES / sizeof(T);

  RotateTestHarness<T> harness;

  // Sweep alignment offsets that produce non-zero pos head, checking that
  // neg_head is also exercised (depends on array size and rot_dist alignment).
  std::vector<EdgeCase> cases = {
    {5 * TS, 2 * TS + 1, 1},
    {5 * TS, 2 * TS + 1, 3},
    {5 * TS, 2 * TS + 1, 7},
    {5 * TS, 2 * TS + 1, 15},
    {5 * TS, 2 * TS + 1, 31},
    {5 * TS + 1, 2 * TS + 3, 1},
    {5 * TS + 1, 2 * TS + 3, 7},
    {5 * TS + 1, 2 * TS + 3, 15},
    {5 * TS + 1, 2 * TS + 3, 31},
    // odd sizes that stress both heads
    {7 * TS + 13, 3 * TS + 7, 5},
    {7 * TS + 13, 3 * TS + 7, 11},
    {7 * TS + 13, 3 * TS + 7, 29},
  };

  for (auto const& tc : cases)
  {
    check_edge_case(harness, tc);
  }
}

// ============================================================================
// Short: single main tile (num_main_tiles == 1)
// ============================================================================

C2H_TEST("DeviceRotate short single tile with alignments", "[device][rotate]", rotate_types)
{
  using T             = c2h::get<0, TestType>;
  constexpr size_t S  = BYTES_PER_SECTOR / sizeof(T);
  constexpr size_t TS = rotate_short::TILE_BYTES / sizeof(T);

  RotateTestHarness<T> harness;

  std::vector<EdgeCase> cases = {
    // minimal: positive region = 1 element beyond head → 1 tile
    {TS + S + 1, S, 0},
    {TS + S + 2, S, 0},
    {TS + S + S, S, 0},
    // with head_size via alignment offset
    {TS + S + 1, S, 1},
    {TS + S + 1, S, 15},
    {TS + S + 1, S, 31},
    // rot_dist = 1, various sizes producing exactly 1 tile
    {TS + 1, 1, 0},
    {TS + 1, 1, 1},
    {TS + 1, 1, 15},
    {TS + 1, 1, 31},
  };

  for (auto const& tc : cases)
  {
    check_edge_case(harness, tc);
  }
}

// ============================================================================
// Naive algorithm: explicit coverage of cases that produce high dep distance
// ============================================================================

C2H_TEST("DeviceRotate naive algorithm fallback", "[device][rotate]", rotate_types)
{
  using T             = c2h::get<0, TestType>;
  constexpr size_t TS = rotate_short::TILE_BYTES / sizeof(T);

  RotateTestHarness<T> harness;

  // rot_dist close to size/2 or large relative to size → high dep distance
  std::vector<EdgeCase> cases = {
    {100'000, 99'000, 0},
    {100'000, 99'000, 1},
    {100'000, 99'000, 15},
    {100'000, 99'000, 31},
    {50'000, 49'000, 0},
    {50'000, 49'000, 1},
    // rot_dist near size — always naive via head-consumes-positive fallback
    {200'000, 199'999, 1},
    {200'000, 199'999, 15},
    {200'000, 199'990, 1},
  };

  for (auto const& tc : cases)
  {
    check_edge_case(harness, tc);
  }
}

// ============================================================================
// Short: rot_dist = 1 with large arrays and non-zero alignment
// ============================================================================

C2H_TEST("DeviceRotate short rot_dist one large array", "[device][rotate]", rotate_types)
{
  using T             = c2h::get<0, TestType>;
  constexpr size_t TS = rotate_short::TILE_BYTES / sizeof(T);

  RotateTestHarness<T> harness;

  std::vector<EdgeCase> cases = {
    {1'000'000, 1, 0},
    {1'000'000, 1, 1},
    {1'000'000, 1, 15},
    {1'000'000, 1, 31},
    {500'000, 1, 0},
    {500'000, 1, 7},
  };

  for (auto const& tc : cases)
  {
    check_edge_case(harness, tc);
  }
}

// ============================================================================
// Long: rot_dist just over TILE_SIZE boundary with all alignment offsets
// ============================================================================

C2H_TEST("DeviceRotate long boundary sweep alignments", "[device][rotate]", rotate_types)
{
  using T             = c2h::get<0, TestType>;
  constexpr size_t S  = BYTES_PER_SECTOR / sizeof(T);
  constexpr size_t TS = rotate_short::TILE_BYTES / sizeof(T);

  RotateTestHarness<T> harness;

  std::vector<EdgeCase> cases = {
    // rot_dist = TS + 1 (minimal long) with every alignment
    {3 * TS, TS + 1, 0},
    {3 * TS, TS + 1, 1},
    {3 * TS, TS + 1, 3},
    {3 * TS, TS + 1, 7},
    {3 * TS, TS + 1, 8},
    {3 * TS, TS + 1, 15},
    {3 * TS, TS + 1, 16},
    {3 * TS, TS + 1, 24},
    {3 * TS, TS + 1, 31},
    // rot_dist = TS + 2
    {3 * TS, TS + 2, 0},
    {3 * TS, TS + 2, 1},
    {3 * TS, TS + 2, 31},
    // rot_dist = TS + S (sector-aligned long distance)
    {4 * TS, TS + S, 0},
    {4 * TS, TS + S, 1},
    {4 * TS, TS + S, 31},
  };

  for (auto const& tc : cases)
  {
    check_edge_case(harness, tc);
  }
}

// ============================================================================
// Long: exact tile-multiple regions (no remainder tiles)
// ============================================================================

C2H_TEST("DeviceRotate long exact tile multiples", "[device][rotate]", rotate_types)
{
  using T             = c2h::get<0, TestType>;
  constexpr size_t TS = rotate_short::TILE_BYTES / sizeof(T);

  RotateTestHarness<T> harness;

  std::vector<EdgeCase> cases = {
    // Both regions exact multiples of TILE_SIZE (no partial tiles)
    {4 * TS, 2 * TS, 0},
    {6 * TS, 2 * TS, 0},
    {6 * TS, 3 * TS, 0},
    {6 * TS, 4 * TS, 0},
    // With alignment
    {4 * TS, 2 * TS, 1},
    {4 * TS, 2 * TS, 31},
    {6 * TS, 3 * TS, 1},
    {6 * TS, 3 * TS, 15},
  };

  for (auto const& tc : cases)
  {
    check_edge_case(harness, tc);
  }
}

// ============================================================================
// Sizes near and beyond UINT32_MAX — stress uint32_t tile counts, BFS
// ordering, and size_t arithmetic throughout the pipeline.
//
// The harness needs ~3× the array size in GPU memory (reference + workspace +
// temp).  We query free memory and skip individual cases that would exceed
// one-third of the available pool.
// ============================================================================

inline size_t get_free_bytes()
{
  size_t free_bytes = 0, total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  return free_bytes;
}

TEST_CASE("DeviceRotate sizes near UINT32_MAX", "[device][rotate]")
{
  using T             = uint8_t;
  constexpr size_t S  = BYTES_PER_SECTOR / sizeof(T);
  constexpr size_t TS = rotate_short::TILE_BYTES / sizeof(T);

  constexpr size_t U32MAX = static_cast<size_t>(std::numeric_limits<uint32_t>::max());

  // Each test case needs ~3 * size * sizeof(T) bytes of GPU memory.
  size_t const max_elems = get_free_bytes() / (3 * sizeof(T));

  RotateTestHarness<T> harness;

  std::vector<EdgeCase> cases = {
    // --- Just below UINT32_MAX ---
    {U32MAX - 1, 1, 0},
    {U32MAX - 1, 1, 1},
    {U32MAX - 1, TS, 0},
    {U32MAX - 1, TS - 1, 0},
    {U32MAX - 1, TS + 1, 0},
    {U32MAX - 1, TS + 1, 1},
    {U32MAX - 1, U32MAX / 2, 0},
    {U32MAX - 1, U32MAX / 2, 1},

    // --- Exactly UINT32_MAX ---
    {U32MAX, 1, 0},
    {U32MAX, 1, 1},
    {U32MAX, TS, 0},
    {U32MAX, TS + 1, 0},
    {U32MAX, TS + 1, 1},
    {U32MAX, U32MAX - 1, 0},
    {U32MAX, U32MAX - 1, 1},

    // --- Just above UINT32_MAX ---
    {U32MAX + 1, 1, 0},
    {U32MAX + 1, 1, 1},
    {U32MAX + 1, TS, 0},
    {U32MAX + 1, TS + 1, 0},
    {U32MAX + 1, TS + 1, 1},
    {U32MAX + 1, U32MAX, 0},
    {U32MAX + 1, U32MAX, 1},

    // --- Modular reduction with size > UINT32_MAX ---
    {U32MAX + 1, U32MAX + 2, 0}, // rot > size → rot %= size = 1
    {U32MAX + 100, U32MAX + 200, 0}, // rot > size → rot %= size = 100

    // --- rot_dist near UINT32_MAX with moderate size ---
    {U32MAX, U32MAX / 2 + 1, 0},
    {U32MAX, U32MAX / 2 + 1, 1},

    // --- Size exactly 2 * UINT32_MAX (if memory allows) ---
    {2 * U32MAX, 1, 0},
    {2 * U32MAX, TS + 1, 0},
    {2 * U32MAX, U32MAX, 0},
  };

  for (auto const& tc : cases)
  {
    if (tc.arr_size > max_elems)
    {
      continue;
    }
    check_edge_case(harness, tc);
  }
}

inline size_t get_max_test_bytes()
{
  size_t free_bytes, total_bytes;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  return std::min(free_bytes / 3, static_cast<size_t>(1) * 1024 * 1024 * 1024);
}

// TODO currently trimmed to take less than 2 minutes in total
C2H_TEST("DeviceRotate random short tests", "[device][rotate]", rotate_types)
{
  using T                = c2h::get<0, TestType>;
  constexpr size_t TS    = rotate_short::TILE_BYTES / sizeof(T);
  size_t const max_elems = get_max_test_bytes() / sizeof(T);

  RotateTestHarness<T> harness;

  for (size_t n = 1; n < max_elems; n *= 10)
  {
    size_t const max_size = std::min(10 * n, max_elems);
    run_random_tests(harness, n, max_size, 1, 1'000, 5);
    run_random_tests(harness, n, max_size, 1'000, TS, 5);
  }
}

C2H_TEST("DeviceRotate random long tests", "[device][rotate]", rotate_types)
{
  using T                = c2h::get<0, TestType>;
  constexpr size_t TS    = rotate_short::TILE_BYTES / sizeof(T);
  size_t const max_elems = get_max_test_bytes() / sizeof(T);

  RotateTestHarness<T> harness;

  for (size_t n = TS; n < max_elems; n *= 10)
  {
    for (size_t k = TS; k < n; k *= 10)
    {
      run_random_tests(harness, n, std::min(10 * n, max_elems), k, 10 * k, 5);
    }
  }
}

