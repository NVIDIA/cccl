// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/device_vector.h>

#include <cuda/__device/all_devices.h>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/ranges>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda/experimental/__multi_gpu/transform.h>
#include <cuda/experimental/stream.cuh>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <nccl.h>
#include <nvbench_helper.cuh>

namespace cudax = cuda::experimental;

namespace
{
using T = double;

constexpr T start_a = 0.1;
constexpr T start_b = 0.2;
constexpr T start_c = 0.3;
constexpr T scalar  = 3.0;

// 2^25 elements keeps each double-precision array well above the 1M-element STREAM minimum.
// 2^31 elements is useful for large-memory GPUs; nvbench skips cases that fail allocation.
constexpr cuda::std::array<cuda::std::int64_t, 2> array_size_powers{25, 31};
constexpr cuda::std::array<cuda::std::int64_t, 3> local_device_counts{1, 2, 4};
constexpr cuda::std::array<cuda::std::int64_t, 1> contexts_per_device_counts{1};

void check_cuda(cudaError_t status, const char* msg)
{
  if (status != cudaSuccess)
  {
    throw std::runtime_error(std::string{msg} + ": " + cudaGetErrorString(status));
  }
}

#define CHECK_NCCL(expr, __VA_ARGS__)                          \
  do                                                           \
  {                                                            \
    if (const ncclResult_t __ret = expr; __ret != ncclSuccess) \
    {                                                          \
      std::stringstream ss;                                    \
      ss << #expr ": " << ncclGetErrorString(__ret);           \
      /* NOLINTNEXTLINE(bugprone-macro-parentheses) */         \
      ss << " " << __VA_ARGS__;                                \
      throw std::runtime_error{ss.str()};                      \
    }                                                          \
  } while (0)

[[nodiscard]] bool
validate_topology(cuda::std::size_t local_devices, cuda::std::size_t contexts_per_device, nvbench::state& state)
{
  if (contexts_per_device != 1)
  {
    state.skip("Skipping: green contexts per device are not implemented yet.");
    return false;
  }

  if (local_devices == 0)
  {
    throw std::invalid_argument{"LocalDevices must be positive."};
  }

  if (local_devices > cuda::devices.size())
  {
    throw std::invalid_argument{"LocalDevices exceeds the number of visible CUDA devices."};
  }

  return true;
}

class local_nccl_context
{
public:
  explicit local_nccl_context(const std::vector<cudax::logical_device>& devices)
      : shared_mem_{new cuda::std::byte[cudax::nccl_thread_group::required_shared_memory_size(devices.size())]}
  {
    const auto group_size = static_cast<int>(devices.size());
    cudax::nccl_thread_group::initialize_shared_memory(shared_mem_.get());

    comms_.resize(devices.size());

    ncclUniqueId id{};
    CHECK_NCCL(ncclGetUniqueId(&id), "Error in ncclGetUniqueId");

    CHECK_NCCL(ncclGroupStart(), "Error in ncclGroupStart");
    try
    {
      for (cuda::std::size_t rank = 0; rank < devices.size(); ++rank)
      {
        check_cuda(cudaSetDevice(devices[rank].underlying_device().get()), "Failed to set current CUDA device");
        CHECK_NCCL(ncclCommInitRank(&comms_[rank], group_size, id, static_cast<int>(rank)),
                   "Error in ncclCommInitRank");
      }
      CHECK_NCCL(ncclGroupEnd(), "Error in ncclGroupEnd");
    }
    catch (...)
    {
      static_cast<void>(ncclGroupEnd());
      cleanup();
      throw;
    }

    groups_.reserve(devices.size());
    for (cuda::std::size_t rank = 0; rank < devices.size(); ++rank)
    {
      groups_.emplace_back(comms_[rank], devices[rank], shared_mem_);
    }
  }

  local_nccl_context(const local_nccl_context&)            = delete;
  local_nccl_context& operator=(const local_nccl_context&) = delete;

  ~local_nccl_context()
  {
    groups_.clear();
    cleanup();
  }

  [[nodiscard]] const std::vector<cudax::nccl_thread_group>& groups() const
  {
    return groups_;
  }

private:
  void cleanup() noexcept
  {
    for (auto& comm : comms_)
    {
      if (comm != nullptr)
      {
        static_cast<void>(ncclCommDestroy(comm));
        comm = nullptr;
      }
    }
  }

  std::shared_ptr<cuda::std::byte[]> shared_mem_{};
  std::vector<ncclComm_t> comms_;
  std::vector<cudax::nccl_thread_group> groups_;
};

class stream_workload
{
public:
  stream_workload(cuda::std::size_t elements, cuda::std::uint64_t local_devices)
      : devices_{[&] {
        std::vector<cudax::logical_device> ret;

        ret.reserve(local_devices);
        for (cuda::std::size_t i = 0; i < local_devices; ++i)
        {
          ret.emplace_back(cuda::devices[i]);
        }
        return ret;
      }()}
      , nccl_{devices_}
      , sizes_{split_elements_(elements, devices_.size())}
      , streams_{devices_.begin(), devices_.end()}
  {}

  [[nodiscard]] std::vector<thrust::device_vector<T>> allocate_vectors(T value) const
  {
    std::vector<thrust::device_vector<T>> ret;

    ret.reserve(devices_.size());
    for (cuda::std::size_t i = 0; i < devices_.size(); ++i)
    {
      check_cuda(cudaSetDevice(devices_[i].underlying_device().get()), "Failed to set current CUDA device");
      ret.emplace_back(sizes_[i], value);
    }

    return ret;
  }

  template <class InputRange, class OutputRange, class Op>
  void launch(nvbench::state& state, InputRange&& inputs, OutputRange&& outputs, Op op)
  {
    state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync, [&](nvbench::launch&) {
      cudax::transform(nccl_.groups(), streams_, inputs, outputs, op);

      sync_();
    });
  }

private:
  [[nodiscard]] static std::vector<cuda::std::size_t>
  split_elements_(cuda::std::size_t elements, cuda::std::size_t parts)
  {
    const auto base      = elements / parts;
    const auto remainder = elements % parts;

    std::vector<cuda::std::size_t> sizes;

    sizes.reserve(parts);
    for (cuda::std::size_t i = 0; i < parts; ++i)
    {
      sizes.push_back(base + (i < remainder ? 1 : 0));
    }

    return sizes;
  }

  void sync_() const
  {
    for (const auto& stream : streams_)
    {
      stream.sync();
    }
  }

  std::vector<cudax::logical_device> devices_;
  local_nccl_context nccl_;
  std::vector<cuda::std::size_t> sizes_;
  std::vector<cudax::stream> streams_;
};

template <class Benchmark>
void run_stream_benchmark(nvbench::state& state, cuda::std::size_t reads_per_element, Benchmark&& benchmark)
{
  const auto elements            = state.get_int64("Elements");
  const auto num_local_devices   = state.get_int64("LocalDevices");
  const auto contexts_per_device = state.get_int64("ContextsPerDevice");

  if (!validate_topology(num_local_devices, contexts_per_device, state))
  {
    return;
  }

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(reads_per_element * elements);
  state.add_global_memory_writes<T>(elements);

  stream_workload workload{elements, num_local_devices};

  try
  {
    benchmark(workload);
  }
  catch (const std::bad_alloc&)
  {
    state.skip("Skipping: out of memory.");
  }
}
} // namespace

namespace
{
void copy(nvbench::state& state)
{
  run_stream_benchmark(state, /*reads_per_element=*/1, [&state](stream_workload& workload) {
    auto inputs  = workload.allocate_vectors(start_a);
    auto outputs = workload.allocate_vectors(start_c);

    workload.launch(state, inputs, outputs, [] _CCCL_DEVICE(T ai) {
      return ai;
    });
  });
}
} // namespace

NVBENCH_BENCH(copy)
  .set_name("copy")
  .add_int64_power_of_two_axis("Elements", array_size_powers)
  .add_int64_axis("LocalDevices", local_device_counts)
  .add_int64_axis("ContextsPerDevice", contexts_per_device_counts);

namespace
{
void scale(nvbench::state& state)
{
  run_stream_benchmark(state, /*reads_per_element=*/1, [&state](stream_workload& workload) {
    auto inputs  = workload.allocate_vectors(start_b);
    auto outputs = workload.allocate_vectors(start_c);

    workload.launch(state, inputs, outputs, [] _CCCL_DEVICE(T ci) {
      return scalar * ci;
    });
  });
}
} // namespace

NVBENCH_BENCH(scale)
  .set_name("scale")
  .add_int64_power_of_two_axis("Elements", array_size_powers)
  .add_int64_axis("LocalDevices", local_device_counts)
  .add_int64_axis("ContextsPerDevice", contexts_per_device_counts);

namespace
{
void add(nvbench::state& state)
{
  run_stream_benchmark(state, /*reads_per_element=*/2, [&state](stream_workload& workload) {
    auto a      = workload.allocate_vectors(start_a);
    auto b      = workload.allocate_vectors(start_b);
    auto inputs = cuda::std::views::zip(a, b) | cuda::std::views::transform([](const auto& tuple) {
                    return cuda::std::views::zip(cuda::std::get<0>(tuple), cuda::std::get<1>(tuple));
                  });

    auto outputs = workload.allocate_vectors(start_c);

    workload.launch(state, inputs, outputs, [] _CCCL_DEVICE(const auto& values) {
      return cuda::std::get<0>(values) + cuda::std::get<1>(values);
    });
  });
}
} // namespace

NVBENCH_BENCH(add)
  .set_name("add")
  .add_int64_power_of_two_axis("Elements", array_size_powers)
  .add_int64_axis("LocalDevices", local_device_counts)
  .add_int64_axis("ContextsPerDevice", contexts_per_device_counts);

namespace
{
void triad(nvbench::state& state)
{
  run_stream_benchmark(state, /*reads_per_element=*/2, [&state](stream_workload& workload) {
    auto a      = workload.allocate_vectors(start_a);
    auto b      = workload.allocate_vectors(start_b);
    auto inputs = cuda::std::views::zip(a, b) | cuda::std::views::transform([](const auto& tuple) {
                    return cuda::std::views::zip(cuda::std::get<0>(tuple), cuda::std::get<1>(tuple));
                  });

    auto outputs = workload.allocate_vectors(start_c);

    workload.launch(state, inputs, outputs, [] _CCCL_DEVICE(const auto& values) {
      return cuda::std::get<0>(values) + (scalar * cuda::std::get<1>(values));
    });
  });
}
} // namespace

NVBENCH_BENCH(triad)
  .set_name("triad")
  .add_int64_power_of_two_axis("Elements", array_size_powers)
  .add_int64_axis("LocalDevices", local_device_counts)
  .add_int64_axis("ContextsPerDevice", contexts_per_device_counts);
