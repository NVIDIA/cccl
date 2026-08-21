// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! The STREAM benchmark, distributed over the locality domains of one device, with every kernel
//! expressed through `cudax::transform`.

#include <cuda/__device/logical_device_ref.h>
#include <cuda/__event/event.h>
#include <cuda/__memory_pool/locality_domain_memory_pool.h>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/std/cstddef>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/ranges>
#include <cuda/std/span>
#include <cuda/stream>

#include <cuda/experimental/__device/logical_device.cuh>
#include <cuda/experimental/__multi_gpu/algorithm/transform/transform.h>
#include <cuda/experimental/__multi_gpu/nccl_communicator.h>

#include <stdexcept>
#include <string>
#include <vector>

#include <nccl.h>

#include <nvbench/nvbench.cuh>

namespace cudax = cuda::experimental;

namespace
{
using T = float;

inline constexpr T start_a = 1;
inline constexpr T start_b = 2;
inline constexpr T start_c = 0;
inline constexpr T scalar  = 3;

inline constexpr int min_elements_pow2 = 26;
inline constexpr int max_elements_pow2 = 30;
inline constexpr int elements_stride   = 1;

[[nodiscard]] cuda::device_ref state_device(nvbench::state& state)
{
  return cuda::devices[state.get_device()->get_id()];
}

[[nodiscard]] std::vector<cudax::nccl_communicator>
make_communicators(cuda::device_ref device, cuda::std::span<const cuda::__logical_device_ref> domains)
{
  std::vector<ncclComm_t> raw_comms(domains.size());
  std::vector<int> devs(domains.size(), device.get());

  if (const auto status = ncclCommInitAll(raw_comms.data(), static_cast<int>(devs.size()), devs.data());
      status != ncclSuccess)
  {
    throw std::runtime_error(std::string{"ncclCommInitAll: "} + ncclGetErrorString(status));
  }

  std::vector<cudax::nccl_communicator> comms;

  comms.reserve(domains.size());
  for (cuda::std::size_t domain = 0; domain < domains.size(); ++domain)
  {
    comms.push_back(cudax::nccl_communicator::from_native_handle(raw_comms[domain], domains[domain]));
  }

  return comms;
}

[[nodiscard]] std::vector<cuda::stream> make_streams(cuda::std::span<const cuda::__logical_device_ref> domains)
{
  return {domains.begin(), domains.end()};
}

[[nodiscard]] std::vector<cuda::buffer<T, cuda::mr::device_accessible>> make_buffers(
  cuda::std::span<const cuda::__logical_device_ref> domains,
  const std::vector<cuda::stream>& streams,
  cuda::std::size_t elements,
  const T& value)
{
  std::vector<cuda::buffer<T, cuda::mr::device_accessible>> bufs;

  bufs.reserve(domains.size());
  for (cuda::std::size_t domain = 0; domain < domains.size(); ++domain)
  {
    bufs.push_back(
      cuda::make_buffer<T>(streams[domain], cuda::__device_default_memory_pool(domains[domain]), elements, value));
  }

  return bufs;
}

[[nodiscard]] std::vector<cuda::event> make_events(cuda::device_ref device, cuda::std::size_t count)
{
  std::vector<cuda::event> events;

  events.reserve(count);
  for (cuda::std::size_t i = 0; i < count; ++i)
  {
    events.emplace_back(device);
  }

  return events;
}

template <typename Buffers, typename SubmitFn>
void run_forked_iteration(
  cuda::stream_ref timing_stream, Buffers& bufs, cuda::event& fork, std::vector<cuda::event>& join, SubmitFn submit)
{
  fork.record(timing_stream);
  for (auto& buf : bufs)
  {
    buf.stream().wait(fork);
  }

  submit();

  // All records before any waits: interleaving them blocks the host between records.
  for (cuda::std::size_t domain = 0; domain < bufs.size(); ++domain)
  {
    join[domain].record(bufs[domain].stream());
  }
  for (auto& event : join)
  {
    timing_stream.wait(event);
  }
}

void copy(nvbench::state& state)
{
  const auto elements = static_cast<cuda::std::size_t>(state.get_int64("Elements"));
  const auto device   = state_device(state);
  const auto domains  = device.__locality_domains();

  if (domains.size() < 2)
  {
    state.skip("the GPU does not expose multiple locality domains");
    return;
  }

  auto comms   = make_communicators(device, domains);
  auto streams = make_streams(domains);
  auto a       = make_buffers(domains, streams, elements, start_a);
  auto c       = make_buffers(domains, streams, elements, start_c);

  cuda::event fork{device};
  auto join = make_events(device, domains.size());

  const auto total = elements * domains.size();
  state.add_element_count(total);
  state.add_global_memory_reads<T>(total);
  state.add_global_memory_writes<T>(total);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    run_forked_iteration(cuda::stream_ref{launch.get_stream().get_stream()}, a, fork, join, [&] {
      cudax::transform(
        cudax::distributed,
        comms,
        c | cuda::std::views::transform([](auto& buf) {
          return cuda::std::execution::env{buf.stream(), buf.memory_resource()};
        }),
        a | cuda::std::views::transform(cuda::std::ranges::begin),
        a | cuda::std::views::transform(cuda::std::ranges::size),
        c | cuda::std::views::transform(cuda::std::ranges::begin),
        cuda::std::identity{});
    });
  });
}

NVBENCH_BENCH(copy).set_name("copy").add_int64_power_of_two_axis(
  "Elements", nvbench::range(min_elements_pow2, max_elements_pow2, elements_stride));

void mul(nvbench::state& state)
{
  const auto elements = static_cast<cuda::std::size_t>(state.get_int64("Elements"));
  const auto device   = state_device(state);
  const auto domains  = device.__locality_domains();

  if (domains.size() < 2)
  {
    state.skip("the GPU does not expose multiple locality domains");
    return;
  }

  auto comms   = make_communicators(device, domains);
  auto streams = make_streams(domains);
  auto b       = make_buffers(domains, streams, elements, start_b);
  auto c       = make_buffers(domains, streams, elements, start_c);

  cuda::event fork{device};
  auto join = make_events(device, domains.size());

  const auto total = elements * domains.size();
  state.add_element_count(total);
  state.add_global_memory_reads<T>(total);
  state.add_global_memory_writes<T>(total);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    run_forked_iteration(cuda::stream_ref{launch.get_stream().get_stream()}, b, fork, join, [&] {
      cudax::transform(
        cudax::distributed,
        comms,
        b | cuda::std::views::transform([](auto& buf) {
          return cuda::std::execution::env{buf.stream(), buf.memory_resource()};
        }),
        c | cuda::std::views::transform(cuda::std::ranges::begin),
        c | cuda::std::views::transform(cuda::std::ranges::size),
        b | cuda::std::views::transform(cuda::std::ranges::begin),
        [] __device__(const T& ci) {
          return scalar * ci;
        });
    });
  });
}
NVBENCH_BENCH(mul).set_name("mul").add_int64_power_of_two_axis(
  "Elements", nvbench::range(min_elements_pow2, max_elements_pow2, elements_stride));

void add(nvbench::state& state)
{
  const auto elements = static_cast<cuda::std::size_t>(state.get_int64("Elements"));
  const auto device   = state_device(state);
  const auto domains  = device.__locality_domains();

  if (domains.size() < 2)
  {
    state.skip("the GPU does not expose multiple locality domains");
    return;
  }

  auto comms   = make_communicators(device, domains);
  auto streams = make_streams(domains);
  auto a       = make_buffers(domains, streams, elements, start_a);
  auto b       = make_buffers(domains, streams, elements, start_b);
  auto c       = make_buffers(domains, streams, elements, start_c);

  cuda::event fork{device};
  auto join = make_events(device, domains.size());

  const auto total = elements * domains.size();
  state.add_element_count(total);
  state.add_global_memory_reads<T>(2 * total);
  state.add_global_memory_writes<T>(total);

  // `transform` takes one input iterator, so each rank's two inputs are zipped into one range.
  // The zipped range iterates as a tuple of both elements.
  auto ab = cuda::std::views::zip(a, b) | cuda::std::views::transform([](auto&& in) {
              return cuda::std::views::zip(cuda::std::get<0>(in), cuda::std::get<1>(in));
            });

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    run_forked_iteration(cuda::stream_ref{launch.get_stream().get_stream()}, c, fork, join, [&] {
      cudax::transform(
        cudax::distributed,
        comms,
        c | cuda::std::views::transform([](auto& buf) {
          return cuda::std::execution::env{buf.stream(), buf.memory_resource()};
        }),
        ab | cuda::std::views::transform(cuda::std::ranges::begin),
        ab | cuda::std::views::transform(cuda::std::ranges::size),
        c | cuda::std::views::transform(cuda::std::ranges::begin),
        [] __device__(const auto& in) {
          return cuda::std::get<0>(in) + cuda::std::get<1>(in);
        });
    });
  });
}
NVBENCH_BENCH(add).set_name("add").add_int64_power_of_two_axis(
  "Elements", nvbench::range(min_elements_pow2, max_elements_pow2, elements_stride));

void triad(nvbench::state& state)
{
  const auto elements = static_cast<cuda::std::size_t>(state.get_int64("Elements"));
  const auto device   = state_device(state);
  const auto domains  = device.__locality_domains();

  if (domains.size() < 2)
  {
    state.skip("the GPU does not expose multiple locality domains");
    return;
  }

  auto comms   = make_communicators(device, domains);
  auto streams = make_streams(domains);
  auto a       = make_buffers(domains, streams, elements, start_a);
  auto b       = make_buffers(domains, streams, elements, start_b);
  auto c       = make_buffers(domains, streams, elements, start_c);

  cuda::event fork{device};
  auto join = make_events(device, domains.size());

  const auto total = elements * domains.size();

  state.add_element_count(total);
  state.add_global_memory_reads<T>(2 * total);
  state.add_global_memory_writes<T>(total);

  // `transform` takes one input iterator, so each rank's two inputs are zipped into one range.
  // The zipped range iterates as a tuple of both elements.
  auto bc = cuda::std::views::zip(b, c) | cuda::std::views::transform([](auto&& in) {
              return cuda::std::views::zip(cuda::std::get<0>(in), cuda::std::get<1>(in));
            });

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    run_forked_iteration(cuda::stream_ref{launch.get_stream().get_stream()}, a, fork, join, [&] {
      cudax::transform(
        cudax::distributed,
        comms,
        a | cuda::std::views::transform([](auto& buf) {
          return cuda::std::execution::env{buf.stream(), buf.memory_resource()};
        }),
        bc | cuda::std::views::transform(cuda::std::ranges::begin),
        bc | cuda::std::views::transform(cuda::std::ranges::size),
        a | cuda::std::views::transform(cuda::std::ranges::begin),
        [] __device__(const auto& in) {
          return cuda::std::get<0>(in) + (scalar * cuda::std::get<1>(in));
        });
    });
  });
}
NVBENCH_BENCH(triad).set_name("triad").add_int64_power_of_two_axis(
  "Elements", nvbench::range(min_elements_pow2, max_elements_pow2, elements_stride));
} // namespace
