// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! The STREAM benchmark, distributed over one device, with every kernel expressed through
//! `cudax::transform`. The "Locality" axis runs each variant over the locality domains of the
//! device, and over the whole device as a single logical device.

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
#include <string_view>
#include <vector>

#include <nccl.h>

#include <nvbench/nvbench.cuh>

namespace cudax = cuda::experimental;

namespace
{
using element_types = nvbench::type_list<float, double>;

template <class T>
inline constexpr T start_a = T{1};
template <class T>
inline constexpr T start_b = T{2};
template <class T>
inline constexpr T start_c = T{0};
template <class T>
inline constexpr T scalar = T{3};

inline constexpr int min_elements_pow2 = 26;
inline constexpr int max_elements_pow2 = 32;
inline constexpr int elements_stride   = 1;

enum class locality : cuda::std::int8_t
{
  split,
  whole,
};

constexpr locality ALL_LOCALITIES[] = {locality::split, locality::whole};

[[nodiscard]] std::string_view to_string(locality loc)
{
  switch (loc)
  {
    case locality::split:
      return "split";
    case locality::whole:
      return "whole";
  }
  throw std::runtime_error{"Unknown locality kind: " + std::to_string(static_cast<cuda::std::int8_t>(loc))};
}

[[nodiscard]] locality locality_from_string(std::string_view str)
{
  for (const auto loc : ALL_LOCALITIES)
  {
    if (to_string(loc) == str)
    {
      return loc;
    }
  }
  throw std::runtime_error{"unknown locality: " + std::string{str}};
}

[[nodiscard]] std::vector<std::string> locality_axis_values()
{
  std::vector<std::string> values;

  values.reserve(cuda::std::size(ALL_LOCALITIES));
  for (const auto loc : ALL_LOCALITIES)
  {
    values.emplace_back(to_string(loc));
  }

  return values;
}

void add_summary(nvbench::state& state, cuda::std::size_t num_domains)
{
  auto& summary = state.add_summary("mgmn/locality_domains");

  summary.set_string("name", "Domains");
  summary.set_string("hint", "");
  summary.set_string("description", "Locality domains the device was split into");
  summary.set_int64("value", static_cast<cuda::std::int64_t>(num_domains));
}

[[nodiscard]] cuda::device_ref state_device(nvbench::state& state)
{
  return cuda::devices[state.get_device().value().get_id()]; // NOLINT(bugprone-unchecked-optional-access)
}

[[nodiscard]] std::vector<cuda::__logical_device_ref> state_domains(nvbench::state& state, cuda::device_ref device)
{
  const auto loc = locality_from_string(state.get_string("Locality"));

  switch (loc)
  {
    case locality::whole:
      return {cuda::__logical_device_ref{device}};
    case locality::split: {
      const auto domains = device.__locality_domains();

      return {domains.begin(), domains.end()};
    }
  }
  throw std::runtime_error{"Unknown locality kind: " + std::to_string(static_cast<cuda::std::int8_t>(loc))};
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

template <class T>
[[nodiscard]] std::vector<cuda::device_buffer<T>> make_buffers(
  cuda::std::span<const cuda::__logical_device_ref> domains,
  const std::vector<cuda::stream>& streams,
  cuda::std::size_t elements,
  const T& value)
{
  std::vector<cuda::device_buffer<T>> bufs;

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
  for (auto&& [event, buf] : cuda::std::views::zip(join, bufs))
  {
    event.record(buf.stream());
  }

  for (auto& event : join)
  {
    timing_stream.wait(event);
  }
}

template <class T>
void copy(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<cuda::std::size_t>(state.get_int64("Elements"));
  const auto device   = state_device(state);
  const auto domains  = state_domains(state, device);

  add_summary(state, domains.size());

  auto comms   = make_communicators(device, domains);
  auto streams = make_streams(domains);
  auto a       = make_buffers<T>(domains, streams, elements, start_a<T>);
  auto c       = make_buffers<T>(domains, streams, elements, start_c<T>);

  cuda::event fork{device};
  auto join = make_events(device, domains.size());

  const auto total = elements * domains.size();
  state.add_element_count(total);
  state.add_global_memory_reads<T>(total);
  state.add_global_memory_writes<T>(total);

  for (auto&& s : streams)
  {
    s.sync();
  }

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
NVBENCH_BENCH_TYPES(copy, NVBENCH_TYPE_AXES(element_types))
  .set_name("copy")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(min_elements_pow2, max_elements_pow2, elements_stride))
  .add_string_axis("Locality", locality_axis_values());

template <class T>
void mul(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<cuda::std::size_t>(state.get_int64("Elements"));
  const auto device   = state_device(state);
  const auto domains  = state_domains(state, device);

  add_summary(state, domains.size());

  auto comms   = make_communicators(device, domains);
  auto streams = make_streams(domains);
  auto b       = make_buffers<T>(domains, streams, elements, start_b<T>);
  auto c       = make_buffers<T>(domains, streams, elements, start_c<T>);

  cuda::event fork{device};
  auto join = make_events(device, domains.size());

  const auto total = elements * domains.size();
  state.add_element_count(total);
  state.add_global_memory_reads<T>(total);
  state.add_global_memory_writes<T>(total);

  for (auto&& s : streams)
  {
    s.sync();
  }

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
        [] __device__(const T ci) -> T {
          return scalar<T> * ci;
        });
    });
  });
}
NVBENCH_BENCH_TYPES(mul, NVBENCH_TYPE_AXES(element_types))
  .set_name("mul")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(min_elements_pow2, max_elements_pow2, elements_stride))
  .add_string_axis("Locality", locality_axis_values());

template <class T>
void add(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<cuda::std::size_t>(state.get_int64("Elements"));
  const auto device   = state_device(state);
  const auto domains  = state_domains(state, device);

  add_summary(state, domains.size());

  auto comms   = make_communicators(device, domains);
  auto streams = make_streams(domains);
  auto a       = make_buffers<T>(domains, streams, elements, start_a<T>);
  auto b       = make_buffers<T>(domains, streams, elements, start_b<T>);
  auto c       = make_buffers<T>(domains, streams, elements, start_c<T>);

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

  for (auto&& s : streams)
  {
    s.sync();
  }

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
        [] __device__(const auto in) -> T {
          return cuda::std::get<0>(in) + cuda::std::get<1>(in);
        });
    });
  });
}
NVBENCH_BENCH_TYPES(add, NVBENCH_TYPE_AXES(element_types))
  .set_name("add")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(min_elements_pow2, max_elements_pow2, elements_stride))
  .add_string_axis("Locality", locality_axis_values());

template <class T>
void triad(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<cuda::std::size_t>(state.get_int64("Elements"));
  const auto device   = state_device(state);
  const auto domains  = state_domains(state, device);

  add_summary(state, domains.size());

  auto comms   = make_communicators(device, domains);
  auto streams = make_streams(domains);
  auto a       = make_buffers<T>(domains, streams, elements, start_a<T>);
  auto b       = make_buffers<T>(domains, streams, elements, start_b<T>);
  auto c       = make_buffers<T>(domains, streams, elements, start_c<T>);

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

  for (auto&& s : streams)
  {
    s.sync();
  }

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
        [] __device__(const auto in) -> T {
          return cuda::std::get<0>(in) + (scalar<T> * cuda::std::get<1>(in));
        });
    });
  });
}
NVBENCH_BENCH_TYPES(triad, NVBENCH_TYPE_AXES(element_types))
  .set_name("triad")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(min_elements_pow2, max_elements_pow2, elements_stride))
  .add_string_axis("Locality", locality_axis_values());
} // namespace
