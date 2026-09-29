//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc
// UNSUPPORTED: libcpp-no-exceptions
// REQUIRES: linux
// ADDITIONAL_COMPILE_DEFINITIONS: _LIBCUDACXX_FORCE_INCLUDE_H

#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/launch>
#include <cuda/logical_endpoint>
#include <cuda/memory_pool>
#include <cuda/std/cstdint>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>

#include <cuda_runtime_api.h>
#include <unistd.h>

#include "logical_endpoint_test_helper.h"
#include "test_macros.h"
#include <sys/wait.h>

#if _CCCL_CTK_AT_LEAST(13, 3)

namespace
{
constexpr int driver_version_13_4 = 13040;

enum class child_command : int
{
  skip,
  import_unicast_handle,
  put_to_unicast_endpoint,
  import_multicast_handle,
  finish
};

enum class child_result : int
{
  success,
  failure
};

struct child_request
{
  child_command command{};
  cuda::logical_endpoint_fabric_handle handle{};
};

static_assert(cuda::std::is_trivially_copyable_v<child_request>);

bool read_exactly(int fd, void* data, size_t bytes)
{
  auto* cursor = static_cast<unsigned char*>(data);
  while (bytes != 0)
  {
    const ssize_t count = ::read(fd, cursor, bytes);
    if (count == 0)
    {
      return false;
    }
    if (count < 0)
    {
      if (errno == EINTR)
      {
        continue;
      }
      return false;
    }
    cursor += count;
    bytes -= static_cast<size_t>(count);
  }
  return true;
}

bool write_exactly(int fd, const void* data, size_t bytes)
{
  const auto* cursor = static_cast<const unsigned char*>(data);
  while (bytes != 0)
  {
    const ssize_t count = ::write(fd, cursor, bytes);
    if (count < 0)
    {
      if (errno == EINTR)
      {
        continue;
      }
      return false;
    }
    cursor += count;
    bytes -= static_cast<size_t>(count);
  }
  return true;
}

void close_fd(int fd)
{
  if (fd >= 0)
  {
    static_cast<void>(::close(fd));
  }
}

bool report_child_result(int fd, child_result result)
{
  return write_exactly(fd, &result, sizeof(result));
}

int wait_for_child(pid_t child)
{
  int status = 0;
  while (::waitpid(child, &status, 0) < 0)
  {
    if (errno == EINTR)
    {
      continue;
    }
    std::perror("waitpid");
    return EXIT_FAILURE;
  }

  if (WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS)
  {
    return EXIT_SUCCESS;
  }

  std::fprintf(stderr, "child process failed with wait status %d\n", status);
  return EXIT_FAILURE;
}

bool send_request(int fd, child_command command, const cuda::logical_endpoint_fabric_handle& handle = {})
{
  child_request request{};
  request.command = command;
  request.handle  = handle;
  return write_exactly(fd, &request, sizeof(request));
}

struct parent_child
{
  int request_fd;
  int result_fd;
  pid_t pid;
  bool waited = false;
};

int wait_for(parent_child& child)
{
  if (child.waited)
  {
    return EXIT_SUCCESS;
  }

  child.waited = true;
  return wait_for_child(child.pid);
}

void send_no_throw(parent_child& child, child_command command)
{
  if (!child.waited)
  {
    static_cast<void>(send_request(child.request_fd, command));
  }
}

int skip(parent_child& child, const char* reason)
{
  std::fprintf(stderr, "skipping: %s\n", reason);
  if (!send_request(child.request_fd, child_command::skip))
  {
    std::fprintf(stderr, "parent failed to send skip request\n");
    return EXIT_FAILURE;
  }
  return wait_for(child);
}

void abort_child_no_throw(parent_child& child)
{
  if (!child.waited)
  {
    send_no_throw(child, child_command::skip);
    static_cast<void>(wait_for(child));
  }
}

void send_or_throw(
  parent_child& child,
  child_command command,
  const cuda::logical_endpoint_fabric_handle& handle,
  const char* failure_message)
{
  if (!send_request(child.request_fd, command, handle))
  {
    throw std::runtime_error(failure_message);
  }
}

child_result read_child_result(parent_child& child)
{
  child_result result = child_result::failure;
  if (!read_exactly(child.result_fd, &result, sizeof(result)))
  {
    throw std::runtime_error("parent failed to read the child result");
  }
  return result;
}

void require_success_and_wait(
  parent_child& child,
  child_command command,
  const cuda::logical_endpoint_fabric_handle& handle,
  const char* send_failure_message,
  const char* child_failure_message)
{
  send_or_throw(child, command, handle, send_failure_message);
  const child_result result = read_child_result(child);
  const int child_status   = wait_for(child);
  if (child_status != EXIT_SUCCESS || result != child_result::success)
  {
    throw std::runtime_error(child_failure_message);
  }
}

void request_unicast_import(parent_child& child, const cuda::logical_endpoint_fabric_handle& handle)
{
  require_success_and_wait(
    child,
    child_command::import_unicast_handle,
    handle,
    "parent failed to send the unicast logical endpoint handle",
    "child failed to import the unicast logical endpoint");
}

void request_unicast_put(parent_child& child, const cuda::logical_endpoint_fabric_handle& handle)
{
  require_success_and_wait(
    child,
    child_command::put_to_unicast_endpoint,
    handle,
    "parent failed to send the unicast logical endpoint handle",
    "child failed to put to imported unicast logical endpoint");
}

void request_multicast_import(parent_child& child, const cuda::logical_endpoint_fabric_handle& handle)
{
  send_or_throw(
    child,
    child_command::import_multicast_handle,
    handle,
    "parent failed to send the multicast logical endpoint handle");
  if (read_child_result(child) != child_result::success)
  {
    static_cast<void>(wait_for(child));
    throw std::runtime_error("child failed to import the multicast logical endpoint");
  }
}

int finish_multicast_import(parent_child& child)
{
  send_or_throw(child, child_command::finish, {}, "parent failed to send multicast finish command");
  return wait_for(child);
}

struct skipped_test
{
  const char* reason;
};

#define SKIP_IF_REASON(reason_expr)                                        \
  do                                                                       \
  {                                                                        \
    if (const char* cccl_logical_endpoint_skip_reason = (reason_expr))     \
    {                                                                      \
      throw skipped_test{cccl_logical_endpoint_skip_reason};               \
    }                                                                      \
  } while (false)

#define SKIP_UNLESS(condition, reason)                         \
  do                                                          \
  {                                                           \
    if (!(condition))                                         \
    {                                                         \
      throw skipped_test{reason};                             \
    }                                                         \
  } while (false)

#define FAIL_UNLESS(condition, reason)                         \
  do                                                          \
  {                                                           \
    if (!(condition))                                         \
    {                                                         \
      throw std::runtime_error(reason);                       \
    }                                                         \
  } while (false)

template <class... Devices>
const char* fabric_ipc_unsupported_reason(int minimum_device_count, cuda::device_ref device, Devices... devices)
{
  if (const char* reason = logical_endpoint_test::runtime_unsupported_reason(minimum_device_count))
  {
    return reason;
  }

#  if _CCCL_CTK_AT_LEAST(13, 4)
  int driver_version = 0;
  if (::cudaDriverGetVersion(&driver_version) != cudaSuccess)
  {
    return "logical endpoint fabric IPC tests could not query the CUDA driver version";
  }
  if (driver_version < driver_version_13_4)
  {
    return "logical endpoint fabric IPC tests require a CUDA 13.4 driver";
  }

  try
  {
    const cuda::device_ref checked_devices[] = {device, devices...};
    for (const cuda::device_ref checked_device : checked_devices)
    {
      const auto supported_handle_types =
        checked_device.attribute(cuda::device_attributes::logical_endpoint_supported_handle_types);
      if ((supported_handle_types & cuda::std::to_underlying(cuda::logical_endpoint_ipc_handle_type::fabric)) == 0)
      {
        return "fabric logical endpoint IPC handles are not supported by the selected device";
      }
    }
  }
  catch (const std::exception&)
  {
    return "logical endpoint fabric IPC support could not be queried";
  }
  return nullptr;
#  else // ^^^ _CCCL_CTK_AT_LEAST(13, 4) ^^^ / vvv _CCCL_CTK_BELOW(13, 4) vvv
  (void) device;
  ((void) devices, ...);
  return "logical endpoint fabric IPC tests require CUDA Toolkit 13.4";
#  endif // _CCCL_CTK_BELOW(13, 4)
}

bool endpoint_size(cuda::logical_endpoint_limits limits, cuda::std::uint64_t& bytes)
{
  bytes = logical_endpoint_test::endpoint_size(limits);
  return bytes >= logical_endpoint_test::minimum_bytes && limits.bind_alignment != 0
      && (limits.max_size == 0 || bytes <= limits.max_size);
}

struct endpoint_config
{
  cuda::logical_endpoint_limits limits{};
  cuda::std::uint64_t bytes{};
};

template <class Spec, class... Devices>
bool probe_endpoint_support(
  const Spec& spec, logical_endpoint_test::support_result& support, const char*& reason, Devices... devices)
{
  try
  {
    support = logical_endpoint_test::probe_logical_endpoint_support(spec, devices...);
  }
  catch (const std::exception&)
  {
    reason = "logical endpoint support could not be queried";
    return false;
  }

  if (!support.supported)
  {
    reason = support.reason;
    return false;
  }
  return true;
}

template <class Spec, class... Devices>
endpoint_config require_endpoint_config(const Spec& spec, Devices... devices)
{
  logical_endpoint_test::support_result support{};
  const char* support_reason = nullptr;
  SKIP_UNLESS(probe_endpoint_support(spec, support, support_reason, devices...), support_reason);

  cuda::std::uint64_t bytes = 0;
  FAIL_UNLESS(endpoint_size(support.limits, bytes), "logical endpoint smoke size is not valid for reported limits");
  return {support.limits, bytes};
}

cuda::unicast_logical_endpoint import_unicast(const cuda::logical_endpoint_fabric_handle& handle)
{
  cuda::unicast_logical_endpoint imported{handle};
  if (!imported.wait_ready_for(logical_endpoint_test::ready_timeout))
  {
    throw std::runtime_error("imported unicast logical endpoint did not become ready");
  }
  return imported;
}

cuda::multicast_logical_endpoint import_multicast(
  const cuda::logical_endpoint_fabric_handle& handle, cuda::device_ref device)
{
  cuda::multicast_logical_endpoint imported{handle};
  imported.add_device(device);
  if (!imported.wait_ready_for(logical_endpoint_test::ready_timeout))
  {
    throw std::runtime_error("imported multicast logical endpoint did not become ready");
  }
  return imported;
}

void put_to_imported_unicast(const cuda::logical_endpoint_fabric_handle& handle)
{
  cuda::device_ref device{0};
  cuda::stream stream{device};
  auto status = cuda::make_device_buffer<cuda::std::uint32_t>(stream, device, 1, cuda::no_init);
  cuda::fill_bytes(stream, status, 0);

  cuda::unicast_logical_endpoint imported = import_unicast(handle);
  auto config = cuda::make_config(cuda::make_hierarchy(cuda::grid_dims(1), cuda::block_dims<1>()));
  cuda::launch(
    stream,
    config,
    logical_endpoint_test::fabric_try_put_smoke_kernel,
    imported,
    cuda::std::uint64_t{0},
    status.data());
  stream.sync();

  cuda::std::uint32_t host_status = 0;
  cuda::copy_bytes(stream, status, cuda::std::span<cuda::std::uint32_t>{&host_status, 1});
  stream.sync();
  if (host_status != logical_endpoint_test::status_success)
  {
    throw std::runtime_error("child fabric put kernel failed");
  }
}

template <class Action>
int run_child_action(int result_fd, const char* failure_message, const char* report_failure_message, Action action)
{
  try
  {
    action();
  }
  catch (const std::exception& e)
  {
    std::fprintf(stderr, "%s: %s\n", failure_message, e.what());
    static_cast<void>(report_child_result(result_fd, child_result::failure));
    return EXIT_FAILURE;
  }

  if (!report_child_result(result_fd, child_result::success))
  {
    std::fprintf(stderr, "%s\n", report_failure_message);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int child_import_unicast(const cuda::logical_endpoint_fabric_handle& handle, int result_fd)
{
  return run_child_action(
    result_fd,
    "child failed to import the unicast logical endpoint",
    "child failed to report unicast import success",
    [&handle] {
      static_cast<void>(import_unicast(handle));
    });
}

int child_put_to_unicast(const cuda::logical_endpoint_fabric_handle& handle, int result_fd)
{
  return run_child_action(
    result_fd,
    "child failed to put to imported unicast logical endpoint",
    "child failed to report fabric put success",
    [&handle] {
      put_to_imported_unicast(handle);
    });
}

int child_import_multicast(const cuda::logical_endpoint_fabric_handle& handle, int request_fd, int result_fd)
{
  try
  {
    cuda::multicast_logical_endpoint imported = import_multicast(handle, cuda::device_ref{1});
    // Keep the imported endpoint alive until the parent has completed its multicast bind check.
    static_cast<void>(imported.id());

    if (!report_child_result(result_fd, child_result::success))
    {
      std::fprintf(stderr, "child failed to report multicast import success\n");
      return EXIT_FAILURE;
    }

    child_request finish{};
    if (!read_exactly(request_fd, &finish, sizeof(finish)) || finish.command != child_command::finish)
    {
      std::fprintf(stderr, "child did not receive multicast finish command\n");
      return EXIT_FAILURE;
    }
  }
  catch (const std::exception& e)
  {
    std::fprintf(stderr, "child failed to import the multicast logical endpoint: %s\n", e.what());
    static_cast<void>(report_child_result(result_fd, child_result::failure));
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int child_main(int request_fd, int result_fd)
{
  child_request request{};
  if (!read_exactly(request_fd, &request, sizeof(request)))
  {
    std::fprintf(stderr, "child failed to read the logical endpoint request\n");
    return EXIT_FAILURE;
  }

  switch (request.command)
  {
    case child_command::skip:
      return EXIT_SUCCESS;
    case child_command::import_unicast_handle:
      return child_import_unicast(request.handle, result_fd);
    case child_command::put_to_unicast_endpoint:
      return child_put_to_unicast(request.handle, result_fd);
    case child_command::import_multicast_handle:
      return child_import_multicast(request.handle, request_fd, result_fd);
    case child_command::finish:
      break;
  }

  std::fprintf(stderr, "child received an invalid first command\n");
  return EXIT_FAILURE;
}

template <class Action>
int run_parent_action(parent_child& child, const char* failure_message, Action action)
{
  try
  {
    return action();
  }
  catch (const skipped_test& skipped)
  {
    return skip(child, skipped.reason);
  }
  catch (const cuda::cuda_error& e)
  {
    if (e.status() == cudaErrorNoDevice)
    {
      return skip(child, "logical endpoint tests require a CUDA device");
    }
    std::fprintf(stderr, "%s: %s\n", failure_message, e.what());
  }
  catch (const std::exception& e)
  {
    std::fprintf(stderr, "%s: %s\n", failure_message, e.what());
  }

  abort_child_no_throw(child);
  return EXIT_FAILURE;
}

int parent_test_unicast_import(int request_fd, int result_fd, pid_t child)
{
  parent_child parent{request_fd, result_fd, child};
  cuda::unicast_logical_endpoint local;

  return run_parent_action(parent, "parent failed to create/export the unicast logical endpoint", [&] {
    cuda::device_ref device{0};
    SKIP_IF_REASON(fabric_ipc_unsupported_reason(1, device));

    auto spec = cuda::unicast_logical_endpoint_spec{device};
    const endpoint_config config = require_endpoint_config(spec, device);

    local = cuda::unicast_logical_endpoint{spec, config.bytes};
    FAIL_UNLESS(
      local.wait_ready_for(logical_endpoint_test::ready_timeout),
      "local unicast logical endpoint did not become ready");
    request_unicast_import(parent, local.export_endpoint(cuda::fabric_handle));
    return EXIT_SUCCESS;
  });
}

int parent_test_unicast_imported_put(int request_fd, int result_fd, pid_t child)
{
  parent_child parent{request_fd, result_fd, child};
  cuda::unicast_logical_endpoint local;

  return run_parent_action(parent, "parent failed to run imported unicast put test", [&] {
    cuda::device_ref device{0};
    SKIP_IF_REASON(fabric_ipc_unsupported_reason(1, device));
    SKIP_UNLESS(
      logical_endpoint_test::fabric_memory_pools_supported(device), "fabric memory pool allocations are not supported");
    SKIP_UNLESS(
      logical_endpoint_test::fabric_ptx_supported(device),
      "fabric PTX logical endpoint smoke requires an SM 100+ device and PTX ISA 9.3+");

    auto spec = cuda::unicast_logical_endpoint_spec{device};
    const endpoint_config config = require_endpoint_config(spec, device);

    const auto alignment        = config.limits.bind_alignment;
    const auto allocation_bytes = config.bytes + alignment;
    cuda::stream stream{device};
    cuda::shared_device_memory_pool resource{device, logical_endpoint_test::fabric_memory_pool_properties()};
    auto allocation = cuda::make_buffer<cuda::std::uint8_t>(stream, resource, allocation_bytes, cuda::no_init);
    stream.sync();

    const auto allocation_addr = reinterpret_cast<cuda::std::uintptr_t>(allocation.data());
    const auto bind_addr       = logical_endpoint_test::align_up(allocation_addr, alignment);
    void* bind_ptr             = reinterpret_cast<void*>(bind_addr);
    FAIL_UNLESS(
      bind_addr + config.bytes <= allocation_addr + allocation_bytes, "aligned bind range falls outside allocation");

    local = cuda::unicast_logical_endpoint{spec, config.bytes};
    FAIL_UNLESS(
      local.wait_ready_for(logical_endpoint_test::ready_timeout),
      "local unicast logical endpoint did not become ready");
    local.bind(device, 0, bind_ptr, config.bytes);
    // If this isolated test process fails after binding, process teardown is enough; keep the test body simple.
    cuda::fill_bytes(stream,
                     cuda::std::span<cuda::std::uint8_t>{
                       static_cast<cuda::std::uint8_t*>(bind_ptr), logical_endpoint_test::payload_bytes},
                     0);
    stream.sync();

    request_unicast_put(parent, local.export_endpoint(cuda::fabric_handle));

    cuda::std::uint32_t observed[logical_endpoint_test::payload_words]{};
    cuda::copy_bytes(stream,
                     cuda::std::span<cuda::std::uint32_t>{
                       static_cast<cuda::std::uint32_t*>(bind_ptr), logical_endpoint_test::payload_words},
                     cuda::std::span<cuda::std::uint32_t>{observed, logical_endpoint_test::payload_words});
    stream.sync();
    local.unbind(device, 0, config.bytes);

    FAIL_UNLESS(
      observed[0] == 0x13572468u && observed[1] == 0x24681357u && observed[2] == 0xdeadbeefu
        && observed[3] == 0xcafef00du,
      "parent observed unexpected payload");
    return EXIT_SUCCESS;
  });
}

int parent_test_multicast_import(int request_fd, int result_fd, pid_t child)
{
  parent_child parent{request_fd, result_fd, child};
  cuda::multicast_logical_endpoint local;

  return run_parent_action(parent, "parent failed to run multicast import test", [&] {
    cuda::device_ref device{0};
    cuda::device_ref child_device{1};
    SKIP_IF_REASON(fabric_ipc_unsupported_reason(2, device, child_device));
    SKIP_UNLESS(
      logical_endpoint_test::fabric_memory_pools_supported(device, child_device),
      "fabric memory pool allocations are not supported");

    auto spec = cuda::multicast_logical_endpoint_spec{2};
    const endpoint_config config = require_endpoint_config(spec, device, child_device);

    local = cuda::multicast_logical_endpoint{spec, config.bytes};
    local.add_device(device);
    request_multicast_import(parent, local.export_endpoint(cuda::fabric_handle));
    FAIL_UNLESS(
      local.wait_ready_for(logical_endpoint_test::ready_timeout),
      "local multicast logical endpoint did not become ready");

    const auto alignment        = config.limits.bind_alignment;
    const auto allocation_bytes = config.bytes + alignment;
    cuda::stream stream{device};
    cuda::shared_device_memory_pool resource{device, logical_endpoint_test::fabric_memory_pool_properties()};
    auto allocation = cuda::make_buffer<cuda::std::uint8_t>(stream, resource, allocation_bytes, cuda::no_init);
    stream.sync();

    const auto allocation_addr = reinterpret_cast<cuda::std::uintptr_t>(allocation.data());
    const auto bind_addr       = logical_endpoint_test::align_up(allocation_addr, alignment);
    void* bind_ptr             = reinterpret_cast<void*>(bind_addr);
    FAIL_UNLESS(
      bind_addr + config.bytes <= allocation_addr + allocation_bytes, "aligned bind range falls outside allocation");

    local.bind(device, 0, bind_ptr, config.bytes);
    local.unbind(device, 0, config.bytes);

    return finish_multicast_import(parent);
  });
}

using parent_case = int (*)(int, int, pid_t);

int run_child_case(parent_case parent)
{
  int parent_to_child[2] = {-1, -1};
  int child_to_parent[2] = {-1, -1};
  if (::pipe(parent_to_child) != 0 || ::pipe(child_to_parent) != 0)
  {
    std::perror("pipe");
    close_fd(parent_to_child[0]);
    close_fd(parent_to_child[1]);
    close_fd(child_to_parent[0]);
    close_fd(child_to_parent[1]);
    return EXIT_FAILURE;
  }

  // Fork before CUDA support probes. Some probes initialize CUDA, and forking after CUDA initialization is not
  // reliable, so unsupported configurations are reported to the child with a skip command.
  pid_t child = ::fork();
  if (child < 0)
  {
    std::perror("fork");
    close_fd(parent_to_child[0]);
    close_fd(parent_to_child[1]);
    close_fd(child_to_parent[0]);
    close_fd(child_to_parent[1]);
    return EXIT_FAILURE;
  }

  if (child == 0)
  {
    close_fd(parent_to_child[1]);
    close_fd(child_to_parent[0]);
    const int result = child_main(parent_to_child[0], child_to_parent[1]);
    close_fd(parent_to_child[0]);
    close_fd(child_to_parent[1]);
    std::_Exit(result);
  }

  close_fd(parent_to_child[0]);
  close_fd(child_to_parent[1]);
  const int result = parent(parent_to_child[1], child_to_parent[0], child);
  close_fd(parent_to_child[1]);
  close_fd(child_to_parent[0]);
  return result;
}

int run_isolated_child_case(parent_case parent)
{
  // Keep the original test process CUDA-clean. Each case runs in a fresh process that can fork its child before that
  // case performs CUDA support probes or creates contexts.
  const pid_t case_process = ::fork();
  if (case_process < 0)
  {
    std::perror("fork");
    return EXIT_FAILURE;
  }

  if (case_process == 0)
  {
    std::_Exit(run_child_case(parent));
  }

  return wait_for_child(case_process);
}

int run_test(int argc, char** argv)
{
  (void) argc;
  (void) argv;

  if (run_isolated_child_case(parent_test_unicast_import) != EXIT_SUCCESS)
  {
    return EXIT_FAILURE;
  }
  if (run_isolated_child_case(parent_test_unicast_imported_put) != EXIT_SUCCESS)
  {
    return EXIT_FAILURE;
  }
  if (run_isolated_child_case(parent_test_multicast_import) != EXIT_SUCCESS)
  {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

#undef FAIL_UNLESS
#undef SKIP_UNLESS
#undef SKIP_IF_REASON
} // namespace

#endif // _CCCL_CTK_AT_LEAST(13, 3)

#if _CCCL_CTK_AT_LEAST(13, 3)
int main(int argc, char** argv)
{
  return run_test(argc, argv);
}
#else // ^^^ _CCCL_CTK_AT_LEAST(13, 3) ^^^ / vvv !_CCCL_CTK_AT_LEAST(13, 3)
int main(int argc, char** argv)
{
  (void) argc;
  (void) argv;
  return 0;
}
#endif // _CCCL_CTK_AT_LEAST(13, 3)
