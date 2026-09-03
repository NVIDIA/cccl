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
#include <cuda/stream>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <exception>

#include <cuda_runtime_api.h>
#include <unistd.h>

#include "logical_endpoint_test_helper.h"
#include "test_macros.h"
#include <sys/wait.h>

#if _CCCL_CTK_AT_LEAST(13, 3)

namespace
{
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
  cuda::logical_endpoint_handle handle{};
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

bool send_request(int fd, child_command command, const cuda::logical_endpoint_handle& handle = {})
{
  child_request request{};
  request.command = command;
  request.handle  = handle;
  return write_exactly(fd, &request, sizeof(request));
}

int skip_parent(int request_fd, pid_t child, const char* reason)
{
  std::fprintf(stderr, "skipping: %s\n", reason);
  if (!send_request(request_fd, child_command::skip))
  {
    std::fprintf(stderr, "parent failed to send skip request\n");
    return EXIT_FAILURE;
  }
  return wait_for_child(child);
}

int fail_parent(int request_fd, pid_t child, const char* reason)
{
  std::fprintf(stderr, "%s\n", reason);
  static_cast<void>(send_request(request_fd, child_command::skip));
  static_cast<void>(wait_for_child(child));
  return EXIT_FAILURE;
}

bool endpoint_size(cuda::logical_endpoint_limits limits, cuda::std::uint64_t& bytes)
{
  bytes = logical_endpoint_test::endpoint_size(limits);
  return bytes >= logical_endpoint_test::minimum_bytes && limits.bind_alignment != 0
      && (limits.max_size == 0 || bytes <= limits.max_size);
}

int child_import_unicast(const cuda::logical_endpoint_handle& handle, int result_fd)
{
  try
  {
    cuda::unicast_logical_endpoint imported{handle};
    if (!imported.wait_until_ready(logical_endpoint_test::ready_timeout))
    {
      std::fprintf(stderr, "imported unicast logical endpoint did not become ready\n");
      static_cast<void>(report_child_result(result_fd, child_result::failure));
      return EXIT_FAILURE;
    }
  }
  catch (const std::exception& e)
  {
    std::fprintf(stderr, "child failed to import the unicast logical endpoint: %s\n", e.what());
    static_cast<void>(report_child_result(result_fd, child_result::failure));
    return EXIT_FAILURE;
  }

  if (!report_child_result(result_fd, child_result::success))
  {
    std::fprintf(stderr, "child failed to report unicast import success\n");
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int child_put_to_unicast(const cuda::logical_endpoint_handle& handle, int result_fd)
{
  try
  {
    cuda::device_ref device{0};
    cuda::stream stream{device};
    auto status = cuda::make_device_buffer<cuda::std::uint32_t>(stream, device, 1, cuda::no_init);
    cuda::fill_bytes(stream, status, 0);

    cuda::unicast_logical_endpoint imported{handle};
    if (!imported.wait_until_ready(logical_endpoint_test::ready_timeout))
    {
      std::fprintf(stderr, "imported unicast logical endpoint did not become ready\n");
      static_cast<void>(report_child_result(result_fd, child_result::failure));
      return EXIT_FAILURE;
    }

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
      std::fprintf(stderr, "child fabric put kernel status was %u\n", host_status);
      static_cast<void>(report_child_result(result_fd, child_result::failure));
      return EXIT_FAILURE;
    }
  }
  catch (const std::exception& e)
  {
    std::fprintf(stderr, "child failed to put to imported unicast logical endpoint: %s\n", e.what());
    static_cast<void>(report_child_result(result_fd, child_result::failure));
    return EXIT_FAILURE;
  }

  if (!report_child_result(result_fd, child_result::success))
  {
    std::fprintf(stderr, "child failed to report fabric put success\n");
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int child_import_multicast(const cuda::logical_endpoint_handle& handle, int request_fd, int result_fd)
{
  try
  {
    cuda::device_ref device{1};
    cuda::multicast_logical_endpoint imported{handle};
    imported.add_device(device);
    if (!imported.wait_until_ready(logical_endpoint_test::ready_timeout))
    {
      std::fprintf(stderr, "imported multicast logical endpoint did not become ready\n");
      static_cast<void>(report_child_result(result_fd, child_result::failure));
      return EXIT_FAILURE;
    }

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

int read_child_result_and_wait(int result_fd, pid_t child)
{
  child_result result = child_result::failure;
  if (!read_exactly(result_fd, &result, sizeof(result)))
  {
    std::fprintf(stderr, "parent failed to read the child result\n");
    static_cast<void>(wait_for_child(child));
    return EXIT_FAILURE;
  }

  const int child_status = wait_for_child(child);
  if (child_status != EXIT_SUCCESS || result != child_result::success)
  {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int parent_import_unicast(int request_fd, int result_fd, pid_t child)
{
  cuda::unicast_logical_endpoint local;

  try
  {
    if (const char* reason = logical_endpoint_test::runtime_unsupported_reason())
    {
      return skip_parent(request_fd, child, reason);
    }

    cuda::device_ref device{0};
    auto spec    = cuda::unicast_logical_endpoint_spec{device};
    auto support = logical_endpoint_test::probe_logical_endpoint_support(spec, device);
    if (!support.supported)
    {
      return skip_parent(request_fd, child, support.reason);
    }

    cuda::std::uint64_t bytes = 0;
    if (!endpoint_size(support.limits, bytes))
    {
      return fail_parent(request_fd, child, "logical endpoint smoke size is not valid for reported limits");
    }

    local = cuda::unicast_logical_endpoint{spec, bytes};
    if (!local.wait_until_ready(logical_endpoint_test::ready_timeout))
    {
      return fail_parent(request_fd, child, "local unicast logical endpoint did not become ready");
    }
    if (!send_request(request_fd, child_command::import_unicast_handle, local.export_handle()))
    {
      return fail_parent(request_fd, child, "parent failed to send the unicast logical endpoint handle");
    }
  }
  catch (const std::exception& e)
  {
    std::fprintf(stderr, "parent failed to create/export the unicast logical endpoint: %s\n", e.what());
    static_cast<void>(send_request(request_fd, child_command::skip));
    static_cast<void>(wait_for_child(child));
    return EXIT_FAILURE;
  }

  return read_child_result_and_wait(result_fd, child);
}

int parent_put_to_unicast(int request_fd, int result_fd, pid_t child)
{
  cuda::unicast_logical_endpoint local;

  try
  {
    if (const char* reason = logical_endpoint_test::runtime_unsupported_reason())
    {
      return skip_parent(request_fd, child, reason);
    }

    cuda::device_ref device{0};
    if (!logical_endpoint_test::fabric_memory_pools_supported(device))
    {
      return skip_parent(request_fd, child, "fabric memory pool allocations are not supported");
    }
    if (!logical_endpoint_test::fabric_ptx_supported(device))
    {
      return skip_parent(
        request_fd, child, "fabric PTX logical endpoint smoke requires an SM 100+ device and PTX ISA 9.3+");
    }

    auto spec    = cuda::unicast_logical_endpoint_spec{device};
    auto support = logical_endpoint_test::probe_logical_endpoint_support(spec, device);
    if (!support.supported)
    {
      return skip_parent(request_fd, child, support.reason);
    }

    cuda::std::uint64_t bytes = 0;
    if (!endpoint_size(support.limits, bytes))
    {
      return fail_parent(request_fd, child, "logical endpoint smoke size is not valid for reported limits");
    }

    const auto alignment        = support.limits.bind_alignment;
    const auto allocation_bytes = bytes + alignment;
    cuda::stream stream{device};
    cuda::shared_device_memory_pool resource{device, logical_endpoint_test::fabric_memory_pool_properties()};
    auto allocation = cuda::make_buffer<cuda::std::uint8_t>(stream, resource, allocation_bytes, cuda::no_init);
    stream.sync();

    const auto allocation_addr = reinterpret_cast<cuda::std::uintptr_t>(allocation.data());
    const auto bind_addr       = logical_endpoint_test::align_up(allocation_addr, alignment);
    void* bind_ptr             = reinterpret_cast<void*>(bind_addr);
    if (bind_addr + bytes > allocation_addr + allocation_bytes)
    {
      return fail_parent(request_fd, child, "aligned bind range falls outside allocation");
    }

    local = cuda::unicast_logical_endpoint{spec, bytes};
    if (!local.wait_until_ready(logical_endpoint_test::ready_timeout))
    {
      return fail_parent(request_fd, child, "local unicast logical endpoint did not become ready");
    }
    local.bind(device, 0, bind_ptr, bytes);
    cuda::fill_bytes(stream,
                     cuda::std::span<cuda::std::uint8_t>{
                       static_cast<cuda::std::uint8_t*>(bind_ptr), logical_endpoint_test::payload_bytes},
                     0);
    stream.sync();

    if (!send_request(request_fd, child_command::put_to_unicast_endpoint, local.export_handle()))
    {
      local.unbind(device, 0, bytes);
      return fail_parent(request_fd, child, "parent failed to send the unicast logical endpoint handle");
    }

    child_result result = child_result::failure;
    if (!read_exactly(result_fd, &result, sizeof(result)))
    {
      local.unbind(device, 0, bytes);
      std::fprintf(stderr, "parent failed to read the child result\n");
      static_cast<void>(wait_for_child(child));
      return EXIT_FAILURE;
    }
    const int child_status = wait_for_child(child);

    cuda::std::uint32_t observed[logical_endpoint_test::payload_words]{};
    cuda::copy_bytes(stream,
                     cuda::std::span<cuda::std::uint32_t>{
                       static_cast<cuda::std::uint32_t*>(bind_ptr), logical_endpoint_test::payload_words},
                     cuda::std::span<cuda::std::uint32_t>{observed, logical_endpoint_test::payload_words});
    stream.sync();
    local.unbind(device, 0, bytes);

    if (child_status != EXIT_SUCCESS || result != child_result::success)
    {
      return EXIT_FAILURE;
    }
    if (observed[0] != 0x13572468u || observed[1] != 0x24681357u || observed[2] != 0xdeadbeefu
        || observed[3] != 0xcafef00du)
    {
      std::fprintf(
        stderr,
        "parent observed unexpected payload: %08x %08x %08x %08x\n",
        observed[0],
        observed[1],
        observed[2],
        observed[3]);
      return EXIT_FAILURE;
    }
  }
  catch (const std::exception& e)
  {
    std::fprintf(stderr, "parent failed to run imported unicast put test: %s\n", e.what());
    static_cast<void>(send_request(request_fd, child_command::skip));
    static_cast<void>(wait_for_child(child));
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int parent_import_multicast(int request_fd, int result_fd, pid_t child)
{
  cuda::multicast_logical_endpoint local;

  try
  {
    if (const char* reason = logical_endpoint_test::runtime_unsupported_reason(2))
    {
      return skip_parent(request_fd, child, reason);
    }

    cuda::device_ref device{0};
    cuda::device_ref child_device{1};
    if (!logical_endpoint_test::fabric_memory_pools_supported(device, child_device))
    {
      return skip_parent(request_fd, child, "fabric memory pool allocations are not supported");
    }

    auto spec    = cuda::multicast_logical_endpoint_spec{2};
    auto support = logical_endpoint_test::probe_logical_endpoint_support(spec, device, child_device);
    if (!support.supported)
    {
      return skip_parent(request_fd, child, support.reason);
    }

    cuda::std::uint64_t bytes = 0;
    if (!endpoint_size(support.limits, bytes))
    {
      return fail_parent(request_fd, child, "logical endpoint smoke size is not valid for reported limits");
    }

    local = cuda::multicast_logical_endpoint{spec, bytes};
    local.add_device(device);
    if (!send_request(request_fd, child_command::import_multicast_handle, local.export_handle()))
    {
      return fail_parent(request_fd, child, "parent failed to send the multicast logical endpoint handle");
    }

    child_result result = child_result::failure;
    if (!read_exactly(result_fd, &result, sizeof(result)))
    {
      std::fprintf(stderr, "parent failed to read the child result\n");
      static_cast<void>(send_request(request_fd, child_command::skip));
      static_cast<void>(wait_for_child(child));
      return EXIT_FAILURE;
    }
    if (result != child_result::success || !local.wait_until_ready(logical_endpoint_test::ready_timeout))
    {
      static_cast<void>(send_request(request_fd, child_command::finish));
      static_cast<void>(wait_for_child(child));
      return EXIT_FAILURE;
    }

    const auto alignment        = support.limits.bind_alignment;
    const auto allocation_bytes = bytes + alignment;
    cuda::stream stream{device};
    cuda::shared_device_memory_pool resource{device, logical_endpoint_test::fabric_memory_pool_properties()};
    auto allocation = cuda::make_buffer<cuda::std::uint8_t>(stream, resource, allocation_bytes, cuda::no_init);
    stream.sync();

    const auto allocation_addr = reinterpret_cast<cuda::std::uintptr_t>(allocation.data());
    const auto bind_addr       = logical_endpoint_test::align_up(allocation_addr, alignment);
    void* bind_ptr             = reinterpret_cast<void*>(bind_addr);
    if (bind_addr + bytes > allocation_addr + allocation_bytes)
    {
      static_cast<void>(send_request(request_fd, child_command::finish));
      static_cast<void>(wait_for_child(child));
      std::fprintf(stderr, "aligned bind range falls outside allocation\n");
      return EXIT_FAILURE;
    }

    local.bind(device, 0, bind_ptr, bytes);
    local.unbind(device, 0, bytes);

    if (!send_request(request_fd, child_command::finish))
    {
      std::fprintf(stderr, "parent failed to send multicast finish command\n");
      static_cast<void>(wait_for_child(child));
      return EXIT_FAILURE;
    }
  }
  catch (const std::exception& e)
  {
    std::fprintf(stderr, "parent failed to run multicast import test: %s\n", e.what());
    static_cast<void>(send_request(request_fd, child_command::skip));
    static_cast<void>(wait_for_child(child));
    return EXIT_FAILURE;
  }

  return wait_for_child(child);
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

  if (run_isolated_child_case(parent_import_unicast) != EXIT_SUCCESS)
  {
    return EXIT_FAILURE;
  }
  if (run_isolated_child_case(parent_put_to_unicast) != EXIT_SUCCESS)
  {
    return EXIT_FAILURE;
  }
  if (run_isolated_child_case(parent_import_multicast) != EXIT_SUCCESS)
  {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
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
