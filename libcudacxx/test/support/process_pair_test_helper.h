//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_PROCESS_PAIR_TEST_HELPER_H
#define TEST_SUPPORT_PROCESS_PAIR_TEST_HELPER_H

#include <cuda/std/type_traits>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <system_error>

#include <unistd.h>

#include <sys/socket.h>
#include <sys/wait.h>

namespace process_pair_test
{
struct skipped_test
{
  const char* reason;
};

inline void skip_unless(bool supported, const char* reason)
{
  if (!supported)
  {
    throw skipped_test{reason};
  }
}

namespace detail
{
struct peer_closed
{};

inline constexpr int skipped_status     = 77;
inline constexpr int peer_closed_status = 78;

inline int wait_for_process(pid_t pid)
{
  int status = 0;
  while (::waitpid(pid, &status, 0) < 0)
  {
    if (errno != EINTR)
    {
      throw std::system_error(errno, std::generic_category(), "waitpid");
    }
  }
  return WIFEXITED(status) ? WEXITSTATUS(status) : EXIT_FAILURE;
}
} // namespace detail

class ipc_channel
{
public:
  explicit ipc_channel(int fd)
      : fd_(fd)
  {}

  ipc_channel(const ipc_channel&)            = delete;
  ipc_channel& operator=(const ipc_channel&) = delete;

  ~ipc_channel()
  {
    static_cast<void>(::close(fd_));
  }

  template <class T>
  void send(const T& value)
  {
    static_assert(cuda::std::is_trivially_copyable_v<T>);
    const auto* cursor = reinterpret_cast<const char*>(&value);
    size_t remaining   = sizeof(T);
    while (remaining != 0)
    {
      const auto count = ::send(fd_, cursor, remaining, MSG_NOSIGNAL);
      if (count < 0)
      {
        if (errno == EINTR)
        {
          continue;
        }
        throw std::system_error(errno, std::generic_category(), "send");
      }
      if (count == 0)
      {
        throw std::runtime_error("send made no progress");
      }
      cursor += count;
      remaining -= static_cast<size_t>(count);
    }
  }

  template <class T>
  T receive()
  {
    static_assert(cuda::std::is_trivially_copyable_v<T>);
    T value{};
    auto* cursor     = reinterpret_cast<char*>(&value);
    size_t remaining = sizeof(T);
    while (remaining != 0)
    {
      const auto count = ::recv(fd_, cursor, remaining, 0);
      if (count < 0)
      {
        if (errno == EINTR)
        {
          continue;
        }
        throw std::system_error(errno, std::generic_category(), "receive");
      }
      if (count == 0)
      {
        if (remaining == sizeof(T))
        {
          throw detail::peer_closed{};
        }
        throw std::runtime_error("received an incomplete IPC message");
      }
      cursor += count;
      remaining -= static_cast<size_t>(count);
    }
    return value;
  }

  void sync()
  {
    // A host barrier; callbacks must drain GPU work before signaling its completion.
    send(char{});
    static_cast<void>(receive<char>());
  }

private:
  int fd_;
};

namespace detail
{
template <class Action>
int run_role(const char* name, const char* role, ipc_channel& peer, Action action)
{
  try
  {
    action(peer);
    return EXIT_SUCCESS;
  }
  catch (const skipped_test& skipped)
  {
    std::fprintf(stderr, "skipping %s: %s\n", name, skipped.reason);
    return skipped_status;
  }
  catch (const peer_closed&)
  {
    return peer_closed_status;
  }
  catch (const std::exception& error)
  {
    std::fprintf(stderr, "%s %s failed: %s\n", name, role, error.what());
    return EXIT_FAILURE;
  }
}
} // namespace detail

template <class Exporter, class Importer>
int run_process_pair(const char* name, Exporter exporter, Importer importer)
{
  // Call from a CUDA-clean process and construct CUDA objects inside the callbacks. The exporter may skip before
  // sending the first message; the importer must start by receiving that message.
  // Each scenario gets a fresh process so both roles fork before any CUDA initialization.
  const auto scenario = ::fork();
  if (scenario < 0)
  {
    throw std::system_error(errno, std::generic_category(), "fork scenario");
  }
  if (scenario != 0)
  {
    return detail::wait_for_process(scenario);
  }

  int result = EXIT_FAILURE;
  try
  {
    int sockets[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0)
    {
      throw std::system_error(errno, std::generic_category(), "socketpair");
    }

    const auto child = ::fork();
    if (child < 0)
    {
      const auto error = errno;
      static_cast<void>(::close(sockets[0]));
      static_cast<void>(::close(sockets[1]));
      throw std::system_error(error, std::generic_category(), "fork importer");
    }
    if (child == 0)
    {
      static_cast<void>(::close(sockets[0]));
      ipc_channel peer{sockets[1]};
      std::_Exit(detail::run_role(name, "importer", peer, importer));
    }

    static_cast<void>(::close(sockets[1]));
    int exporter_status;
    {
      ipc_channel peer{sockets[0]};
      exporter_status = detail::run_role(name, "exporter", peer, exporter);
    }
    // Closing the channel wakes an importer waiting for a handle when the exporter skips or fails.
    const auto importer_status = detail::wait_for_process(child);
    if ((exporter_status == EXIT_SUCCESS && importer_status == EXIT_SUCCESS)
        || (exporter_status == detail::skipped_status && importer_status == detail::peer_closed_status))
    {
      result = EXIT_SUCCESS;
    }
    else
    {
      std::fprintf(
        stderr, "%s failed: exporter status %d, importer status %d\n", name, exporter_status, importer_status);
    }
  }
  catch (const std::exception& error)
  {
    std::fprintf(stderr, "%s process setup failed: %s\n", name, error.what());
  }
  std::_Exit(result);
}
} // namespace process_pair_test

#endif // TEST_SUPPORT_PROCESS_PAIR_TEST_HELPER_H
