//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <cuda/std/__system_error_>
#include <cuda/std/utility>

#include <cerrno>

static_assert(cuda::std::to_underlying(cuda::std::errc::address_family_not_supported) == EAFNOSUPPORT);
static_assert(cuda::std::to_underlying(cuda::std::errc::address_in_use) == EADDRINUSE);
static_assert(cuda::std::to_underlying(cuda::std::errc::address_not_available) == EADDRNOTAVAIL);
static_assert(cuda::std::to_underlying(cuda::std::errc::already_connected) == EISCONN);
static_assert(cuda::std::to_underlying(cuda::std::errc::argument_list_too_long) == E2BIG);
static_assert(cuda::std::to_underlying(cuda::std::errc::argument_out_of_domain) == EDOM);
static_assert(cuda::std::to_underlying(cuda::std::errc::bad_address) == EFAULT);
static_assert(cuda::std::to_underlying(cuda::std::errc::bad_file_descriptor) == EBADF);
static_assert(cuda::std::to_underlying(cuda::std::errc::bad_message) == EBADMSG);
static_assert(cuda::std::to_underlying(cuda::std::errc::broken_pipe) == EPIPE);
static_assert(cuda::std::to_underlying(cuda::std::errc::connection_aborted) == ECONNABORTED);
static_assert(cuda::std::to_underlying(cuda::std::errc::connection_already_in_progress) == EALREADY);
static_assert(cuda::std::to_underlying(cuda::std::errc::connection_refused) == ECONNREFUSED);
static_assert(cuda::std::to_underlying(cuda::std::errc::connection_reset) == ECONNRESET);
static_assert(cuda::std::to_underlying(cuda::std::errc::cross_device_link) == EXDEV);
static_assert(cuda::std::to_underlying(cuda::std::errc::destination_address_required) == EDESTADDRREQ);
static_assert(cuda::std::to_underlying(cuda::std::errc::device_or_resource_busy) == EBUSY);
static_assert(cuda::std::to_underlying(cuda::std::errc::directory_not_empty) == ENOTEMPTY);
static_assert(cuda::std::to_underlying(cuda::std::errc::executable_format_error) == ENOEXEC);
static_assert(cuda::std::to_underlying(cuda::std::errc::file_exists) == EEXIST);
static_assert(cuda::std::to_underlying(cuda::std::errc::file_too_large) == EFBIG);
static_assert(cuda::std::to_underlying(cuda::std::errc::filename_too_long) == ENAMETOOLONG);
static_assert(cuda::std::to_underlying(cuda::std::errc::function_not_supported) == ENOSYS);
static_assert(cuda::std::to_underlying(cuda::std::errc::host_unreachable) == EHOSTUNREACH);
static_assert(cuda::std::to_underlying(cuda::std::errc::identifier_removed) == EIDRM);
static_assert(cuda::std::to_underlying(cuda::std::errc::illegal_byte_sequence) == EILSEQ);
static_assert(cuda::std::to_underlying(cuda::std::errc::inappropriate_io_control_operation) == ENOTTY);
static_assert(cuda::std::to_underlying(cuda::std::errc::interrupted) == EINTR);
static_assert(cuda::std::to_underlying(cuda::std::errc::invalid_argument) == EINVAL);
static_assert(cuda::std::to_underlying(cuda::std::errc::invalid_seek) == ESPIPE);
static_assert(cuda::std::to_underlying(cuda::std::errc::io_error) == EIO);
static_assert(cuda::std::to_underlying(cuda::std::errc::is_a_directory) == EISDIR);
static_assert(cuda::std::to_underlying(cuda::std::errc::message_size) == EMSGSIZE);
static_assert(cuda::std::to_underlying(cuda::std::errc::network_down) == ENETDOWN);
static_assert(cuda::std::to_underlying(cuda::std::errc::network_reset) == ENETRESET);
static_assert(cuda::std::to_underlying(cuda::std::errc::network_unreachable) == ENETUNREACH);
static_assert(cuda::std::to_underlying(cuda::std::errc::no_buffer_space) == ENOBUFS);
static_assert(cuda::std::to_underlying(cuda::std::errc::no_child_process) == ECHILD);
static_assert(cuda::std::to_underlying(cuda::std::errc::no_link) == ENOLINK);
static_assert(cuda::std::to_underlying(cuda::std::errc::no_lock_available) == ENOLCK);
static_assert(cuda::std::to_underlying(cuda::std::errc::no_message_available) == ENODATA);
static_assert(cuda::std::to_underlying(cuda::std::errc::no_message) == ENOMSG);
static_assert(cuda::std::to_underlying(cuda::std::errc::no_protocol_option) == ENOPROTOOPT);
static_assert(cuda::std::to_underlying(cuda::std::errc::no_space_on_device) == ENOSPC);
static_assert(cuda::std::to_underlying(cuda::std::errc::no_stream_resources) == ENOSR);
static_assert(cuda::std::to_underlying(cuda::std::errc::no_such_device_or_address) == ENXIO);
static_assert(cuda::std::to_underlying(cuda::std::errc::no_such_device) == ENODEV);
static_assert(cuda::std::to_underlying(cuda::std::errc::no_such_file_or_directory) == ENOENT);
static_assert(cuda::std::to_underlying(cuda::std::errc::no_such_process) == ESRCH);
static_assert(cuda::std::to_underlying(cuda::std::errc::not_a_directory) == ENOTDIR);
static_assert(cuda::std::to_underlying(cuda::std::errc::not_a_socket) == ENOTSOCK);
static_assert(cuda::std::to_underlying(cuda::std::errc::not_a_stream) == ENOSTR);
static_assert(cuda::std::to_underlying(cuda::std::errc::not_connected) == ENOTCONN);
static_assert(cuda::std::to_underlying(cuda::std::errc::not_enough_memory) == ENOMEM);
static_assert(cuda::std::to_underlying(cuda::std::errc::not_supported) == ENOTSUP);
static_assert(cuda::std::to_underlying(cuda::std::errc::operation_canceled) == ECANCELED);
static_assert(cuda::std::to_underlying(cuda::std::errc::operation_in_progress) == EINPROGRESS);
static_assert(cuda::std::to_underlying(cuda::std::errc::operation_not_permitted) == EPERM);
static_assert(cuda::std::to_underlying(cuda::std::errc::operation_not_supported) == EOPNOTSUPP);
static_assert(cuda::std::to_underlying(cuda::std::errc::operation_would_block) == EWOULDBLOCK);
static_assert(cuda::std::to_underlying(cuda::std::errc::owner_dead) == EOWNERDEAD);
static_assert(cuda::std::to_underlying(cuda::std::errc::permission_denied) == EACCES);
static_assert(cuda::std::to_underlying(cuda::std::errc::protocol_error) == EPROTO);
static_assert(cuda::std::to_underlying(cuda::std::errc::protocol_not_supported) == EPROTONOSUPPORT);
static_assert(cuda::std::to_underlying(cuda::std::errc::read_only_file_system) == EROFS);
static_assert(cuda::std::to_underlying(cuda::std::errc::resource_deadlock_would_occur) == EDEADLK);
static_assert(cuda::std::to_underlying(cuda::std::errc::resource_unavailable_try_again) == EAGAIN);
static_assert(cuda::std::to_underlying(cuda::std::errc::result_out_of_range) == ERANGE);
static_assert(cuda::std::to_underlying(cuda::std::errc::state_not_recoverable) == ENOTRECOVERABLE);
static_assert(cuda::std::to_underlying(cuda::std::errc::stream_timeout) == ETIME);
static_assert(cuda::std::to_underlying(cuda::std::errc::text_file_busy) == ETXTBSY);
static_assert(cuda::std::to_underlying(cuda::std::errc::timed_out) == ETIMEDOUT);
static_assert(cuda::std::to_underlying(cuda::std::errc::too_many_files_open_in_system) == ENFILE);
static_assert(cuda::std::to_underlying(cuda::std::errc::too_many_files_open) == EMFILE);
static_assert(cuda::std::to_underlying(cuda::std::errc::too_many_links) == EMLINK);
static_assert(cuda::std::to_underlying(cuda::std::errc::too_many_symbolic_link_levels) == ELOOP);
static_assert(cuda::std::to_underlying(cuda::std::errc::value_too_large) == EOVERFLOW);
static_assert(cuda::std::to_underlying(cuda::std::errc::wrong_protocol_type) == EPROTOTYPE);

int main(int, char**)
{
  return 0;
}
