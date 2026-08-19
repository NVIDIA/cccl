//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef LIBNVCC_EXCEPTION_FILTER_HPP
#define LIBNVCC_EXCEPTION_FILTER_HPP

namespace libnvcc
{
//! Restores the process-wide unhandled-exception filter across a libnvcc call.
//!
//! The first time any llvm::sys signal API runs (sys::RemoveFileOnSignal,
//! sys::PrintStackTraceOnErrorSignal, ...) LLVM installs its own top-level
//! filter via SetUnhandledExceptionFilter, and llvm::sys::unregisterHandlers()
//! is an empty function on Windows, so it is never taken back down.
//!
//! LLVM's filter does not chain to the filter it replaced: it writes a minidump,
//! prints an LLVM stack trace and returns EXCEPTION_EXECUTE_HANDLER, which ends
//! the process. Embedding libnvcc therefore silently and permanently replaces
//! the host application's crash reporting from the first build onwards.
//!
//! Scope the damage to the call by putting back whatever the host had installed.
//! Nesting-safe: only the outermost guard restores.
class UnhandledExceptionFilterGuard
{
public:
  UnhandledExceptionFilterGuard();
  ~UnhandledExceptionFilterGuard();

  UnhandledExceptionFilterGuard(const UnhandledExceptionFilterGuard&)            = delete;
  UnhandledExceptionFilterGuard& operator=(const UnhandledExceptionFilterGuard&) = delete;
};
} // namespace libnvcc

#endif // LIBNVCC_EXCEPTION_FILTER_HPP
