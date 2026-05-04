//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <format>
#include <memory>
#include <utility>

#include <nvrtc/nvjitlink_helper.h>
#include <util/errors.h>

// Links LTO-IR blobs via nvJitLink → returns SASS cubin (PTX fallback if SASS unavailable).
// Caller owns the returned buffer.
inline std::pair<std::unique_ptr<char[]>, size_t>
nvjitlink_link(const void** blobs, const size_t* sizes, size_t num, int cc_major, int cc_minor)
{
  const std::string arch = std::format("-arch=sm_{}{}", cc_major, cc_minor);
  const char* lopts[]    = {"-lto", arch.c_str()};

  nvJitLinkHandle h{};
  check(nvJitLinkCreate(&h, 2, lopts));

  auto cleanup = [&]() {
    if (h)
    {
      nvJitLinkDestroy(&h);
      h = nullptr;
    }
  };

  try
  {
    for (size_t i = 0; i < num; ++i)
    {
      if (blobs[i] && sizes[i] > 0)
      {
        check(nvJitLinkAddData(h, NVJITLINK_INPUT_ANY, blobs[i], sizes[i], "aot_input"));
      }
    }

    auto rc = nvJitLinkComplete(h);

    size_t log_size = 0;
    check(nvJitLinkGetErrorLogSize(h, &log_size));
    if (log_size > 1)
    {
      auto log = std::make_unique<char[]>(log_size);
      check(nvJitLinkGetErrorLog(h, log.get()));
      fprintf(stderr, "%s\n", log.get());
    }
    check(rc);

    size_t cubin_size = 0;
    bool use_ptx      = (nvJitLinkGetLinkedCubinSize(h, &cubin_size) != NVJITLINK_SUCCESS);
    if (use_ptx)
    {
      check(nvJitLinkGetLinkedPtxSize(h, &cubin_size));
    }
    auto cubin = std::make_unique<char[]>(cubin_size);
    if (use_ptx)
    {
      check(nvJitLinkGetLinkedPtx(h, cubin.get()));
    }
    else
    {
      check(nvJitLinkGetLinkedCubin(h, cubin.get()));
    }

    cleanup();
    return {std::move(cubin), cubin_size};
  }
  catch (...)
  {
    cleanup();
    throw;
  }
}
