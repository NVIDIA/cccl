//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// libnvccCompileProgramToHostIR: the host translation unit of a CUB call as
// LLVM IR. The textual form must show the launch surface a consumer rewrites
// (kernel stubs calling cudaLaunchKernel after __cudaPushCallConfiguration),
// the fatbin registration, and the exported entry; the bitcode form must be a
// bitcode file; both must still produce the cubin.

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <hostjit/config.hpp>
#include <libnvcc/libnvcc.h>

static const char* k_source = R"(
#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>

extern "C" _CCCL_VISIBILITY_EXPORT cudaError_t
sum_f32(void* d_temp, size_t* temp_bytes, const float* d_in, float* d_out, int num_items, cudaStream_t stream)
{
  return cub::DeviceReduce::Sum(d_temp, *temp_bytes, d_in, d_out, num_items, stream);
}
)";

static bool read_text(const std::string& path, std::string& out)
{
  std::ifstream in(path, std::ios::binary);
  if (!in)
  {
    return false;
  }
  out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  return true;
}

static bool expect_contains(const std::string& text, const char* needle)
{
  if (text.find(needle) == std::string::npos)
  {
    std::fprintf(stderr, "host IR does not mention '%s'\n", needle);
    return false;
  }
  return true;
}

int main()
{
  auto config = hostjit::detectDefaultConfig();
  std::vector<std::string> options;
  config.appendCommandLineArguments(options);
  std::vector<const char*> option_ptrs;
  for (const auto& option : options)
  {
    option_ptrs.push_back(option.c_str());
  }

  auto temp_dir = std::filesystem::temp_directory_path() / "hostjit_host_ir_test";
  std::filesystem::create_directories(temp_dir);
  const std::string ll_path    = (temp_dir / "host.ll").string();
  const std::string bc_path    = (temp_dir / "host.bc").string();
  const std::string cubin_path = (temp_dir / "device.cubin").string();

  libnvccProgram prog = nullptr;
  if (libnvccCreateProgram(&prog, k_source, "sum.cu") != LIBNVCC_SUCCESS)
  {
    std::fprintf(stderr, "libnvccCreateProgram failed\n");
    return 1;
  }

  auto report = [&](const char* step) {
    size_t log_size = 0;
    libnvccGetProgramLogSize(prog, &log_size);
    std::string log(log_size, '\0');
    libnvccGetProgramLog(prog, log.data());
    std::fprintf(stderr, "%s failed:\n%s\n", step, log.c_str());
  };

  // Textual IR: inspect the launch surface.
  if (libnvccCompileProgramToHostIR(
        prog, ll_path.c_str(), cubin_path.c_str(), static_cast<int>(option_ptrs.size()), option_ptrs.data())
      != LIBNVCC_SUCCESS)
  {
    report("libnvccCompileProgramToHostIR (.ll)");
    return 1;
  }
  std::string ir;
  if (!read_text(ll_path, ir) || ir.empty())
  {
    std::fprintf(stderr, "no textual IR written to %s\n", ll_path.c_str());
    return 1;
  }
  bool ok = true;
  ok &= expect_contains(ir, "define ");
  ok &= expect_contains(ir, "@sum_f32");
  ok &= expect_contains(ir, "__cudaPushCallConfiguration");
  ok &= expect_contains(ir, "cudaLaunchKernel");
  ok &= expect_contains(ir, "__cudaRegisterFatBinary");
  if (!ok)
  {
    return 1;
  }
  if (!std::filesystem::exists(cubin_path) || std::filesystem::file_size(cubin_path) == 0)
  {
    std::fprintf(stderr, "cubin not written alongside the host IR\n");
    return 1;
  }

  // Bitcode: a bitcode file (magic 'BC' 0xC0 0xDE).
  if (libnvccCompileProgramToHostIR(
        prog, bc_path.c_str(), nullptr, static_cast<int>(option_ptrs.size()), option_ptrs.data())
      != LIBNVCC_SUCCESS)
  {
    report("libnvccCompileProgramToHostIR (.bc)");
    return 1;
  }
  std::string bc;
  if (!read_text(bc_path, bc) || bc.size() < 4 || std::memcmp(bc.data(), "BC\xC0\xDE", 4) != 0)
  {
    std::fprintf(stderr, "no bitcode written to %s\n", bc_path.c_str());
    return 1;
  }

  libnvccDestroyProgram(&prog);
  std::filesystem::remove_all(temp_dir);
  std::printf("host IR output: %zu bytes of textual IR, %zu bytes of bitcode\n", ir.size(), bc.size());
  return 0;
}
