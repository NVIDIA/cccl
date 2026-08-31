//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include <cuda.h>

#include <cccl/c/device_copy.h>
#include <hostjit/codegen/cub_call.hpp>
#include <hostjit/jit_compiler.hpp>
#include <util/build_utils.h>

namespace
{
constexpr const char* device_copy_fn_name = "cccl_jit_device_copy";

using device_copy_fn_t = int (*)(
  const void* source_data,
  unsigned long long source_byte_offset,
  void* destination_data,
  unsigned long long destination_byte_offset,
  unsigned long long num_items,
  void* stream);

bool is_power_of_two(size_t value)
{
  return value != 0 && (value & (value - 1)) == 0;
}

bool effective_address_is_aligned(const void* data, uint64_t byte_offset, size_t alignment)
{
  if (data == nullptr || alignment == 0)
  {
    return false;
  }
  if (alignment == 1)
  {
    return true;
  }

  const auto base = reinterpret_cast<std::uintptr_t>(data);
  if (byte_offset > static_cast<uint64_t>(std::numeric_limits<std::uintptr_t>::max() - base))
  {
    return false;
  }

  const auto effective_address = base + static_cast<std::uintptr_t>(byte_offset);
  return (effective_address % alignment) == 0;
}

bool is_contiguous_layout(cccl_device_copy_layout_kind_t layout)
{
  return layout == CCCL_DEVICE_COPY_LAYOUT_RIGHT || layout == CCCL_DEVICE_COPY_LAYOUT_LEFT;
}

CUresult validate_build_spec(cccl_device_copy_build_spec_t spec)
{
  if (spec.value_type.size == 0 || !is_power_of_two(spec.value_type.alignment))
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if ((spec.value_type.size % spec.value_type.alignment) != 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (spec.rank != 1 || spec.shape == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (spec.shape[0].kind != CCCL_DEVICE_COPY_AXIS_RUNTIME || spec.shape[0].value != 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!is_contiguous_layout(spec.source.layout) || !is_contiguous_layout(spec.destination.layout))
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  return CUDA_SUCCESS;
}

std::string make_device_copy_source(cccl_type_info value_type)
{
  std::string src = R"(#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/__driver/driver_api.h>
#include <cuda/std/mdspan>
#include <cuda/stream_ref>
#include <cub/device/device_copy.cuh>

#if _CCCL_HOSTJIT() && !_CCCL_HOSTED() && !defined(__CUDA_ARCH__) && !defined(_WIN32)
extern "C" void* dlopen(const char*, int);
extern "C" void* dlsym(void*, const char*);
#  ifndef RTLD_NOW
#    define RTLD_NOW 2
#  endif
#endif

static int __cccl_hostjit_init_cuda_driver()
{
  static int status = []() {
#if _CCCL_HOSTJIT() && !_CCCL_HOSTED()
#  if defined(__CUDA_ARCH__)
  return static_cast<int>(cudaSuccess);
#  elif defined(_WIN32)
  return static_cast<int>(cudaErrorNotSupported);
#  else
  if (::cuda::__driver::__getProcAddressFn() != nullptr)
  {
    return static_cast<int>(cudaSuccess);
  }

  static void* driver_library = ::dlopen("libcuda.so.1", RTLD_NOW);
  if (driver_library == nullptr)
  {
    return static_cast<int>(cudaErrorInitializationError);
  }

  static void* get_proc_address = ::dlsym(driver_library, "cuGetProcAddress_v2");
  if (get_proc_address == nullptr)
  {
    return static_cast<int>(cudaErrorInitializationError);
  }

  auto* stored_get_proc_address = ::cuda::__driver::__getProcAddressFn(
    reinterpret_cast<decltype(::cuGetProcAddress)*>(get_proc_address),
    true);
  if (stored_get_proc_address == nullptr)
  {
    return static_cast<int>(cudaErrorInitializationError);
  }
#  endif
#endif

  return static_cast<int>(cudaSuccess);
  }();
  return status;
}

)";

  src += "struct alignas(" + std::to_string(value_type.alignment) + ") cccl_device_copy_value_t\n";
  src += "{\n";
  src += "  char data[" + std::to_string(value_type.size) + "];\n";
  src += "};\n";
  src += "static_assert(sizeof(cccl_device_copy_value_t) == " + std::to_string(value_type.size) + ");\n\n";

  src += R"(extern "C" _CCCL_VISIBILITY_EXPORT int cccl_jit_device_copy(
  const void* source_data,
  unsigned long long source_byte_offset,
  void* destination_data,
  unsigned long long destination_byte_offset,
  unsigned long long num_items,
  void* stream)
{
  const int init_status = __cccl_hostjit_init_cuda_driver();
  if (init_status != static_cast<int>(cudaSuccess))
  {
    return init_status;
  }

  using value_type   = cccl_device_copy_value_t;
  using index_type   = unsigned long long;
  using extents_type = ::cuda::std::dextents<index_type, 1>;
  using input_type   = ::cuda::std::mdspan<const value_type, extents_type>;
  using output_type  = ::cuda::std::mdspan<value_type, extents_type>;

  const auto* source =
    reinterpret_cast<const value_type*>(static_cast<const char*>(source_data) + source_byte_offset);
  auto* destination =
    reinterpret_cast<value_type*>(static_cast<char*>(destination_data) + destination_byte_offset);

  return static_cast<int>(
    cub::DeviceCopy::Copy(
      input_type{source, num_items},
      output_type{destination, num_items},
      ::cuda::stream_ref{reinterpret_cast<cudaStream_t>(stream)}));
}
)";

  return src;
}
} // namespace

CUresult cccl_device_copy_build_ex(
  cccl_device_copy_build_result_t* build_ptr,
  cccl_device_copy_build_spec_t spec,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* build_config)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *build_ptr = {};

  if (CUresult status = validate_build_spec(spec); status != CUDA_SUCCESS)
  {
    return status;
  }

  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(build_config, cub_path, thrust_path);

  auto jit_config = hostjit::codegen::CubCall::make_jit_config(
    cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path, device_copy_fn_name);
  auto source = make_device_copy_source(spec.value_type);

  if (const char* dump_path = std::getenv("CUBCALL_DUMP_SOURCE"))
  {
    std::ofstream f(dump_path);
    f << source;
  }

  auto compiler = std::make_unique<hostjit::JITCompiler>(jit_config);
  if (!compiler->compile(source))
  {
    throw std::runtime_error("DeviceCopy HostJIT compilation failed: " + compiler->getLastError());
  }

  auto fn = compiler->getFunction<device_copy_fn_t>(device_copy_fn_name);
  if (fn == nullptr)
  {
    throw std::runtime_error("DeviceCopy HostJIT function lookup failed: " + compiler->getLastError());
  }

  cccl::detail::copy_cubin(compiler->getCubin(), build_ptr->payload, build_ptr->payload_size);
  build_ptr->cc                 = cc_major * 10 + cc_minor;
  build_ptr->jit_compiler       = compiler.release();
  build_ptr->copy_fn            = reinterpret_cast<void*>(fn);
  build_ptr->value_type         = spec.value_type;
  build_ptr->rank               = spec.rank;
  build_ptr->source_layout      = spec.source.layout;
  build_ptr->destination_layout = spec.destination.layout;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  std::fprintf(stderr, "\nEXCEPTION in cccl_device_copy_build_ex(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_copy_build(
  cccl_device_copy_build_result_t* build_ptr,
  cccl_device_copy_build_spec_t spec,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_copy_build_ex(
    build_ptr, spec, cc_major, cc_minor, cub_path, thrust_path, libcudacxx_path, ctk_path, nullptr);
}

CUresult cccl_device_copy(cccl_device_copy_build_result_t build,
                          cccl_device_copy_source_view_t source,
                          cccl_device_copy_destination_view_t destination,
                          CUstream stream)
try
{
  if (build.copy_fn == nullptr || build.rank != 1 || !is_contiguous_layout(build.source_layout)
      || !is_contiguous_layout(build.destination_layout))
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (source.shape == nullptr || destination.shape == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (source.shape[0] < 0 || destination.shape[0] < 0 || source.shape[0] != destination.shape[0])
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!effective_address_is_aligned(source.data, source.byte_offset, build.value_type.alignment)
      || !effective_address_is_aligned(destination.data, destination.byte_offset, build.value_type.alignment))
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  const auto num_items = static_cast<unsigned long long>(source.shape[0]);
  auto fn              = reinterpret_cast<device_copy_fn_t>(build.copy_fn);
  const int status =
    fn(source.data,
       static_cast<unsigned long long>(source.byte_offset),
       destination.data,
       static_cast<unsigned long long>(destination.byte_offset),
       num_items,
       reinterpret_cast<void*>(stream));

  return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}
catch (const std::exception& exc)
{
  std::fprintf(stderr, "\nEXCEPTION in cccl_device_copy(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_copy_cleanup(cccl_device_copy_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  cccl::detail::release_jit_artifacts(build_ptr);
  build_ptr->copy_fn = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  std::fprintf(stderr, "\nEXCEPTION in cccl_device_copy_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
