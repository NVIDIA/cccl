//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cub/detail/choose_offset.cuh>
#include <cub/grid/grid_even_share.cuh>
#include <cub/util_device.cuh>

#include <format>
#include <memory>
#include <type_traits>
#include <vector>

#include "util/nvjitlink.h"
#include <cccl/c/aot.h>
#include <cccl/c/for.h>
#include <cccl/c/types.h>
#include <for/for_op_helper.h>
#include <nvrtc/command_list.h>
#include <nvrtc/ltoir_list_appender.h>
#include <util/aot_serialize.h>
#include <util/build_utils.h>
#include <util/context.h>
#include <util/errors.h>
#include <util/types.h>

struct op_wrapper;
struct device_reduce_policy;

using OffsetT = unsigned long long;
static_assert(std::is_same_v<cub::detail::choose_offset_t<OffsetT>, OffsetT>, "OffsetT must be size_t");

static cudaError_t
Invoke(cccl_iterator_t d_in, size_t num_items, cccl_op_t op, int /*cc*/, CUfunction static_kernel, CUstream stream)
{
  cudaError error = cudaSuccess;

  if (num_items == 0)
  {
    return error;
  }

  auto for_kernel_state = make_for_kernel_state(op, d_in);

  void* args[] = {&num_items, for_kernel_state.get()};

  const unsigned int thread_count = 256;
  const size_t items_per_block    = 512;
  const size_t block_sz           = cuda::ceil_div(num_items, items_per_block);

  if (block_sz > std::numeric_limits<unsigned int>::max())
  {
    return cudaErrorInvalidValue;
  }
  const unsigned int block_count = static_cast<unsigned int>(block_sz);

  check(cuLaunchKernel(static_kernel, block_count, 1, 1, thread_count, 1, 1, 0, stream, args, 0));

  // Check for failure to launch
  error = CubDebug(cudaPeekAtLastError());

  return error;
}

struct for_each_wrapper;

static std::string get_device_for_kernel_name()
{
  std::string offset_t;
  std::string function_op_t;
  check(cccl_type_name_from_nvrtc<for_each_wrapper>(&function_op_t));
  check(cccl_type_name_from_nvrtc<OffsetT>(&offset_t));

  return std::format(
    "cub::detail::for_each::static_kernel<device_for_policy_selector, {0}, {1}>", offset_t, function_op_t);
}

CUresult cccl_device_for_compile(
  cccl_device_for_build_result_t* build_ptr,
  cccl_iterator_t d_data,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config)
try
{
  if (d_data.type == cccl_iterator_kind_t::CCCL_ITERATOR)
  {
    throw std::runtime_error(std::string("Iterators are unsupported in for_each currently"));
  }

  const char* name = "test";

  const int cc = cc_major * 10 + cc_minor;

  const std::string for_kernel_name   = get_device_for_kernel_name();
  const std::string device_for_kernel = get_for_kernel(op, d_data);

  const std::string arch = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);

  std::vector<const char*> args = {
    arch.c_str(), cub_path, thrust_path, libcudacxx_path, ctk_path, "-rdc=true", "-dlto", "-DCUB_DISABLE_CDP"};

  cccl::detail::extend_args_with_build_config(args, config);

  constexpr size_t num_lto_args   = 2;
  const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

  std::string lowered_name;

  const bool kernel_only = is_custom_op(op);

  auto post_build =
    begin_linking_nvrtc_program(kernel_only ? 0 : num_lto_args, kernel_only ? nullptr : lopts)
      ->add_program(nvrtc_translation_unit{device_for_kernel, name})
      ->add_expression({for_kernel_name})
      ->compile_program({args.data(), args.size()})
      ->get_name({for_kernel_name, lowered_name});

  auto kernel_name = std::unique_ptr<char[]>(duplicate_c_string(lowered_name));

  build_ptr->cc = cc;
  // Zero-init fields set by _load, not _compile.
  build_ptr->library       = nullptr;
  build_ptr->static_kernel = nullptr;

  // All potentially-throwing operations come before any release() calls so that
  // unique_ptrs automatically clean up on exception.
  if (kernel_only)
  {
    auto [ltoir_size, ltoir_data] = post_build->get_program_ltoir();
    build_ptr->payload            = ltoir_data.release();
    build_ptr->payload_size       = ltoir_size;
    build_ptr->payload_kind       = CCCL_PAYLOAD_LTOIR;
  }
  else
  {
    nvrtc_linkable_list linkable_list;
    nvrtc_linkable_list_appender appender{linkable_list};
    appender.append_operation(op);
    if (cccl_iterator_kind_t::CCCL_ITERATOR == d_data.type)
    {
      appender.append_operation(d_data.advance);
      appender.append_operation(d_data.dereference);
    }
    nvrtc_link_result result = post_build->link_program()->add_link_list(linkable_list)->finalize_program();
    build_ptr->payload       = (void*) result.data.release();
    build_ptr->payload_size  = result.size;
    build_ptr->payload_kind  = CCCL_PAYLOAD_CUBIN;
  }

  build_ptr->static_kernel_lowered_name = kernel_name.release();

  return CUDA_SUCCESS;
}
catch (...)
{
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_for_load(cccl_device_for_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr || build_ptr->payload == nullptr || build_ptr->payload_size == 0
      || build_ptr->payload_kind != CCCL_PAYLOAD_CUBIN || build_ptr->static_kernel_lowered_name == nullptr
      || build_ptr->static_kernel_lowered_name[0] == '\0')
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult status =
    cuLibraryLoadData(&build_ptr->library, build_ptr->payload, nullptr, nullptr, 0, nullptr, nullptr, 0);
  if (status != CUDA_SUCCESS)
  {
    return status;
  }
  try
  {
    check(cuLibraryGetKernel(&build_ptr->static_kernel, build_ptr->library, build_ptr->static_kernel_lowered_name));
  }
  catch (...)
  {
    cuLibraryUnload(build_ptr->library);
    build_ptr->library = nullptr;
    throw;
  }
  return CUDA_SUCCESS;
}
catch (...)
{
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_for_link_ltoir(
  cccl_device_for_build_result_t* build_ptr, const void** input_blobs, const size_t* input_sizes, size_t num_inputs)
try
{
  if (build_ptr == nullptr || build_ptr->payload == nullptr || build_ptr->payload_size == 0
      || build_ptr->payload_kind != CCCL_PAYLOAD_LTOIR)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int cc_major = build_ptr->cc / 10;
  const int cc_minor = build_ptr->cc % 10;
  std::vector<const void*> all_blobs;
  std::vector<size_t> all_sizes;
  all_blobs.push_back(build_ptr->payload);
  all_sizes.push_back(build_ptr->payload_size);
  if (num_inputs > 0 && (input_blobs == nullptr || input_sizes == nullptr))
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  for (size_t i = 0; i < num_inputs; ++i)
  {
    if (input_blobs[i] == nullptr || input_sizes[i] == 0)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }
    all_blobs.push_back(input_blobs[i]);
    all_sizes.push_back(input_sizes[i]);
  }
  auto [cubin, cubin_size] = nvjitlink_link(all_blobs.data(), all_sizes.data(), all_blobs.size(), cc_major, cc_minor);
  delete[] static_cast<char*>(build_ptr->payload);
  build_ptr->payload      = (void*) cubin.release();
  build_ptr->payload_size = cubin_size;
  build_ptr->payload_kind = CCCL_PAYLOAD_CUBIN;
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  printf("\nEXCEPTION in cccl_device_for_link_ltoir(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_for_build_ex(
  cccl_device_for_build_result_t* build_ptr,
  cccl_iterator_t d_data,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config)
{
  CUresult r = cccl_device_for_compile(
    build_ptr, d_data, op, cc_major, cc_minor, cub_path, thrust_path, libcudacxx_path, ctk_path, config);
  if (r != CUDA_SUCCESS)
  {
    return r;
  }
  CUresult load_r = cccl_device_for_load(build_ptr);
  if (load_r != CUDA_SUCCESS)
  {
    cccl_device_for_cleanup(build_ptr);
  }
  return load_r;
}

CUresult cccl_device_for(
  cccl_device_for_build_result_t build, cccl_iterator_t d_data, uint64_t num_items, cccl_op_t op, CUstream stream)
{
  bool pushed    = false;
  CUresult error = CUDA_SUCCESS;

  try
  {
    pushed           = try_push_context();
    auto exec_status = Invoke(d_data, num_items, op, build.cc, (CUfunction) build.static_kernel, stream);
    error            = static_cast<CUresult>(exec_status);
  }
  catch (...)
  {
    error = CUDA_ERROR_UNKNOWN;
  }

  if (pushed)
  {
    CUcontext dummy;
    cuCtxPopCurrent(&dummy);
  }

  return error;
}

CUresult cccl_device_for_build(
  cccl_device_for_build_result_t* build,
  cccl_iterator_t d_data,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_for_build_ex(
    build, d_data, op, cc_major, cc_minor, cub_path, thrust_path, libcudacxx_path, ctk_path, nullptr);
}

CUresult cccl_device_for_cleanup(cccl_device_for_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  std::unique_ptr<char[]> payload(reinterpret_cast<char*>(build_ptr->payload));
  std::unique_ptr<char[]> kernel_name(build_ptr->static_kernel_lowered_name);
  if (build_ptr->library != nullptr)
  {
    check(cuLibraryUnload(build_ptr->library));
  }

  return CUDA_SUCCESS;
}
catch (...)
{
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_for_serialize(const cccl_device_for_build_result_t* build_ptr, void** out_buf, size_t* out_size)
try
{
  if (build_ptr == nullptr || out_buf == nullptr || out_size == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (build_ptr->payload == nullptr || build_ptr->payload_size == 0)
  {
    *out_buf  = nullptr;
    *out_size = 0;
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (build_ptr->payload_kind != CCCL_PAYLOAD_LTOIR && build_ptr->payload_kind != CCCL_PAYLOAD_CUBIN)
  {
    *out_buf  = nullptr;
    *out_size = 0;
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (build_ptr->static_kernel_lowered_name == nullptr || build_ptr->static_kernel_lowered_name[0] == '\0')
  {
    *out_buf  = nullptr;
    *out_size = 0;
    return CUDA_ERROR_INVALID_VALUE;
  }

  *out_buf  = nullptr;
  *out_size = 0;

  using namespace cccl::aot;
  buffer_writer w;
  write_header(w, CCCL_AOT_ALGO_FOR, build_ptr->payload_kind, build_ptr->cc);
  w.write_blob(build_ptr->payload, build_ptr->payload_size);
  w.write_cstring(build_ptr->static_kernel_lowered_name);
  w.release(out_buf, out_size);
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_for_serialize(): %s\n", exc.what());
  fflush(stdout);
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_for_deserialize(cccl_device_for_build_result_t* build_ptr, const void* buf, size_t size)
try
{
  if (build_ptr == nullptr || buf == nullptr || size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  using namespace cccl::aot;
  buffer_reader r{buf, size};
  const auto h = read_and_validate_header(r, CCCL_AOT_ALGO_FOR);

  std::unique_ptr<char[]> payload_owner;
  size_t payload_size = 0;
  {
    void* p = nullptr;
    r.read_blob_new(&p, &payload_size);
    payload_owner.reset(static_cast<char*>(p));
  }
  if (payload_size == 0)
  {
    throw std::runtime_error("aot blob: empty payload");
  }

  std::unique_ptr<char[]> n_kernel{r.read_cstring_dup()};
  if (!n_kernel || n_kernel[0] == '\0')
  {
    throw std::runtime_error("aot blob: empty or missing static kernel name");
  }

  // Commit-on-success: populate a zero-initialized local result and assign it only
  // after all reads succeed. This guarantees the load-only fields (library,
  // static_kernel) are null and leaves *build_ptr untouched on any earlier throw.
  cccl_device_for_build_result_t result{};
  result.cc                         = static_cast<int>(h.cc);
  result.payload_kind               = static_cast<cccl_payload_kind_t>(h.payload_kind);
  result.payload                    = payload_owner.release();
  result.payload_size               = payload_size;
  result.static_kernel_lowered_name = n_kernel.release();
  *build_ptr                        = result;
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_for_deserialize(): %s\n", exc.what());
  fflush(stdout);
  return CUDA_ERROR_UNKNOWN;
}
