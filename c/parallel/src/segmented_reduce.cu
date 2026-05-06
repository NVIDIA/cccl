//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cub/detail/choose_offset.cuh> // cub::detail::choose_offset_t
#include <cub/detail/launcher/cuda_driver.cuh> // cub::detail::CudaDriverLauncherFactory
#include <cub/device/dispatch/dispatch_segmented_reduce.cuh> // cub::DispatchSegmentedReduce
#include <cub/thread/thread_load.cuh> // cub::LoadModifier

#include <exception> // std::exception
#include <format>
#include <mutex>
#include <string> // std::string
#include <string_view> // std::string_view
#include <type_traits> // std::is_same_v
#include <vector> // std::format

#include <stdio.h> // printf

#include "jit_templates/templates/input_iterator.h"
#include "jit_templates/templates/operation.h"
#include "jit_templates/templates/output_iterator.h"
#include "jit_templates/traits.h"
#include "util/nvjitlink.h"
#include <cccl/c/segmented_reduce.h>
#include <cccl/c/types.h> // cccl_type_info
#include <nvrtc/command_list.h>
#include <nvrtc/ltoir_list_appender.h>
#include <util/build_utils.h>
#include <util/context.h>
#include <util/errors.h>
#include <util/indirect_arg.h>
#include <util/types.h>

struct device_segmented_reduce_policy;
using OffsetT = unsigned long long;
static_assert(std::is_same_v<cub::detail::choose_offset_t<OffsetT>, OffsetT>, "OffsetT must be size_t");

// check we can map OffsetT to ::cuda::std::uint64_t
static_assert(std::is_unsigned_v<OffsetT>);
static_assert(sizeof(OffsetT) == sizeof(::cuda::std::uint64_t));

namespace segmented_reduce
{
static cccl_type_info get_accumulator_type(cccl_op_t /*op*/, cccl_iterator_t /*input_it*/, cccl_value_t init)
{
  // TODO Should be decltype(op(init, *input_it)) but haven't implemented type arithmetic yet
  //      so switching back to the old accumulator type logic for now
  return init.type;
}

std::string get_device_segmented_reduce_kernel_name(
  std::string_view reduction_op_t,
  std::string_view input_iterator_t,
  std::string_view output_iterator_t,
  std::string_view start_offset_iterator_t,
  std::string_view end_offset_iterator_t,
  cccl_value_t init,
  std::string_view accum_t)
{
  std::string policy_selector_t;
  check(cccl_type_name_from_nvrtc<device_segmented_reduce_policy>(&policy_selector_t));

  std::string offset_t;
  check(cccl_type_name_from_nvrtc<OffsetT>(&offset_t));

  const std::string init_t = cccl_type_enum_to_name(init.type.type);

  /*
  template <typename PolicySelector,       // 0
            typename InputIteratorT,       // 1
            typename OutputIteratorT,      // 2
            typename BeginOffsetIteratorT, // 3
            typename EndOffsetIteratorT,   // 4
            typename OffsetT,              // 5
            typename ReductionOpT,         // 6
            typename InitT,                // 7
            typename AccumT>               // 8
   DeviceSegmentedReduceKernel(...);
  */
  return std::format(
    "cub::detail::segmented_reduce::DeviceSegmentedReduceKernel<{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}>",
    policy_selector_t, // 0
    input_iterator_t, // 1
    output_iterator_t, // 2
    start_offset_iterator_t, // 3
    end_offset_iterator_t, // 4
    offset_t, // 5
    reduction_op_t, // 6
    init_t, // 7
    accum_t); // 8
}

struct segmented_reduce_kernel_source
{
  cccl_device_segmented_reduce_build_result_t& build;

  CUkernel SegmentedReduceKernel() const
  {
    return build.segmented_reduce_kernel;
  }
};
} // namespace segmented_reduce

struct segmented_reduce_input_iterator_tag;
struct segmented_reduce_output_iterator_tag;
struct segmented_reduce_start_offset_iterator_tag;
struct segmented_reduce_end_offset_iterator_tag;
struct segmented_reduce_operation_tag;

CUresult cccl_device_segmented_reduce_compile(
  cccl_device_segmented_reduce_build_result_t* build_ptr,
  cccl_iterator_t input_it,
  cccl_iterator_t output_it,
  cccl_iterator_t start_offset_it,
  cccl_iterator_t end_offset_it,
  cccl_op_t op,
  cccl_value_t init,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config)
try
{
  const char* name = "device_segmented_reduce";

  const cccl_type_info accum_t = segmented_reduce::get_accumulator_type(op, input_it, init);
  const auto accum_cpp         = cccl_type_enum_to_name(accum_t.type);

  const auto [input_iterator_name, input_iterator_src] =
    get_specialization<segmented_reduce_input_iterator_tag>(template_id<input_iterator_traits>(), input_it);

  const auto [output_iterator_name, output_iterator_src] =
    get_specialization<segmented_reduce_output_iterator_tag>(template_id<output_iterator_traits>(), output_it, accum_t);

  const auto [start_offset_iterator_name, start_offset_iterator_src] =
    get_specialization<segmented_reduce_start_offset_iterator_tag>(
      template_id<input_iterator_traits>(), start_offset_it);

  const auto [end_offset_iterator_name, end_offset_iterator_src] =
    get_specialization<segmented_reduce_end_offset_iterator_tag>(template_id<input_iterator_traits>(), end_offset_it);

  const auto [op_name, op_src] = get_specialization<segmented_reduce_operation_tag>(
    template_id<binary_user_operation_traits>(), op, accum_t, accum_t, accum_t);

  // OffsetT is checked to match have 64-bit size
  const auto offset_t = cccl_type_enum_to_name(cccl_type_enum::CCCL_UINT64);

  const auto policy_sel = [&] {
    using namespace cub::detail;

    const auto accum_type  = cccl_type_enum_to_cub_type(accum_t.type);
    const auto operation_t = cccl_op_kind_to_cub_op(op.type);

    const int offset_size = int{sizeof(OffsetT)};
    return cub::detail::segmented_reduce::policy_selector{
      accum_type, operation_t, offset_size, static_cast<int>(accum_t.size)};
  }();

  // TODO(bgruber): drop this if tuning policies become formattable
  std::stringstream policy_sel_str;
  policy_sel_str << policy_sel(cuda::compute_capability{cc_major, cc_minor});

  const auto policy_sel_expr =
    std::format("cub::detail::segmented_reduce::policy_selector_from_types<{}, {}, {}>", accum_cpp, offset_t, op_name);

  const auto final_src = std::format(
    R"XXX(
#include <cub/block/block_reduce.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_reduce.cuh>
#include <cub/device/dispatch/kernels/kernel_segmented_reduce.cuh>
{0}
struct __align__({2}) storage_t {{
  char data[{1}];
}};
{3}
{4}
{5}
{6}
{7}
using device_segmented_reduce_policy = {8};
using namespace cub;
using namespace cub::detail::reduce;
using namespace cub::detail::segmented_reduce;
static_assert(
  device_segmented_reduce_policy()(detail::current_tuning_cc()) == {9},
  "Host generated and JIT compiled policy mismatch");
)XXX",
    jit_template_header_contents, // 0
    input_it.value_type.size, // 1
    input_it.value_type.alignment, // 2
    input_iterator_src, // 3
    output_iterator_src, // 4
    op_src, // 5
    start_offset_iterator_src, // 6
    end_offset_iterator_src, // 7
    policy_sel_expr, // 8
    policy_sel_str.view()); // 9

#if false // CCCL_DEBUGGING_SWITCH
  fflush(stderr);
  printf("\nCODE4NVRTC BEGIN\n%sCODE4NVRTC END\n", final_src.c_str());
  fflush(stdout);
#endif

  std::string segmented_reduce_kernel_name = segmented_reduce::get_device_segmented_reduce_kernel_name(
    op_name,
    input_iterator_name,
    output_iterator_name,
    start_offset_iterator_name,
    end_offset_iterator_name,
    init,
    accum_cpp);
  std::string segmented_reduce_kernel_lowered_name;

  const std::string arch = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);

  std::vector<const char*> args = {
    arch.c_str(),
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    "-rdc=true",
    "-dlto",
    "-DCUB_DISABLE_CDP",
    "-std=c++20",
    "-default-device"};

  cccl::detail::extend_args_with_build_config(args, config);

  constexpr size_t num_lto_args   = 2;
  const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

  const bool kernel_only = (op.code_size == 0) && (op.name != nullptr) && (op.name[0] != '\0');

  // Collect all LTO-IRs to be linked (empty when op.code_size == 0 — kernel-only mode).
  nvrtc_linkable_list linkable_list;
  nvrtc_linkable_list_appender appender{linkable_list};

  appender.append_operation(op);
  appender.add_iterator_definition(input_it);
  appender.add_iterator_definition(output_it);
  appender.add_iterator_definition(start_offset_it);
  appender.add_iterator_definition(end_offset_it);

  auto post_build =
    begin_linking_nvrtc_program(kernel_only ? 0 : num_lto_args, kernel_only ? nullptr : lopts)
      ->add_program(nvrtc_translation_unit{final_src.c_str(), name})
      ->add_expression({segmented_reduce_kernel_name})
      ->compile_program({args.data(), args.size()})
      ->get_name({segmented_reduce_kernel_name, segmented_reduce_kernel_lowered_name});

  auto policy      = std::make_unique<cub::detail::segmented_reduce::policy_selector>(policy_sel);
  auto kernel_name = std::unique_ptr<char[]>(duplicate_c_string(segmented_reduce_kernel_lowered_name));

  build_ptr->cc                                   = cc_major * 10 + cc_minor;
  build_ptr->accumulator_size                     = accum_t.size;
  build_ptr->runtime_policy                       = policy.release();
  build_ptr->runtime_policy_size                  = sizeof(cub::detail::segmented_reduce::policy_selector);
  build_ptr->segmented_reduce_kernel_lowered_name = kernel_name.release();

  if (kernel_only)
  {
    auto [ltoir_size, ltoir_data] = post_build->get_program_ltoir();
    build_ptr->kernel_ltoir       = ltoir_data.release();
    build_ptr->kernel_ltoir_size  = ltoir_size;
  }
  else
  {
    nvrtc_link_result result = post_build->link_program()->add_link_list(linkable_list)->finalize_program();
    build_ptr->cubin         = (void*) result.data.release();
    build_ptr->cubin_size    = result.size;
  }

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_segmented_reduce_compile(): %s\n", exc.what());
  fflush(stdout);

  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_segmented_reduce_load(cccl_device_segmented_reduce_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr || build_ptr->cubin == nullptr || build_ptr->cubin_size == 0
      || build_ptr->segmented_reduce_kernel_lowered_name == nullptr
      || build_ptr->segmented_reduce_kernel_lowered_name[0] == '\0')
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  check(cuLibraryLoadData(&build_ptr->library, build_ptr->cubin, nullptr, nullptr, 0, nullptr, nullptr, 0));
  try
  {
    check(cuLibraryGetKernel(
      &build_ptr->segmented_reduce_kernel, build_ptr->library, build_ptr->segmented_reduce_kernel_lowered_name));
  }
  catch (...)
  {
    cuLibraryUnload(build_ptr->library);
    build_ptr->library = nullptr;
    throw;
  }
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_segmented_reduce_load(): %s\n", exc.what());
  fflush(stdout);
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_segmented_reduce_build_ex(
  cccl_device_segmented_reduce_build_result_t* build_ptr,
  cccl_iterator_t input_it,
  cccl_iterator_t output_it,
  cccl_iterator_t start_offset_it,
  cccl_iterator_t end_offset_it,
  cccl_op_t op,
  cccl_value_t init,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config)
{
  if (build_ptr->kernel_ltoir != nullptr && build_ptr->kernel_ltoir_size > 0)
  {
    nvrtc_linkable_list linkable_list;
    nvrtc_linkable_list_appender appender{linkable_list};
    appender.append_operation(op);
    appender.add_iterator_definition(input_it);
    appender.add_iterator_definition(output_it);
    appender.add_iterator_definition(start_offset_it);
    appender.add_iterator_definition(end_offset_it);
    std::vector<const void*> blobs;
    std::vector<size_t> sizes;
    for (const auto& item : linkable_list)
    {
      if (std::holds_alternative<nvrtc_ltoir>(item))
      {
        const auto& l = std::get<nvrtc_ltoir>(item);
        blobs.push_back(l.ltoir);
        sizes.push_back(l.size);
      }
    }
    CUresult r = cccl_device_segmented_reduce_link_ltoir(build_ptr, blobs.data(), sizes.data(), blobs.size());
    if (r != CUDA_SUCCESS)
    {
      return r;
    }
    return cccl_device_segmented_reduce_load(build_ptr);
  }
  CUresult r = cccl_device_segmented_reduce_compile(
    build_ptr,
    input_it,
    output_it,
    start_offset_it,
    end_offset_it,
    op,
    init,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    config);
  if (r != CUDA_SUCCESS)
  {
    return r;
  }
  return cccl_device_segmented_reduce_load(build_ptr);
}

CUresult cccl_device_segmented_reduce(
  cccl_device_segmented_reduce_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_segments,
  cccl_iterator_t start_offset,
  cccl_iterator_t end_offset,
  cccl_op_t op,
  cccl_value_t init,
  size_t max_segment_size,
  CUstream stream)
{
  bool pushed    = false;
  CUresult error = CUDA_SUCCESS;
  try
  {
    pushed = try_push_context();

    CUdevice cu_device;
    check(cuCtxGetDevice(&cu_device));

    auto exec_status = cub::detail::segmented_reduce::dispatch</* OverrideAccumT */ void, OffsetT>(
      d_temp_storage,
      *temp_storage_bytes,
      indirect_arg_t{d_in},
      indirect_iterator_t{d_out},
      num_segments,
      indirect_iterator_t{start_offset},
      indirect_iterator_t{end_offset},
      indirect_arg_t{op},
      indirect_arg_t{init},
      max_segment_size,
      stream,
      *static_cast<cub::detail::segmented_reduce::policy_selector*>(build.runtime_policy),
      segmented_reduce::segmented_reduce_kernel_source{build},
      cub::detail::CudaDriverLauncherFactory{cu_device, build.cc});

    error = static_cast<CUresult>(exec_status);
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_segmented_reduce(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }

  if (pushed)
  {
    CUcontext dummy;
    cuCtxPopCurrent(&dummy);
  }

  return error;
}

CUresult cccl_device_segmented_reduce_build(
  cccl_device_segmented_reduce_build_result_t* build,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  cccl_iterator_t begin_offset_in,
  cccl_iterator_t end_offset_in,
  cccl_op_t op,
  cccl_value_t init,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_segmented_reduce_build_ex(
    build,
    d_in,
    d_out,
    begin_offset_in,
    end_offset_in,
    op,
    init,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    nullptr);
}

CUresult cccl_device_segmented_reduce_cleanup(cccl_device_segmented_reduce_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  std::unique_ptr<char[]> cubin(reinterpret_cast<char*>(build_ptr->cubin));
  std::unique_ptr<char[]> kernel_ltoir(static_cast<char*>(build_ptr->kernel_ltoir));
  std::unique_ptr<cub::detail::segmented_reduce::policy_selector> policy(
    static_cast<cub::detail::segmented_reduce::policy_selector*>(build_ptr->runtime_policy));
  std::unique_ptr<char[]> kernel_name(build_ptr->segmented_reduce_kernel_lowered_name);
  if (build_ptr->library != nullptr)
  {
    check(cuLibraryUnload(build_ptr->library));
  }

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_segmented_reduce_cleanup(): %s\n", exc.what());
  fflush(stdout);

  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_segmented_reduce_link_ltoir(
  cccl_device_segmented_reduce_build_result_t* build_ptr,
  const void** input_blobs,
  const size_t* input_sizes,
  size_t num_inputs)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int cc_major = build_ptr->cc / 10;
  const int cc_minor = build_ptr->cc % 10;
  std::vector<const void*> all_blobs;
  std::vector<size_t> all_sizes;
  if (build_ptr->kernel_ltoir != nullptr && build_ptr->kernel_ltoir_size > 0)
  {
    all_blobs.push_back(build_ptr->kernel_ltoir);
    all_sizes.push_back(build_ptr->kernel_ltoir_size);
  }
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
  delete[] static_cast<char*>(build_ptr->kernel_ltoir);
  build_ptr->kernel_ltoir      = nullptr;
  build_ptr->kernel_ltoir_size = 0;
  build_ptr->cubin             = (void*) cubin.release();
  build_ptr->cubin_size        = cubin_size;
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  printf("\nEXCEPTION in cccl_device_segmented_reduce_link_ltoir(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
