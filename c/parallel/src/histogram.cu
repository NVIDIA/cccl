//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cub/detail/launcher/cuda_driver.cuh>
#include <cub/device/device_histogram.cuh>

#include <cuda/std/__algorithm_>

#include <format>
#include <vector>

#include "cccl/c/types.h"
#include "cub/util_type.cuh"
#include "kernels/iterators.h"
#include "util/context.h"
#include "util/indirect_arg.h"
#include "util/types.h"
#include <cccl/c/histogram.h>
#include <nvrtc/ltoir_list_appender.h>
#include <util/build_utils.h>

// int32_t is generally faster. Depending on the number of samples we
// instantiate the kernels below with int32 or int64, but we set this to int64
// here because it's needed for host computation as well.
using OffsetT = int64_t;
// Largest type we support for now. Tricky to make this an indirect_arg_t since
// we are passing in cuda::std::arrays holding the values of the levels which
// are used to do host computation.
using LevelT = double;

struct samples_iterator_t;

namespace histogram
{
struct histogram_runtime_tuning_policy
{
  int block_threads;
  int pixels_per_thread;

  int BlockThreads() const
  {
    return block_threads;
  }

  int PixelsPerThread() const
  {
    return pixels_per_thread;
  }
};

template <auto* GetPolicy>
struct dynamic_histogram_policy_t
{
  using MaxPolicy = dynamic_histogram_policy_t;

  template <typename F>
  cudaError_t Invoke(int device_ptx_version, F& op)
  {
    return op.template Invoke<histogram_runtime_tuning_policy>(
      GetPolicy(device_ptx_version, sample_t, num_active_channels));
  }

  cccl_type_info sample_t;
  int num_active_channels;
};

struct histogram_kernel_source
{
  cccl_device_histogram_build_result_t& build;

  CUkernel HistogramInitKernel() const
  {
    return build.init_kernel;
  }

  template <typename PolicyT, int PRIVATIZED_SMEM_BINS, typename PrivatizedDecodeOpT, typename OutputDecodeOpT>
  CUkernel HistogramSweepKernel() const
  {
    return build.sweep_kernel;
  }

  std::size_t CounterSize() const
  {
    return build.counter_type.size;
  }
};

histogram_runtime_tuning_policy get_policy(int /*cc*/, cccl_type_info sample_t, int num_active_channels)
{
  const int v_scale                      = (sample_t.size + sizeof(int) - 1) / sizeof(int);
  constexpr int nominal_items_per_thread = 16;

  int pixels_per_thread = (::cuda::std::max) (nominal_items_per_thread / num_active_channels / v_scale, 1);

  return {384, pixels_per_thread};
}

std::string get_init_kernel_name(int num_active_channels, std::string_view counter_t, std::string_view offset_t)
{
  return std::format(
    "cub::detail::histogram::DeviceHistogramInitKernel<{0}, {1}, {2}>", num_active_channels, counter_t, offset_t);
}

std::string get_sweep_kernel_name(
  std::string_view chained_policy_t,
  int privatized_smem_bins,
  int num_channels,
  int num_active_channels,
  cccl_iterator_t d_samples,
  std::string_view counter_t,
  std::string_view level_t,
  std::string_view offset_t,
  bool is_evenly_segmented,
  bool is_byte_sample)
{
  std::string samples_iterator_name;
  check(nvrtcGetTypeName<samples_iterator_t>(&samples_iterator_name));

  const std::string samples_iterator_t =
    d_samples.type == cccl_iterator_kind_t::CCCL_POINTER //
      ? cccl_type_enum_to_name(d_samples.value_type.type, true) //
      : samples_iterator_name;

  const std::string transforms_t = std::format(
    "cub::detail::histogram::Transforms<{0}, {1}, {2}>",
    level_t,
    offset_t,
    cccl_type_enum_to_name(d_samples.value_type.type));

  std::string privatized_decode_op_t = std::format("{0}::PassThruTransform", transforms_t);
  std::string output_decode_op_t =
    is_evenly_segmented
      ? std::format("{0}::ScaleTransform", transforms_t)
      : std::format("{0}::SearchTransform<const {1}*>", transforms_t, level_t);

  if (!is_byte_sample)
  {
    std::swap(privatized_decode_op_t, output_decode_op_t);
  }

  return std::format(
    "cub::detail::histogram::DeviceHistogramSweepKernel<{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}>",
    chained_policy_t,
    privatized_smem_bins,
    num_channels,
    num_active_channels,
    samples_iterator_t,
    counter_t,
    privatized_decode_op_t,
    output_decode_op_t,
    offset_t);
}

} // namespace histogram

CUresult cccl_device_histogram_build_ex(
  cccl_device_histogram_build_result_t* build_ptr,
  int num_channels,
  int num_active_channels,
  cccl_iterator_t d_samples,
  int num_output_levels_val,
  cccl_iterator_t d_output_histograms,
  cccl_value_t lower_level,
  int64_t num_rows,
  int64_t row_stride_samples,
  bool is_evenly_segmented,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config)
{
  CUresult error = CUDA_SUCCESS;

  try
  {
    const char* name = "test";

    const int cc           = cc_major * 10 + cc_minor;
    const auto policy      = histogram::get_policy(cc, d_samples.value_type, num_active_channels);
    const auto sample_cpp  = cccl_type_enum_to_name(d_samples.value_type.type);
    const auto counter_cpp = cccl_type_enum_to_name(d_output_histograms.value_type.type);
    const auto level_cpp   = cccl_type_enum_to_name(lower_level.type.type);

    const std::string offset_cpp =
      ((unsigned long long) (num_rows * row_stride_samples * d_samples.value_type.size) < (unsigned long long) INT_MAX)
        ? "int"
        : "long long";

    std::string samples_iterator_name;
    check(nvrtcGetTypeName<samples_iterator_t>(&samples_iterator_name));

    const std::string samples_iterator_src =
      make_kernel_input_iterator(offset_cpp, samples_iterator_name, sample_cpp, d_samples);

    constexpr std::string_view chained_policy_t = "device_histogram_policy";

    constexpr std::string_view src_template = R"XXX(
#include <cub/agent/agent_histogram.cuh>
#include <cub/block/block_load.cuh>
#include <cub/device/dispatch/kernels/histogram.cuh>

struct __align__({1}) storage_t {{
  char data[{0}];
}};
{2}
struct agent_policy_t {{
  static constexpr int BLOCK_THREADS = {3};
  static constexpr int PIXELS_PER_THREAD = {4};
  static constexpr bool IS_RLE_COMPRESS = true;
  static constexpr cub::BlockHistogramMemoryPreference MEM_PREFERENCE = cub::SMEM;
  static constexpr bool IS_WORK_STEALING = false;
  static constexpr int VEC_SIZE = 4;
  static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM = cub::BLOCK_LOAD_DIRECT;
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER = cub::LOAD_LDG;
}};
struct {5} {{
  struct ActivePolicy {{
    using AgentHistogramPolicyT = agent_policy_t;
  }};
}};
)XXX";

    const std::string src = std::format(
      src_template,
      d_samples.value_type.size, // 0
      d_samples.value_type.alignment, // 1
      samples_iterator_src, // 2
      policy.block_threads, // 3
      policy.pixels_per_thread, // 4
      chained_policy_t // 5
    );

#if false // CCCL_DEBUGGING_SWITCH
    fflush(stderr);
    printf("\nCODE4NVRTC BEGIN\n%sCODE4NVRTC END\n", src.c_str());
    fflush(stdout);
#endif

    // TODO: This is tricky because we need to know the input to set this to a
    // value greater than 0 (see dispatch_histogram.cuh), but we don't have this
    // information here.
    const int privatized_smem_bins =
      num_output_levels_val - 1 > cub::detail::histogram::Transforms<int, int, int>::MAX_PRIVATIZED_SMEM_BINS ? 0 : 256;

    const bool is_byte_sample = d_samples.value_type.size == 1;

    std::string init_kernel_name  = histogram::get_init_kernel_name(num_active_channels, counter_cpp, offset_cpp);
    std::string sweep_kernel_name = histogram::get_sweep_kernel_name(
      chained_policy_t,
      privatized_smem_bins,
      num_channels,
      num_active_channels,
      d_samples,
      counter_cpp,
      level_cpp,
      offset_cpp,
      is_evenly_segmented,
      is_byte_sample);

    std::string init_kernel_lowered_name;
    std::string sweep_kernel_lowered_name;

    const std::string arch = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);

    std::vector<const char*> args = {
      arch.c_str(), cub_path, thrust_path, libcudacxx_path, ctk_path, "-rdc=true", "-dlto", "-DCUB_DISABLE_CDP"};

    cccl::detail::extend_args_with_build_config(args, config);

    constexpr size_t num_lto_args   = 2;
    const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

    nvrtc_linkable_list linkable_list;
    nvrtc_linkable_list_appender appender{linkable_list};

    appender.add_iterator_definition(d_samples);
    appender.add_iterator_definition(d_output_histograms);

    nvrtc_link_result result =
      begin_linking_nvrtc_program(num_lto_args, lopts)
        ->add_program(nvrtc_translation_unit({src.c_str(), name}))
        ->add_expression({init_kernel_name})
        ->add_expression({sweep_kernel_name})
        ->compile_program({args.data(), args.size()})
        ->get_name({init_kernel_name, init_kernel_lowered_name})
        ->get_name({sweep_kernel_name, sweep_kernel_lowered_name})
        ->link_program()
        ->add_link_list(linkable_list)
        ->finalize_program();

    cuLibraryLoadData(&build_ptr->library, result.data.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
    check(cuLibraryGetKernel(&build_ptr->init_kernel, build_ptr->library, init_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(&build_ptr->sweep_kernel, build_ptr->library, sweep_kernel_lowered_name.c_str()));

    build_ptr->cc                  = cc;
    build_ptr->cubin               = (void*) result.data.release();
    build_ptr->cubin_size          = result.size;
    build_ptr->counter_type        = d_output_histograms.value_type;
    build_ptr->num_active_channels = num_active_channels;
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_histogram_build(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }

  return error;
}

template <typename is_byte_sample>
CUresult cccl_device_histogram_even_impl(
  cccl_device_histogram_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_samples,
  cccl_iterator_t d_output_histograms,
  cccl_value_t num_output_levels,
  cccl_value_t lower_level,
  cccl_value_t upper_level,
  int64_t num_row_pixels,
  int64_t num_rows,
  int64_t row_stride_samples,
  CUstream stream)
{
  if (cccl_iterator_kind_t::CCCL_POINTER != d_output_histograms.type)
  {
    fflush(stderr);
    printf("\nERROR in cccl_device_histogram_even(): histogram parameters must be pointers (except for d_samples)\n ");
    fflush(stdout);
    return CUDA_ERROR_UNKNOWN;
  }

  CUresult error = CUDA_SUCCESS;
  bool pushed    = false;
  try
  {
    pushed = try_push_context();

    CUdevice cu_device;
    check(cuCtxGetDevice(&cu_device));

    constexpr int NUM_CHANNELS        = 1;
    constexpr int NUM_ACTIVE_CHANNELS = 1;

    ::cuda::std::array<indirect_arg_t*, NUM_ACTIVE_CHANNELS> d_output_histogram_arr{
      static_cast<indirect_arg_t*>(d_output_histograms.state)};
    ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_levels_arr{*static_cast<int*>(num_output_levels.state)};
    ::cuda::std::array<LevelT, NUM_ACTIVE_CHANNELS> lower_level_arr{*static_cast<LevelT*>(lower_level.state)};
    ::cuda::std::array<LevelT, NUM_ACTIVE_CHANNELS> upper_level_arr{*static_cast<LevelT*>(upper_level.state)};

    auto exec_status = cub::DispatchHistogram<
      NUM_CHANNELS,
      NUM_ACTIVE_CHANNELS,
      indirect_arg_t, // SampleIteratorT
      indirect_arg_t, // CounterT
      LevelT, // not indirect_arg_t because used on the host
      OffsetT, // OffsetT
      histogram::dynamic_histogram_policy_t<&histogram::get_policy>,
      histogram::histogram_kernel_source,
      cub::detail::CudaDriverLauncherFactory,
      indirect_arg_t,
      cub::detail::histogram::Transforms<LevelT, // LevelT
                                         OffsetT, // OffsetT
                                         LevelT // SampleT
                                         >>::
      DispatchEven(
        d_temp_storage,
        *temp_storage_bytes,
        d_samples,
        d_output_histogram_arr,
        num_output_levels_arr,
        lower_level_arr,
        upper_level_arr,
        num_row_pixels,
        num_rows,
        row_stride_samples,
        stream,
        is_byte_sample{},
        {build},
        cub::detail::CudaDriverLauncherFactory{cu_device, build.cc},
        {d_samples.value_type, build.num_active_channels});

    error = static_cast<CUresult>(exec_status);
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_radix_sort(): %s\n", exc.what());
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

CUresult cccl_device_histogram_even(
  cccl_device_histogram_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_samples,
  cccl_iterator_t d_output_histograms,
  cccl_value_t num_output_levels,
  cccl_value_t lower_level,
  cccl_value_t upper_level,
  int64_t num_row_pixels,
  int64_t num_rows,
  int64_t row_stride_samples,
  CUstream stream)
{
  auto histogram_impl = d_samples.value_type.size == 1 ? cccl_device_histogram_even_impl<::cuda::std::true_type>
                                                       : cccl_device_histogram_even_impl<::cuda::std::false_type>;

  return histogram_impl(
    build,
    d_temp_storage,
    temp_storage_bytes,
    d_samples,
    d_output_histograms,
    num_output_levels,
    lower_level,
    upper_level,
    num_row_pixels,
    num_rows,
    row_stride_samples,
    stream);
}

CUresult cccl_device_histogram_build(
  cccl_device_histogram_build_result_t* build_ptr,
  int num_channels,
  int num_active_channels,
  cccl_iterator_t d_samples,
  int num_output_levels_val,
  cccl_iterator_t d_output_histograms,
  cccl_value_t lower_level,
  int64_t num_rows,
  int64_t row_stride_samples,
  bool is_evenly_segmented,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_histogram_build_ex(
    build_ptr,
    num_channels,
    num_active_channels,
    d_samples,
    num_output_levels_val,
    d_output_histograms,
    lower_level,
    num_rows,
    row_stride_samples,
    is_evenly_segmented,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    nullptr);
}

CUresult cccl_device_histogram_cleanup(cccl_device_histogram_build_result_t* build_ptr)
{
  try
  {
    if (build_ptr == nullptr)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }

    std::unique_ptr<char[]> cubin(reinterpret_cast<char*>(build_ptr->cubin));
    check(cuLibraryUnload(build_ptr->library));
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_histogram_cleanup(): %s\n", exc.what());
    fflush(stdout);
    return CUDA_ERROR_UNKNOWN;
  }

  return CUDA_SUCCESS;
}
