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
#include <cub/detail/ptx-json-parser.cuh>
#include <cub/device/device_histogram.cuh>

#include <cuda/std/algorithm>

#include <format>
#include <limits>
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

struct device_histogram_policy;

// int32_t is generally faster. Depending on the number of samples we
// instantiate the kernels below with int32 or int64, but we set this to int64
// here because it's needed for host computation as well.
using OffsetT = int64_t;

struct samples_iterator_t;

namespace histogram
{
struct histogram_runtime_tuning_policy
{
  cub::detail::RuntimeHistogramAgentPolicy histogram;

  auto Histogram() const
  {
    return histogram;
  }

  CUB_RUNTIME_FUNCTION int BlockThreads() const
  {
    return histogram.BlockThreads();
  }

  CUB_RUNTIME_FUNCTION int PixelsPerThread() const
  {
    return histogram.PixelsPerThread();
  }

  using HistogramPolicy = cub::detail::RuntimeHistogramAgentPolicy;
  using MaxPolicy       = histogram_runtime_tuning_policy;

  template <typename F>
  cudaError_t Invoke(int, F& op)
  {
    return op.template Invoke<histogram_runtime_tuning_policy>(*this);
  }
};

struct histogram_kernel_source
{
  cccl_device_histogram_build_result_t& build;

  template <typename PolicyT>
  CUkernel HistogramInitKernel() const
  {
    return build.init_kernel;
  }

  template <typename PolicyT,
            int PRIVATIZED_SMEM_BINS,
            typename FirstLevelArrayT,
            typename SecondLevelArrayT,
            bool IsEven,
            bool IsByteSample>
  CUkernel HistogramSweepKernelDeviceInit() const
  {
    return build.sweep_kernel;
  }

  std::size_t CounterSize() const
  {
    return build.counter_type.size;
  }

  // Overflow check is performed before type erasure in
  // cccl_device_histogram_even_impl and stored in build.may_overflow. We return
  // this here to have a similar execution path to the CUB implementation.
  template <typename UpperLevelArrayT, typename LowerLevelArrayT>
  bool MayOverflow(
    int /*num_bins*/, const UpperLevelArrayT& /*upper*/, const LowerLevelArrayT& /*lower*/, int /*channel*/) const
  {
    return build.may_overflow;
  }
};

std::string get_init_kernel_name(int num_active_channels, std::string_view counter_t, std::string_view offset_t)
{
  std::string chained_policy_t;
  check(cccl_type_name_from_nvrtc<device_histogram_policy>(&chained_policy_t));

  return std::format(
    "cub::detail::histogram::DeviceHistogramInitKernel<{0}, {1}, {2}, {3}>",
    chained_policy_t,
    num_active_channels,
    counter_t,
    offset_t);
}

std::string get_sweep_kernel_name(
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
  std::string chained_policy_t;
  check(cccl_type_name_from_nvrtc<device_histogram_policy>(&chained_policy_t));

  std::string samples_iterator_name;
  check(cccl_type_name_from_nvrtc<samples_iterator_t>(&samples_iterator_name));

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

  const std::string first_level_array_t =
    is_evenly_segmented
      ? std::format("cuda::std::array<{0}, {1}>", level_t, num_active_channels)
      : std::format("cuda::std::array<int, {0}>", num_active_channels);
  const std::string second_level_array_t =
    is_evenly_segmented
      ? std::format("cuda::std::array<{0}, {1}>", level_t, num_active_channels)
      : std::format("cuda::std::array<const {0}*, {1}>", level_t, num_active_channels);

  return std::format(
    "cub::detail::histogram::DeviceHistogramSweepDeviceInitKernel<{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, "
    "{10}, {11}>",
    chained_policy_t,
    privatized_smem_bins,
    num_channels,
    num_active_channels,
    samples_iterator_t,
    counter_t,
    first_level_array_t,
    second_level_array_t,
    privatized_decode_op_t,
    output_decode_op_t,
    offset_t,
    is_evenly_segmented ? "true" : "false");
}

template <typename T>
uint64_t compute_level_range(const void* lower, const void* upper)
{
  T lower_val = *static_cast<const T*>(lower);
  T upper_val = *static_cast<const T*>(upper);
  return static_cast<uint64_t>(upper_val - lower_val);
}

uint64_t get_integral_range(cccl_type_enum type, const void* lower, const void* upper)
{
  switch (type)
  {
    case CCCL_INT8:
      return compute_level_range<int8_t>(lower, upper);
    case CCCL_UINT8:
      return compute_level_range<uint8_t>(lower, upper);
    case CCCL_INT16:
      return compute_level_range<int16_t>(lower, upper);
    case CCCL_UINT16:
      return compute_level_range<uint16_t>(lower, upper);
    case CCCL_INT32:
      return compute_level_range<int32_t>(lower, upper);
    case CCCL_UINT32:
      return compute_level_range<uint32_t>(lower, upper);
    case CCCL_INT64:
      return compute_level_range<int64_t>(lower, upper);
    case CCCL_UINT64:
      return compute_level_range<uint64_t>(lower, upper);
    default:
      throw std::runtime_error("get_integral_range: unsupported type");
  }
}

// Check for overflow before type erasure, using actual integer values
// Returns true if overflow may occur
bool check_histogram_overflow(
  const cccl_device_histogram_build_result_t& build,
  int num_bins,
  const cccl_value_t& lower_level,
  const cccl_value_t& upper_level)
{
  auto is_fp = [](cccl_type_enum t) {
    return t == CCCL_FLOAT16 || t == CCCL_FLOAT32 || t == CCCL_FLOAT64;
  };

  if (is_fp(build.level_type.type) || is_fp(build.sample_type.type))
  {
    return false;
  }

  uint64_t range = get_integral_range(build.level_type.type, lower_level.state, upper_level.state);

  // TODO: revisit this when we add support for int128.
  // Mirror IntArithmeticT selection logic:
  // If sizeof(SampleT) + sizeof(CommonT) <= 4, use 32-bit, else 64-bit
  // CommonT size ≈ max(level_size, sample_size) for integral types
  size_t sample_size = build.sample_type.size;
  size_t level_size  = build.level_type.size;
  size_t common_size = (sample_size > level_size) ? sample_size : level_size;

  if (sample_size + common_size <= 4)
  {
    return range > (std::numeric_limits<uint32_t>::max() / static_cast<uint64_t>(num_bins));
  }
  else
  {
    return range > (std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(num_bins));
  }
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
    const auto sample_cpp  = cccl_type_enum_to_name(d_samples.value_type.type);
    const auto counter_cpp = cccl_type_enum_to_name(d_output_histograms.value_type.type);
    const auto level_cpp   = cccl_type_enum_to_name(lower_level.type.type);

    const std::string offset_cpp =
      ((unsigned long long) (num_rows * row_stride_samples * d_samples.value_type.size) < (unsigned long long) INT_MAX)
        ? "int"
        : "long long";

    std::string samples_iterator_name;
    check(cccl_type_name_from_nvrtc<samples_iterator_t>(&samples_iterator_name));

    const std::string samples_iterator_src =
      make_kernel_input_iterator(offset_cpp, samples_iterator_name, sample_cpp, d_samples);

    std::string policy_hub_expr = std::format(
      "cub::detail::histogram::policy_hub<{}, {}, {}, {}, {}>",
      sample_cpp,
      counter_cpp,
      num_channels,
      num_active_channels,
      is_evenly_segmented ? "true" : "false");

    std::string final_src = std::format(
      R"XXX(
#include <cub/agent/agent_histogram.cuh>
#include <cub/block/block_load.cuh>
#include <cub/device/dispatch/kernels/kernel_histogram.cuh>
#include <cub/device/dispatch/tuning/tuning_histogram.cuh>

struct __align__({1}) storage_t {{
  char data[{0}];
}};
{2}
using device_histogram_policy = {3}::MaxPolicy;

#include <cub/detail/ptx-json/json.cuh>
__device__ consteval auto& policy_generator() {{
  return ptx_json::id<ptx_json::string("device_histogram_policy")>()
    = cub::detail::histogram::HistogramPolicyWrapper<device_histogram_policy::ActivePolicy>::EncodedPolicy();
}}
)XXX",
      d_samples.value_type.size, // 0
      d_samples.value_type.alignment, // 1
      samples_iterator_src, // 2
      policy_hub_expr // 3
    );

#if false // CCCL_DEBUGGING_SWITCH
    fflush(stderr);
    printf("\nCODE4NVRTC BEGIN\n%sCODE4NVRTC END\n", final_src.c_str());
    fflush(stdout);
#endif

    // TODO: This is tricky because we need to know the input to set this to a
    // value greater than 0 (see dispatch_histogram.cuh), but we don't have this
    // information here.
    const int privatized_smem_bins =
      num_output_levels_val - 1 > cub::detail::histogram::max_privatized_smem_bins ? 0 : 256;

    const bool is_byte_sample = d_samples.value_type.size == 1;

    std::string init_kernel_name  = histogram::get_init_kernel_name(num_active_channels, counter_cpp, offset_cpp);
    std::string sweep_kernel_name = histogram::get_sweep_kernel_name(
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

    // Note: `-default-device` is needed because of the constexpr functions in
    // tuning_histogram.cuh
    std::vector<const char*> args = {
      arch.c_str(),
      cub_path,
      thrust_path,
      libcudacxx_path,
      ctk_path,
      "-rdc=true",
      "-dlto",
      "-default-device",
      "-DCUB_DISABLE_CDP",
      "-DCUB_ENABLE_POLICY_PTX_JSON",
      "-std=c++20"};

    cccl::detail::extend_args_with_build_config(args, config);

    constexpr size_t num_lto_args   = 2;
    const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

    nvrtc_linkable_list linkable_list;
    nvrtc_linkable_list_appender appender{linkable_list};

    appender.add_iterator_definition(d_samples);
    appender.add_iterator_definition(d_output_histograms);

    nvrtc_link_result result =
      begin_linking_nvrtc_program(num_lto_args, lopts)
        ->add_program(nvrtc_translation_unit({final_src.c_str(), name}))
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

    nlohmann::json runtime_policy =
      cub::detail::ptx_json::parse("device_histogram_policy", {result.data.get(), result.size});

    using cub::detail::RuntimeHistogramAgentPolicy;
    auto histogram_policy = RuntimeHistogramAgentPolicy::from_json(runtime_policy, "HistogramPolicy");

    build_ptr->cc                  = cc;
    build_ptr->cubin               = (void*) result.data.release();
    build_ptr->cubin_size          = result.size;
    build_ptr->counter_type        = d_output_histograms.value_type;
    build_ptr->level_type          = lower_level.type;
    build_ptr->sample_type         = d_samples.value_type;
    build_ptr->num_active_channels = num_active_channels;
    build_ptr->may_overflow = false; // This is set in cccl_device_histogram_even_impl so that kernel source can access
                                     // it later.
    build_ptr->runtime_policy = new histogram::histogram_runtime_tuning_policy{histogram_policy};
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

    // Check for overflow before type erasure (while we still have access to actual types)
    int num_bins       = *static_cast<int*>(num_output_levels.state) - 1;
    build.may_overflow = histogram::check_histogram_overflow(build, num_bins, lower_level, upper_level);

    ::cuda::std::array<indirect_arg_t*, NUM_ACTIVE_CHANNELS> d_output_histogram_arr{
      static_cast<indirect_arg_t*>(d_output_histograms.state)};
    ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_levels_arr{*static_cast<int*>(num_output_levels.state)};
    indirect_arg_t upper_level_arg{upper_level};
    indirect_arg_t lower_level_arg{lower_level};

    auto exec_status = cub::DispatchHistogram<
      NUM_CHANNELS,
      NUM_ACTIVE_CHANNELS,
      indirect_arg_t, // SampleIteratorT
      indirect_arg_t, // CounterT
      indirect_arg_t, // LevelT
      OffsetT, // OffsetT
      histogram::histogram_runtime_tuning_policy, // PolicyHub
      indirect_arg_t, // SampleT
      histogram::histogram_kernel_source, // KernelSource
      cub::detail::CudaDriverLauncherFactory // KernelLauncherFactory
      >::
      __dispatch_even_device_init(
        d_temp_storage,
        *temp_storage_bytes,
        d_samples,
        d_output_histogram_arr,
        num_output_levels_arr,
        lower_level_arg,
        upper_level_arg,
        num_row_pixels,
        num_rows,
        row_stride_samples,
        stream,
        is_byte_sample{},
        {build},
        cub::detail::CudaDriverLauncherFactory{cu_device, build.cc},
        *reinterpret_cast<histogram::histogram_runtime_tuning_policy*>(build.runtime_policy));

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
    std::unique_ptr<char[]> policy(reinterpret_cast<char*>(build_ptr->runtime_policy));
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
