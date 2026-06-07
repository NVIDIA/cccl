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

#include <cuda/__type_traits/is_trivially_copyable.h>

#include <cstdlib>
#include <cstring>
#include <format>
#include <limits>
#include <mutex>
#include <sstream>
#include <vector>

#include "cccl/c/types.h"
#include "kernels/iterators.h"
#include "util/context.h"
#include "util/errors.h"
#include "util/indirect_arg.h"
#include "util/nvjitlink.h"
#include "util/serialization.h"
#include "util/types.h"
#include <cccl/c/histogram.h>
#include <cccl/c/serialization.h>
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
struct histogram_kernel_source
{
  cccl_device_histogram_build_result_t& build;

  template <typename PolicyT>
  CUkernel HistogramInitKernel() const
  {
    return build.init_kernel;
  }

  template <typename PolicyT, int PRIVATIZED_SMEM_BINS, typename PrivatizedDecodeOpT, typename OutputDecodeOpT>
  CUkernel HistogramSmemPrivatizedKernel() const
  {
    return build.sweep_kernel;
  }

  // Cooperative GMEM-privatized gather kernel (HybridSplit=false). Reached on the
  // high-bin tier (PRIVATIZED_SMEM_BINS == 0); JIT-compiled in the build under the
  // same name. Lower tiers never instantiate this (the if-constexpr branch is
  // discarded), so a null handle there is never launched.
  template <typename PolicyT, int PRIVATIZED_SMEM_BINS, typename PrivatizedDecodeOpT, typename OutputDecodeOpT>
  CUkernel HistogramGmemPrivatizedKernel() const
  {
    return build.gather_kernel;
  }

  // Direct-atomic cuckoo-cache kernel family. The host dispatch instantiates this
  // accessor for all six (ProbeOp x SpillOp) combinations, but the C Parallel
  // Library only ever launches `direct_atomic_cache_mode == 0` (cuckoo +
  // output_atomic_spill, with the second probe gated off on the >=262144-bin
  // tier): it never routes through dispatch_by_algorithm / select_algorithm. So
  // only the two output-spill cuckoo handles need to be valid; the unreachable
  // single-probe / no-cache / private-spill variants resolve to the primary
  // cuckoo handle and are never launched.
  template <typename PolicyT,
            int PRIVATIZED_SMEM_BINS,
            typename PrivatizedDecodeOpT,
            typename OutputDecodeOpT,
            typename ProbeOp,
            typename SpillOp>
  CUkernel HistogramDirectKernel() const
  {
    using ::cuda::std::is_same_v;
    if constexpr (is_same_v<ProbeOp, cub::detail::histogram::cuckoo_cache_probe</*DisableSecondProbe=*/true>>
                  && is_same_v<SpillOp, cub::detail::histogram::output_atomic_spill>)
    {
      return build.direct_cuckoo_noprobe2_kernel;
    }
    else
    {
      return build.direct_cuckoo_kernel;
    }
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

// Common type-name pieces shared by every histogram sweep/gather/direct kernel
// name. Factored out so the SMEM-privatized, GMEM-privatized-gather and
// direct-atomic kernel names all agree on the policy, sample-iterator and decode
// op spellings (the decode ops are template parameters that MUST match what the
// host dispatch instantiates, or the JIT lowered name won't resolve).
struct histogram_kernel_type_names
{
  std::string chained_policy_t;
  std::string samples_iterator_t;
  // Per-channel decode ops, already swapped for byte vs non-byte (matches
  // __dispatch_even_host_init / dispatch_even): non-byte EVEN -> privatized =
  // ScaleTransform, output = PassThruTransform; byte -> the reverse.
  std::string privatized_decode_op_t;
  std::string output_decode_op_t;
};

histogram_kernel_type_names get_histogram_kernel_type_names(
  cccl_iterator_t d_samples,
  std::string_view level_t,
  std::string_view offset_t,
  bool is_evenly_segmented,
  bool is_byte_sample)
{
  histogram_kernel_type_names names;
  check(cccl_type_name_from_nvrtc<device_histogram_policy>(&names.chained_policy_t));

  std::string samples_iterator_name;
  check(cccl_type_name_from_nvrtc<samples_iterator_t>(&samples_iterator_name));

  names.samples_iterator_t =
    d_samples.type == cccl_iterator_kind_t::CCCL_POINTER //
      ? cccl_type_enum_to_name(d_samples.value_type.type, true) //
      : samples_iterator_name;

  const std::string transforms_t = std::format(
    "cub::detail::histogram::Transforms<{0}, {1}, {2}>",
    level_t,
    offset_t,
    cccl_type_enum_to_name(d_samples.value_type.type));

  names.privatized_decode_op_t = std::format("{0}::PassThruTransform", transforms_t);
  names.output_decode_op_t =
    is_evenly_segmented
      ? std::format("{0}::ScaleTransform", transforms_t)
      : std::format("{0}::SearchTransform<const {1}*>", transforms_t, level_t);

  if (!is_byte_sample)
  {
    std::swap(names.privatized_decode_op_t, names.output_decode_op_t);
  }

  return names;
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
  const auto names = get_histogram_kernel_type_names(d_samples, level_t, offset_t, is_evenly_segmented, is_byte_sample);

  // HOST-INIT sweep kernel (no device-init): DeviceHistogramSmemPrivatizedKernel.
  // Its template params are <PolicySelector, PrivatizedSmemBins, NumChannels,
  // NumActiveChannels, SampleIteratorT, CounterT, PrivatizedDecodeOpT,
  // OutputDecodeOpT, OffsetT> -- it receives PRE-BUILT decode ops (constructed
  // host-side by build_scale_transform_bytes), so there are no level-array /
  // IsEven / IsByteSample template params.
  return std::format(
    "cub::detail::histogram::DeviceHistogramSmemPrivatizedKernel<{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}>",
    names.chained_policy_t,
    privatized_smem_bins,
    num_channels,
    num_active_channels,
    names.samples_iterator_t,
    counter_t,
    names.privatized_decode_op_t,
    names.output_decode_op_t,
    offset_t);
}

// High-bin cooperative kernel names (privatized_smem_bins == 0). These mirror the
// host-dispatch instantiations reached through the histogram_kernel_source
// accessors on the PRIVATIZED_SMEM_BINS == 0 path. High-bin is always non-byte
// (byte samples cap at 256 bins -> the 256 tier), so the decode ops are
// privatized = ScaleTransform / output = PassThruTransform.

// GMEM-privatized gather kernel: DeviceHistogramGmemPrivatizedKernel<Policy, 0,
// NumChannels, NumActiveChannels, SampleItr, Counter, PrivDecode, OutDecode,
// Offset, /*HybridSplit=*/false>.
std::string get_gather_kernel_name(
  int num_channels,
  int num_active_channels,
  cccl_iterator_t d_samples,
  std::string_view counter_t,
  std::string_view level_t,
  std::string_view offset_t,
  bool is_evenly_segmented)
{
  const auto names = get_histogram_kernel_type_names(
    d_samples, level_t, offset_t, is_evenly_segmented, /*is_byte_sample=*/false);
  return std::format(
    "cub::detail::histogram::DeviceHistogramGmemPrivatizedKernel<{0}, 0, {1}, {2}, {3}, {4}, {5}, {6}, {7}, false>",
    names.chained_policy_t,
    num_channels,
    num_active_channels,
    names.samples_iterator_t,
    counter_t,
    names.privatized_decode_op_t,
    names.output_decode_op_t,
    offset_t);
}

// Direct-atomic cuckoo-cache kernel: DeviceHistogramDirectKernel<Policy, 0,
// NumChannels, NumActiveChannels, SampleItr, Counter, PrivDecode, OutDecode,
// Offset, cuckoo_cache_probe<DisableSecondProbe>, output_atomic_spill>.
std::string get_direct_cuckoo_kernel_name(
  int num_channels,
  int num_active_channels,
  cccl_iterator_t d_samples,
  std::string_view counter_t,
  std::string_view level_t,
  std::string_view offset_t,
  bool is_evenly_segmented,
  bool disable_second_probe)
{
  const auto names = get_histogram_kernel_type_names(
    d_samples, level_t, offset_t, is_evenly_segmented, /*is_byte_sample=*/false);
  return std::format(
    "cub::detail::histogram::DeviceHistogramDirectKernel<{0}, 0, {1}, {2}, {3}, {4}, {5}, {6}, {7}, "
    "cub::detail::histogram::cuckoo_cache_probe<{8}>, cub::detail::histogram::output_atomic_spill>",
    names.chained_policy_t,
    num_channels,
    num_active_channels,
    names.samples_iterator_t,
    counter_t,
    names.privatized_decode_op_t,
    names.output_decode_op_t,
    offset_t,
    disable_second_probe ? "true" : "false");
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

// ---------------------------------------------------------------------------
// Host-side decode-op construction (so C-parallel uses the HOST-INIT histogram
// sweep kernel, never the device-init variant).
//
// The host-init `DeviceHistogramSmemPrivatizedKernel` takes the per-channel
// decode operator (`Transforms<LevelT,OffsetT,SampleT>::ScaleTransform` for the
// EVEN path) BY VALUE as a `_CCCL_GRID_CONSTANT` argument. C-parallel's dispatch
// is compiled with type-erased `indirect_arg_t`, so it cannot name `ScaleTransform`
// at compile time -- but `histogram.cu` is an nvcc TU with the CUB headers, so we
// can instantiate the REAL transform behind a runtime (sample-type, level-type)
// tag dispatch, call its host-runnable `Init`, and hand the resulting POD bytes to
// the kernel. This is the same "tag -> real type" idiom as `get_integral_range`
// above, extended to the 2-D (sample, level) matrix because `ScaleTransform`'s
// layout depends on `CommonT = common_type<LevelT,SampleT>`.
//
// We build the whole `cuda::std::array<ScaleTransform, NumActiveChannels>` into a
// byte buffer; `decode_op_arg_t` (below) owns it and its `operator&` yields
// the buffer address, exactly like `indirect_arg_t`, so the existing launcher
// marshals it as the kernel's grid-constant decode-op argument.

// Build array<ScaleTransform<L,O,S>, N> bytes for the given (already typed) L,S.
template <typename SampleT, typename LevelT, int NumActiveChannels, typename OffsetT>
std::vector<char> build_scale_transform_bytes_typed(
  const std::vector<int>& num_output_levels, const void* lower, const void* upper)
{
  using TransformsT = cub::detail::histogram::Transforms<LevelT, OffsetT, SampleT>;
  using ScaleT      = typename TransformsT::ScaleTransform;
  using ArrayT      = ::cuda::std::array<ScaleT, NumActiveChannels>;

  ArrayT ops{};
  // The C histogram_even API takes a single scalar lower/upper level, broadcast
  // across active channels (the C histogram_even API takes one scalar
  // lower/upper pair, applied to every channel).
  const LevelT lo = *static_cast<const LevelT*>(lower);
  const LevelT up = *static_cast<const LevelT*>(upper);
  for (int ch = 0; ch < NumActiveChannels; ++ch)
  {
    // EVEN: ScaleTransform::Init(num_levels, max_level, min_level).
    ops[ch].Init(num_output_levels[ch], up, lo);
  }
  std::vector<char> bytes(sizeof(ArrayT));
  std::memcpy(bytes.data(), &ops, sizeof(ArrayT));
  return bytes;
}

// Tag-dispatch on the LEVEL type (inner), given an already-resolved sample type.
template <typename SampleT, int NumActiveChannels, typename OffsetT>
std::vector<char> build_scale_transform_bytes_level(
  cccl_type_enum level_type, const std::vector<int>& num_output_levels, const void* lower, const void* upper)
{
  switch (level_type)
  {
    case CCCL_INT8:
      return build_scale_transform_bytes_typed<SampleT, int8_t, NumActiveChannels, OffsetT>(num_output_levels, lower, upper);
    case CCCL_UINT8:
      return build_scale_transform_bytes_typed<SampleT, uint8_t, NumActiveChannels, OffsetT>(num_output_levels, lower, upper);
    case CCCL_INT16:
      return build_scale_transform_bytes_typed<SampleT, int16_t, NumActiveChannels, OffsetT>(num_output_levels, lower, upper);
    case CCCL_UINT16:
      return build_scale_transform_bytes_typed<SampleT, uint16_t, NumActiveChannels, OffsetT>(num_output_levels, lower, upper);
    case CCCL_INT32:
      return build_scale_transform_bytes_typed<SampleT, int32_t, NumActiveChannels, OffsetT>(num_output_levels, lower, upper);
    case CCCL_UINT32:
      return build_scale_transform_bytes_typed<SampleT, uint32_t, NumActiveChannels, OffsetT>(num_output_levels, lower, upper);
    case CCCL_INT64:
      return build_scale_transform_bytes_typed<SampleT, int64_t, NumActiveChannels, OffsetT>(num_output_levels, lower, upper);
    case CCCL_UINT64:
      return build_scale_transform_bytes_typed<SampleT, uint64_t, NumActiveChannels, OffsetT>(num_output_levels, lower, upper);
    case CCCL_FLOAT32:
      return build_scale_transform_bytes_typed<SampleT, float, NumActiveChannels, OffsetT>(num_output_levels, lower, upper);
    case CCCL_FLOAT64:
      return build_scale_transform_bytes_typed<SampleT, double, NumActiveChannels, OffsetT>(num_output_levels, lower, upper);
#if _CCCL_HAS_NVFP16()
    case CCCL_FLOAT16:
      return build_scale_transform_bytes_typed<SampleT, __half, NumActiveChannels, OffsetT>(num_output_levels, lower, upper);
#endif
    default:
      throw std::runtime_error("histogram: unsupported level type for host-side decode-op build");
  }
}

// Tag-dispatch on the SAMPLE type (outer).
template <int NumActiveChannels, typename OffsetT>
std::vector<char> build_scale_transform_bytes(
  cccl_type_enum sample_type,
  cccl_type_enum level_type,
  const std::vector<int>& num_output_levels,
  const void* lower,
  const void* upper)
{
  switch (sample_type)
  {
    case CCCL_INT8:
      return build_scale_transform_bytes_level<int8_t, NumActiveChannels, OffsetT>(level_type, num_output_levels, lower, upper);
    case CCCL_UINT8:
      return build_scale_transform_bytes_level<uint8_t, NumActiveChannels, OffsetT>(level_type, num_output_levels, lower, upper);
    case CCCL_INT16:
      return build_scale_transform_bytes_level<int16_t, NumActiveChannels, OffsetT>(level_type, num_output_levels, lower, upper);
    case CCCL_UINT16:
      return build_scale_transform_bytes_level<uint16_t, NumActiveChannels, OffsetT>(level_type, num_output_levels, lower, upper);
    case CCCL_INT32:
      return build_scale_transform_bytes_level<int32_t, NumActiveChannels, OffsetT>(level_type, num_output_levels, lower, upper);
    case CCCL_UINT32:
      return build_scale_transform_bytes_level<uint32_t, NumActiveChannels, OffsetT>(level_type, num_output_levels, lower, upper);
    case CCCL_INT64:
      return build_scale_transform_bytes_level<int64_t, NumActiveChannels, OffsetT>(level_type, num_output_levels, lower, upper);
    case CCCL_UINT64:
      return build_scale_transform_bytes_level<uint64_t, NumActiveChannels, OffsetT>(level_type, num_output_levels, lower, upper);
    case CCCL_FLOAT32:
      return build_scale_transform_bytes_level<float, NumActiveChannels, OffsetT>(level_type, num_output_levels, lower, upper);
    case CCCL_FLOAT64:
      return build_scale_transform_bytes_level<double, NumActiveChannels, OffsetT>(level_type, num_output_levels, lower, upper);
#if _CCCL_HAS_NVFP16()
    case CCCL_FLOAT16:
      return build_scale_transform_bytes_level<__half, NumActiveChannels, OffsetT>(level_type, num_output_levels, lower, upper);
#endif
    default:
      throw std::runtime_error("histogram: unsupported sample type for host-side decode-op build");
  }
}

// Build array<PassThruTransform, N> bytes. PassThruTransform is (near-)empty and
// type-independent in layout, so we use the int/int instantiation; its size
// matches what the kernel expects for any LevelT/SampleT (a single dummy byte +
// padding, or empty). Output decode op for the non-byte EVEN host-init path.
template <int NumActiveChannels>
std::vector<char> build_passthru_transform_bytes()
{
  using TransformsT = cub::detail::histogram::Transforms<int, OffsetT, int>;
  using PassThruT   = typename TransformsT::PassThruTransform;
  using ArrayT      = ::cuda::std::array<PassThruT, NumActiveChannels>;
  ArrayT ops{};
  std::vector<char> bytes(sizeof(ArrayT));
  std::memcpy(bytes.data(), &ops, sizeof(ArrayT));
  return bytes;
}

// Owns the host-built decode-op bytes and presents them to the launcher like
// indirect_arg_t: `operator&` returns the buffer address, from which the driver
// copies the kernel's grid-constant decode-op parameter (whose true size is the
// JIT-compiled `array<ScaleTransform,N>`; our buffer is exactly that size). The
// launcher takes args by const-ref, so `operator&` is const.
struct decode_op_arg_t
{
  // `dispatch<>` reads `FirstLevelArrayT::value_type` to name the decode-op type
  // for the host-init accessor; for C-parallel that accessor ignores its template
  // args (it returns the JIT-built sweep kernel), so a placeholder suffices.
  using value_type = indirect_arg_t;

  std::vector<char> bytes;
  void* operator&() const
  {
    return const_cast<void*>(static_cast<const void*>(bytes.data()));
  }
};

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

CUresult cccl_device_histogram_compile(
  cccl_device_histogram_build_result_t* build_ptr,
  int num_channels,
  int num_active_channels,
  cccl_iterator_t d_samples,
  int num_output_levels_val,
  cccl_iterator_t d_output_histograms,
  cccl_type_info level_type,
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
try
{
  const char* name = "test";

  const cuda::compute_capability cc{cc_major, cc_minor};
  const auto sample_cpp  = cccl_type_enum_to_name(d_samples.value_type.type);
  const auto counter_cpp = cccl_type_enum_to_name(d_output_histograms.value_type.type);
  const auto level_cpp   = cccl_type_enum_to_name(level_type.type);

  const std::string offset_cpp =
    ((unsigned long long) (num_rows * row_stride_samples * d_samples.value_type.size) < (unsigned long long) INT_MAX)
      ? "int"
      : "long long";

  std::string samples_iterator_name;
  check(cccl_type_name_from_nvrtc<samples_iterator_t>(&samples_iterator_name));

  const std::string samples_iterator_src =
    make_kernel_input_iterator(offset_cpp, samples_iterator_name, sample_cpp, d_samples);

  const bool sample_is_primitive = d_samples.value_type.type != CCCL_STORAGE; // TODO(bgruber): how to check if sample
                                                                              // is primitive?
  const auto policy_sel = cub::detail::histogram::policy_selector{
    sample_is_primitive,
    static_cast<int>(d_samples.value_type.size),
    static_cast<int>(d_output_histograms.value_type.size),
    static_cast<int>(d_samples.value_type.size),
    num_channels,
    num_active_channels,
    is_evenly_segmented};

  const auto active_policy = policy_sel(cc);

  std::stringstream policy_sel_str;
  policy_sel_str << active_policy;

  std::string policy_selector_expr = std::format(
    "cub::detail::histogram::policy_selector_from_types<{}, {}, {}, {}, {}>",
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
using device_histogram_policy = {3};
using namespace cub;
using namespace cub::detail::histogram;
static_assert(device_histogram_policy()(detail::current_tuning_cc()) == {4}, "Host generated and JIT compiled policy mismatch");
)XXX",
    d_samples.value_type.size, // 0
    d_samples.value_type.alignment, // 1
    samples_iterator_src, // 2
    policy_selector_expr, // 3
    policy_sel_str.view() // 4
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

  // On the high-bin tier the host dispatch takes the cooperative path and launches
  // the GMEM-privatized gather kernel and (for >=65536 / >=262144 bins) the
  // direct-atomic cuckoo kernels, all via cuLaunchCooperativeKernel. JIT them here
  // and resolve their handles below; the <=256-bin tier never reaches them, so we
  // skip compiling them there. (sweep_kernel stays the cooperative fallback used
  // when a cooperative launch is unsupported.)
  const bool high_bin                       = (privatized_smem_bins == 0);
  std::string gather_kernel_name            = high_bin ? histogram::get_gather_kernel_name(
                                        num_channels, num_active_channels, d_samples, counter_cpp, level_cpp, offset_cpp, is_evenly_segmented)
                                                        : std::string{};
  std::string direct_cuckoo_kernel_name     = high_bin ? histogram::get_direct_cuckoo_kernel_name(
                                            num_channels,
                                            num_active_channels,
                                            d_samples,
                                            counter_cpp,
                                            level_cpp,
                                            offset_cpp,
                                            is_evenly_segmented,
                                            /*disable_second_probe=*/false)
                                                            : std::string{};
  std::string direct_cuckoo_noprobe2_kernel_name = high_bin ? histogram::get_direct_cuckoo_kernel_name(
                                                     num_channels,
                                                     num_active_channels,
                                                     d_samples,
                                                     counter_cpp,
                                                     level_cpp,
                                                     offset_cpp,
                                                     is_evenly_segmented,
                                                     /*disable_second_probe=*/true)
                                                                : std::string{};

  std::string init_kernel_lowered_name;
  std::string sweep_kernel_lowered_name;
  std::string gather_kernel_lowered_name;
  std::string direct_cuckoo_kernel_lowered_name;
  std::string direct_cuckoo_noprobe2_kernel_lowered_name;

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
      ->add_expression_if(high_bin, {gather_kernel_name})
      ->add_expression_if(high_bin, {direct_cuckoo_kernel_name})
      ->add_expression_if(high_bin, {direct_cuckoo_noprobe2_kernel_name})
      ->compile_program({args.data(), args.size()})
      ->get_name({init_kernel_name, init_kernel_lowered_name})
      ->get_name({sweep_kernel_name, sweep_kernel_lowered_name})
      ->get_name_if(high_bin, {gather_kernel_name, gather_kernel_lowered_name})
      ->get_name_if(high_bin, {direct_cuckoo_kernel_name, direct_cuckoo_kernel_lowered_name})
      ->get_name_if(high_bin, {direct_cuckoo_noprobe2_kernel_name, direct_cuckoo_noprobe2_kernel_lowered_name})
      ->link_program()
      ->add_link_list(linkable_list)
      ->finalize_program();

  struct free_deleter
  {
    void operator()(void* p) const
    {
      std::free(p);
    }
  };
  static_assert(::cuda::is_trivially_copyable_v<cub::detail::histogram::policy_selector>);
  const size_t policy_size = sizeof(policy_sel);
  std::unique_ptr<void, free_deleter> policy_ptr(std::malloc(policy_size));
  if (!policy_ptr)
  {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  std::memcpy(policy_ptr.get(), &policy_sel, sizeof(policy_sel));
  auto init_name  = std::unique_ptr<char[]>(duplicate_c_string(init_kernel_lowered_name));
  auto sweep_name = std::unique_ptr<char[]>(duplicate_c_string(sweep_kernel_lowered_name));
  auto gather_name = std::unique_ptr<char[]>(
    high_bin ? duplicate_c_string(gather_kernel_lowered_name) : nullptr);
  auto direct_cuckoo_name = std::unique_ptr<char[]>(
    high_bin ? duplicate_c_string(direct_cuckoo_kernel_lowered_name) : nullptr);
  auto direct_cuckoo_noprobe2_name = std::unique_ptr<char[]>(
    high_bin ? duplicate_c_string(direct_cuckoo_noprobe2_kernel_lowered_name) : nullptr);

  build_ptr->cc                  = cc.get();
  build_ptr->counter_type        = d_output_histograms.value_type;
  build_ptr->level_type          = level_type;
  build_ptr->sample_type         = d_samples.value_type;
  build_ptr->num_active_channels = num_active_channels;
  build_ptr->may_overflow = false; // This is set in cccl_device_histogram_even_impl so that kernel source can access
                                   // it later.
  // Zero-init fields set by _load, not _compile.
  build_ptr->library      = nullptr;
  build_ptr->init_kernel  = nullptr;
  build_ptr->sweep_kernel = nullptr;
  build_ptr->gather_kernel = nullptr;
  build_ptr->direct_cuckoo_kernel = nullptr;
  build_ptr->direct_cuckoo_noprobe2_kernel = nullptr;

  build_ptr->payload      = (void*) result.data.release();
  build_ptr->payload_size = result.size;
  build_ptr->payload_kind = CCCL_PAYLOAD_CUBIN;

  build_ptr->runtime_policy            = policy_ptr.release();
  build_ptr->runtime_policy_size       = policy_size;
  build_ptr->init_kernel_lowered_name  = init_name.release();
  build_ptr->sweep_kernel_lowered_name = sweep_name.release();
  build_ptr->gather_kernel_lowered_name = gather_name.release();
  build_ptr->direct_cuckoo_kernel_lowered_name = direct_cuckoo_name.release();
  build_ptr->direct_cuckoo_noprobe2_kernel_lowered_name = direct_cuckoo_noprobe2_name.release();

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_histogram_compile(): %s\n", exc.what());
  fflush(stdout);
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_histogram_load(cccl_device_histogram_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr || build_ptr->payload == nullptr || build_ptr->payload_size == 0
      || build_ptr->payload_kind != CCCL_PAYLOAD_CUBIN || build_ptr->init_kernel_lowered_name == nullptr
      || build_ptr->init_kernel_lowered_name[0] == '\0' || build_ptr->sweep_kernel_lowered_name == nullptr
      || build_ptr->sweep_kernel_lowered_name[0] == '\0')
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
    check(cuLibraryGetKernel(&build_ptr->init_kernel, build_ptr->library, build_ptr->init_kernel_lowered_name));
    check(cuLibraryGetKernel(&build_ptr->sweep_kernel, build_ptr->library, build_ptr->sweep_kernel_lowered_name));
    if (build_ptr->gather_kernel_lowered_name != nullptr)
    {
      check(cuLibraryGetKernel(&build_ptr->gather_kernel, build_ptr->library, build_ptr->gather_kernel_lowered_name));
      check(cuLibraryGetKernel(
        &build_ptr->direct_cuckoo_kernel, build_ptr->library, build_ptr->direct_cuckoo_kernel_lowered_name));
      check(cuLibraryGetKernel(&build_ptr->direct_cuckoo_noprobe2_kernel,
                               build_ptr->library,
                               build_ptr->direct_cuckoo_noprobe2_kernel_lowered_name));
    }
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
  printf("\nEXCEPTION in cccl_device_histogram_load(): %s\n", exc.what());
  fflush(stdout);
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_histogram_build_ex(
  cccl_device_histogram_build_result_t* build_ptr,
  int num_channels,
  int num_active_channels,
  cccl_iterator_t d_samples,
  int num_output_levels_val,
  cccl_iterator_t d_output_histograms,
  cccl_type_info level_type,
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
  CUresult r = cccl_device_histogram_compile(
    build_ptr,
    num_channels,
    num_active_channels,
    d_samples,
    num_output_levels_val,
    d_output_histograms,
    level_type,
    num_rows,
    row_stride_samples,
    is_evenly_segmented,
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
  CUresult load_r = cccl_device_histogram_load(build_ptr);
  if (load_r != CUDA_SUCCESS)
  {
    cccl_device_histogram_cleanup(build_ptr);
  }
  return load_r;
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

    // HOST-INIT path (no device-init kernel): build the real decode ops host-side
    // via a (sample,level) type-tag dispatch. The ScaleTransform does the
    // even-bin classify; PassThruTransform is the trivial identity. The kernel
    // reads them as grid-constant args; `decode_op_arg_t::operator&` yields the
    // bytes, just like `indirect_arg_t`.
    //
    // Decode-op assignment mirrors cub::detail::histogram::dispatch_even:
    //   non-byte EVEN: privatized = ScaleTransform, output = PassThruTransform
    //                  (privatized bins == output bins).
    //   byte sample:   privatized = PassThruTransform (256-entry staging),
    //                  output = ScaleTransform; always the 256-bin tier.
    histogram::decode_op_arg_t scale_arg{
      histogram::build_scale_transform_bytes<NUM_ACTIVE_CHANNELS, OffsetT>(
        build.sample_type.type, build.level_type.type, {num_output_levels_arr[0]}, lower_level.state, upper_level.state)};
    histogram::decode_op_arg_t passthru_arg{histogram::build_passthru_transform_bytes<NUM_ACTIVE_CHANNELS>()};

    // For byte samples the ScaleTransform is the OUTPUT op and PassThru is
    // privatized; otherwise ScaleTransform is privatized and PassThru is output.
    histogram::decode_op_arg_t& output_decode_op_arg     = is_byte_sample::value ? scale_arg : passthru_arg;
    histogram::decode_op_arg_t& privatized_decode_op_arg = is_byte_sample::value ? passthru_arg : scale_arg;

    auto exec_status = cub::detail::histogram::__dispatch_even_host_init<
      NUM_CHANNELS,
      NUM_ACTIVE_CHANNELS,
      indirect_arg_t, // SampleIteratorT
      indirect_arg_t, // CounterT
      indirect_arg_t, // LevelT
      OffsetT, // OffsetT
      cub::detail::histogram::policy_selector, // PolicySelector
      histogram::decode_op_arg_t&, // OutputDecodeOpArrayT (holder)
      histogram::decode_op_arg_t&, // PrivatizedDecodeOpArrayT (holder)
      indirect_arg_t, // SampleT
      histogram::histogram_kernel_source, // KernelSource
      cub::detail::CudaDriverLauncherFactory // KernelLauncherFactory
      >(d_temp_storage,
        *temp_storage_bytes,
        d_samples,
        d_output_histogram_arr,
        num_output_levels_arr,
        output_decode_op_arg,
        privatized_decode_op_arg,
        num_row_pixels,
        num_rows,
        row_stride_samples,
        stream,
        is_byte_sample::value,
        *reinterpret_cast<cub::detail::histogram::policy_selector*>(build.runtime_policy),
        {build},
        cub::detail::CudaDriverLauncherFactory{cu_device, build.cc});

    error = static_cast<CUresult>(exec_status);
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_histogram_even_impl(): %s\n", exc.what());
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
  cccl_type_info level_type,
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
    level_type,
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
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  std::unique_ptr<char[]> payload(reinterpret_cast<char*>(build_ptr->payload));
  std::free(build_ptr->runtime_policy);
  std::unique_ptr<char[]> init_name(build_ptr->init_kernel_lowered_name);
  std::unique_ptr<char[]> sweep_name(build_ptr->sweep_kernel_lowered_name);
  std::unique_ptr<char[]> gather_name(build_ptr->gather_kernel_lowered_name);
  std::unique_ptr<char[]> direct_cuckoo_name(build_ptr->direct_cuckoo_kernel_lowered_name);
  std::unique_ptr<char[]> direct_cuckoo_noprobe2_name(build_ptr->direct_cuckoo_noprobe2_kernel_lowered_name);
  if (build_ptr->library != nullptr)
  {
    check(cuLibraryUnload(build_ptr->library));
  }

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_histogram_cleanup(): %s\n", exc.what());
  fflush(stdout);
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_histogram_link_ltoir(
  cccl_device_histogram_build_result_t* build_ptr,
  const void** input_blobs,
  const size_t* input_sizes,
  size_t num_inputs)
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
  printf("\nEXCEPTION in cccl_device_histogram_link_ltoir(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult
cccl_device_histogram_serialize(const cccl_device_histogram_build_result_t* build_ptr, void** out_buf, size_t* out_size)
try
{
  if (build_ptr == nullptr || out_buf == nullptr || out_size == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (build_ptr->payload == nullptr || build_ptr->payload_size == 0 || build_ptr->runtime_policy == nullptr
      || build_ptr->runtime_policy_size == 0)
  {
    *out_buf  = nullptr;
    *out_size = 0;
    return CUDA_ERROR_INVALID_VALUE;
  }

  using namespace cccl::serialization;
  buffer_writer w;
  write_header(w, CCCL_SERIALIZATION_ALGO_HISTOGRAM, build_ptr->payload_kind, build_ptr->cc);
  write_type_info(w, build_ptr->counter_type);
  write_type_info(w, build_ptr->level_type);
  write_type_info(w, build_ptr->sample_type);
  w.write_pod<int32_t>(build_ptr->num_active_channels);
  w.write_pod<uint8_t>(build_ptr->may_overflow ? 1 : 0);
  w.write_blob(build_ptr->payload, build_ptr->payload_size);
  w.write_blob(build_ptr->runtime_policy, build_ptr->runtime_policy_size);
  w.write_cstring(build_ptr->init_kernel_lowered_name);
  w.write_cstring(build_ptr->sweep_kernel_lowered_name);
  w.write_cstring(build_ptr->gather_kernel_lowered_name);
  w.write_cstring(build_ptr->direct_cuckoo_kernel_lowered_name);
  w.write_cstring(build_ptr->direct_cuckoo_noprobe2_kernel_lowered_name);
  w.release(out_buf, out_size);
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_histogram_serialize(): %s\n", exc.what());
  fflush(stdout);
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_histogram_deserialize(cccl_device_histogram_build_result_t* build_ptr, const void* buf, size_t size)
try
{
  if (build_ptr == nullptr || buf == nullptr || size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  using namespace cccl::serialization;
  buffer_reader r{buf, size};
  const auto h = read_and_validate_header(r, CCCL_SERIALIZATION_ALGO_HISTOGRAM);

  const auto counter_t  = read_type_info(r);
  const auto level_t    = read_type_info(r);
  const auto sample_t   = read_type_info(r);
  const int32_t nac     = r.read_pod<int32_t>();
  const bool overflow_b = r.read_pod<uint8_t>() != 0;

  std::unique_ptr<char[]> payload_owner;
  size_t payload_size = 0;
  {
    void* p = nullptr;
    r.read_blob_new(&p, &payload_size);
    payload_owner.reset(static_cast<char*>(p));
  }
  if (payload_size == 0)
  {
    throw std::runtime_error("serialization blob: empty payload");
  }

  std::unique_ptr<cub::detail::histogram::policy_selector, decltype(&std::free)> policy(
    static_cast<cub::detail::histogram::policy_selector*>(std::malloc(sizeof(cub::detail::histogram::policy_selector))),
    std::free);
  if (!policy)
  {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  r.read_into(policy.get(), sizeof(cub::detail::histogram::policy_selector));

  std::unique_ptr<char[]> n_init{r.read_cstring_dup()};
  std::unique_ptr<char[]> n_sweep{r.read_cstring_dup()};
  std::unique_ptr<char[]> n_gather{r.read_cstring_dup()};
  std::unique_ptr<char[]> n_direct_cuckoo{r.read_cstring_dup()};
  std::unique_ptr<char[]> n_direct_cuckoo_noprobe2{r.read_cstring_dup()};

  cccl_device_histogram_build_result_t result{};
  result.cc                        = static_cast<int>(h.cc);
  result.payload_kind              = static_cast<cccl_payload_kind_t>(h.payload_kind);
  result.counter_type              = counter_t;
  result.level_type                = level_t;
  result.sample_type               = sample_t;
  result.num_active_channels       = nac;
  result.may_overflow              = overflow_b;
  result.payload                   = payload_owner.release();
  result.payload_size              = payload_size;
  result.runtime_policy            = policy.release();
  result.runtime_policy_size       = sizeof(cub::detail::histogram::policy_selector);
  result.init_kernel_lowered_name  = n_init.release();
  result.sweep_kernel_lowered_name = n_sweep.release();
  result.gather_kernel_lowered_name = n_gather.release();
  result.direct_cuckoo_kernel_lowered_name = n_direct_cuckoo.release();
  result.direct_cuckoo_noprobe2_kernel_lowered_name = n_direct_cuckoo_noprobe2.release();
  *build_ptr                       = result;
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_histogram_deserialize(): %s\n", exc.what());
  fflush(stdout);
  return CUDA_ERROR_UNKNOWN;
}
