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
#include <cub/detail/launcher/cuda_driver.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/util_device.cuh>

#include <cuda/std/algorithm>
#include <cuda/std/cstdint>
#include <cuda/std/functional> // ::cuda::std::identity
#include <cuda/std/variant>

#include <format>
#include <memory>
#include <vector>

#include "jit_templates/templates/input_iterator.h"
#include "jit_templates/templates/operation.h"
#include "jit_templates/templates/output_iterator.h"
#include "jit_templates/traits.h"
#include "util/context.h"
#include "util/errors.h"
#include "util/indirect_arg.h"
#include "util/types.h"
#include <cccl/c/reduce.h>
#include <nvrtc/command_list.h>
#include <nvrtc/ltoir_list_appender.h>
#include <util/build_utils.h>

struct device_reduce_policy;
using TransformOpT = ::cuda::std::identity;
using OffsetT      = unsigned long long;
static_assert(std::is_same_v<cub::detail::choose_offset_t<OffsetT>, OffsetT>, "OffsetT must be size_t");

namespace reduce
{
static cccl_type_info get_accumulator_type(cccl_op_t /*op*/, cccl_iterator_t /*input_it*/, cccl_value_t init)
{
  // TODO Should be decltype(op(init, *input_it)) but haven't implemented type arithmetic yet
  //      so switching back to the old accumulator type logic for now
  return init.type;
}

std::string get_single_tile_kernel_name(
  std::string_view input_iterator_t,
  std::string_view output_iterator_t,
  std::string_view reduction_op_t,
  cccl_value_t init,
  std::string_view accum_cpp_t,
  bool is_second_kernel)
{
  std::string chained_policy_t;
  check(cccl_type_name_from_nvrtc<device_reduce_policy>(&chained_policy_t));

  const std::string init_t = cccl_type_enum_to_name(init.type.type);

  std::string offset_t;
  if (is_second_kernel)
  {
    // Second kernel is always invoked with an int offset.
    // See the definition of the local variable `reduce_grid_size`
    // in DispatchReduce::InvokePasses.
    check(cccl_type_name_from_nvrtc<int>(&offset_t));
  }
  else
  {
    check(cccl_type_name_from_nvrtc<OffsetT>(&offset_t));
  }

  return std::format(
    "cub::detail::reduce::DeviceReduceSingleTileKernel<{0}, {1}, {2}, {3}, {4}, {5}, {6}>",
    chained_policy_t,
    input_iterator_t,
    output_iterator_t,
    offset_t,
    reduction_op_t,
    init_t,
    accum_cpp_t);
}

std::string get_device_reduce_kernel_name(
  std::string_view reduction_op_t, std::string_view input_iterator_t, std::string_view accum_t)
{
  std::string chained_policy_t;
  check(cccl_type_name_from_nvrtc<device_reduce_policy>(&chained_policy_t));

  std::string offset_t;
  check(cccl_type_name_from_nvrtc<OffsetT>(&offset_t));

  std::string transform_op_t;
  check(cccl_type_name_from_nvrtc<cuda::std::identity>(&transform_op_t));

  return std::format(
    "cub::detail::reduce::DeviceReduceKernel<{0}, {1}, {2}, {3}, {4}, {5}>",
    chained_policy_t,
    input_iterator_t,
    offset_t,
    reduction_op_t,
    accum_t,
    transform_op_t);
}

std::string get_device_reduce_nondeterministic_kernel_name(
  std::string_view input_iterator_t,
  std::string_view output_iterator_t,
  std::string_view reduction_op_t,
  std::string_view accum_t,
  cccl_value_t init)
{
  std::string chained_policy_t;
  check(cccl_type_name_from_nvrtc<device_reduce_policy>(&chained_policy_t));

  std::string offset_t;
  check(cccl_type_name_from_nvrtc<OffsetT>(&offset_t));

  std::string transform_op_t;
  check(cccl_type_name_from_nvrtc<cuda::std::identity>(&transform_op_t));

  const std::string init_t = cccl_type_enum_to_name(init.type.type);

  return std::format(
    "cub::detail::reduce::NondeterministicDeviceReduceAtomicKernel<{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}>",
    chained_policy_t,
    input_iterator_t,
    output_iterator_t,
    offset_t,
    reduction_op_t,
    accum_t,
    init_t,
    transform_op_t);
}

struct reduce_kernel_source
{
  cccl_device_reduce_build_result_t& build;

  std::size_t AccumSize() const
  {
    return build.accumulator_size;
  }
  CUkernel SingleTileKernel() const
  {
    return build.single_tile_kernel;
  }
  CUkernel SingleTileSecondKernel() const
  {
    return build.single_tile_second_kernel;
  }
  CUkernel ReductionKernel() const
  {
    return build.reduction_kernel;
  }
  CUkernel NondeterministicAtomicKernel() const
  {
    return build.nondeterministic_atomic_kernel;
  }
  size_t InitSize() const
  {
    return build.accumulator_size;
  }
};
} // namespace reduce

struct reduce_iterator_tag;
struct reduction_operation_tag;

CUresult cccl_device_reduce_build_ex(
  cccl_device_reduce_build_result_t* build,
  cccl_iterator_t input_it,
  cccl_iterator_t output_it,
  cccl_op_t op,
  cccl_value_t init,
  cccl_determinism_t determinism,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config)
try
{
  if (determinism == CCCL_NOT_GUARANTEED && (op.type != CCCL_PLUS || output_it.type != CCCL_POINTER))
  {
    fflush(stderr);
    printf("\nERROR in cccl_device_reduce_build(): non-deterministic reduce with non-plus operator or non-pointer "
           "output iterator is not supported\n");
    fflush(stdout);
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (determinism == CCCL_GPU_TO_GPU)
  {
    fflush(stderr);
    printf("\nERROR in cccl_device_reduce_build(): gpu-to-gpu determinism is not supported\n");
    fflush(stdout);
    return CUDA_ERROR_INVALID_VALUE;
  }

  const char* name = "device_reduce";

  const cccl_type_info accum_t = reduce::get_accumulator_type(op, input_it, init);
  const auto accum_cpp         = cccl_type_enum_to_name(accum_t.type);

  const auto [input_iterator_name, input_iterator_src] =
    get_specialization<reduce_iterator_tag>(template_id<input_iterator_traits>(), input_it);
  const auto [output_iterator_name, output_iterator_src] =
    get_specialization<reduce_iterator_tag>(template_id<output_iterator_traits>(), output_it, accum_t);

  const auto [op_name, op_src] =
    get_specialization<reduction_operation_tag>(template_id<binary_user_operation_traits>(), op, accum_t);

  const auto offset_t = cccl_type_enum_to_name(cccl_type_enum::CCCL_UINT64);

  const auto policy_sel = [&] {
    using namespace cub::detail::reduce;

    auto accum_type = accum_type::other;
    if (accum_t.type == CCCL_FLOAT32)
    {
      accum_type = accum_type::float32;
    }
    else if (accum_t.type == CCCL_FLOAT64)
    {
      accum_type = accum_type::float64;
    }

    auto operation_t = op_type::unknown;
    switch (op.type)
    {
      case CCCL_PLUS:
        operation_t = op_type::plus;
        break;
      case CCCL_MINIMUM:
      case CCCL_MAXIMUM:
        operation_t = op_type::min_or_max;
        break;
      default:
        break;
    }

    const int offset_size = int{sizeof(OffsetT)};
    return policy_selector{accum_type, operation_t, offset_size, static_cast<int>(accum_t.size)};
  }();

  // TODO(bgruber): drop this if tuning policies become formattable
  std::stringstream policy_sel_str;
  policy_sel_str << policy_sel(cuda::to_arch_id(cuda::compute_capability{cc_major, cc_minor}));

  auto policy_hub_expr =
    std::format("cub::detail::reduce::policy_selector_from_types<{}, {}, {}>", accum_cpp, offset_t, op_name);

  std::string final_src = std::format(
    R"XXX(
#include <cub/device/dispatch/tuning/tuning_reduce.cuh>
#include <cub/device/dispatch/kernels/kernel_reduce.cuh>
{0}
struct __align__({2}) storage_t {{
  char data[{1}];
}};
{3}
{4}
{5}
using device_reduce_policy = {6};
using namespace cub;
using namespace cub::detail::reduce;
static_assert(device_reduce_policy()(::cuda::arch_id{{CUB_PTX_ARCH / 10}}) == {7}, "Host generated and JIT compiled policy mismatch");
)XXX",
    jit_template_header_contents, // 0
    input_it.value_type.size, // 1
    input_it.value_type.alignment, // 2
    input_iterator_src, // 3
    output_iterator_src, // 4
    op_src, // 5
    policy_hub_expr, // 6
    policy_sel_str.view()); // 7

#if false // CCCL_DEBUGGING_SWITCH
  fflush(stderr);
  printf("\nCODE4NVRTC BEGIN\n%sCODE4NVRTC END\n", final_src.c_str());
  fflush(stdout);
#endif

  std::string single_tile_kernel_name =
    reduce::get_single_tile_kernel_name(input_iterator_name, output_iterator_name, op_name, init, accum_cpp, false);
  std::string single_tile_second_kernel_name = reduce::get_single_tile_kernel_name(
    cccl_type_enum_to_name(accum_t.type, true), output_iterator_name, op_name, init, accum_cpp, true);
  std::string reduction_kernel_name = reduce::get_device_reduce_kernel_name(op_name, input_iterator_name, accum_cpp);
  std::string single_tile_kernel_lowered_name;
  std::string single_tile_second_kernel_lowered_name;
  std::string reduction_kernel_lowered_name;
  std::string nondeterministic_kernel_lowered_name;

  // Only build nondeterministic kernel for CCCL_NOT_GUARANTEED (which requires plus op)
  const bool build_nondeterministic = (determinism == CCCL_NOT_GUARANTEED);
  std::string nondeterministic_kernel_name;
  if (build_nondeterministic)
  {
    nondeterministic_kernel_name = reduce::get_device_reduce_nondeterministic_kernel_name(
      input_iterator_name, output_iterator_name, op_name, accum_cpp, init);
  }

  const std::string arch = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);

  // Build compilation arguments
  std::vector<const char*> args = {
    arch.c_str(),
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    "-rdc=true",
    "-dlto",
    "-DCUB_DISABLE_CDP",
    "-std=c++20"};

  // Add user's extra flags if config is provided
  cccl::detail::extend_args_with_build_config(args, config);

  constexpr size_t num_lto_args   = 2;
  const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

  // Collect all LTO-IRs to be linked.
  nvrtc_linkable_list linkable_list;
  nvrtc_linkable_list_appender appender{linkable_list};

  appender.append_operation(op);
  appender.add_iterator_definition(input_it);
  appender.add_iterator_definition(output_it);

  nvrtc_link_result result =
    begin_linking_nvrtc_program(num_lto_args, lopts)
      ->add_program(nvrtc_translation_unit{final_src.c_str(), name})
      ->add_expression({single_tile_kernel_name})
      ->add_expression({single_tile_second_kernel_name})
      ->add_expression({reduction_kernel_name})
      ->add_expression_if(build_nondeterministic, {nondeterministic_kernel_name})
      ->compile_program({args.data(), args.size()})
      ->get_name({single_tile_kernel_name, single_tile_kernel_lowered_name})
      ->get_name({single_tile_second_kernel_name, single_tile_second_kernel_lowered_name})
      ->get_name({reduction_kernel_name, reduction_kernel_lowered_name})
      ->get_name_if(build_nondeterministic, {nondeterministic_kernel_name, nondeterministic_kernel_lowered_name})
      ->link_program()
      ->add_link_list(linkable_list)
      ->finalize_program();

  cuLibraryLoadData(&build->library, result.data.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
  check(cuLibraryGetKernel(&build->single_tile_kernel, build->library, single_tile_kernel_lowered_name.c_str()));
  check(cuLibraryGetKernel(
    &build->single_tile_second_kernel, build->library, single_tile_second_kernel_lowered_name.c_str()));
  check(cuLibraryGetKernel(&build->reduction_kernel, build->library, reduction_kernel_lowered_name.c_str()));
  if (build_nondeterministic)
  {
    check(cuLibraryGetKernel(
      &build->nondeterministic_atomic_kernel, build->library, nondeterministic_kernel_lowered_name.c_str()));
  }

  build->cc               = cc_major * 10 + cc_minor;
  build->cubin            = (void*) result.data.release();
  build->cubin_size       = result.size;
  build->accumulator_size = accum_t.size;
  build->determinism      = determinism;
  build->runtime_policy   = new cub::detail::reduce::policy_selector{policy_sel};

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_reduce_build(): %s\n", exc.what());
  fflush(stdout);

  return CUDA_ERROR_UNKNOWN;
}

// c.parallel provides two separate reduce functions, one for each determinism
// level, rather than a single function with a runtime switch. This mirrors CUB's
// design, which uses distinct dispatch functions because the host-side logic
// differs between determinism levels. Keeping the functions separate avoids
// branching at runtime to select the appropriate one; cuda.compute selects the
// appropriate function to call at build time.

CUresult cccl_device_reduce(
  cccl_device_reduce_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  cccl_value_t init,
  CUstream stream)
{
  assert(build.determinism == CCCL_RUN_TO_RUN);

  bool pushed    = false;
  CUresult error = CUDA_SUCCESS;
  try
  {
    pushed = try_push_context();

    CUdevice cu_device;
    check(cuCtxGetDevice(&cu_device));

    auto exec_status = cub::detail::reduce::dispatch<void>(
      d_temp_storage,
      *temp_storage_bytes,
      indirect_arg_t{d_in}, // could be indirect_iterator_t, but CUB does not need to increment it
      indirect_arg_t{d_out}, // could be indirect_iterator_t, but CUB does not need to increment it
      static_cast<OffsetT>(num_items),
      indirect_arg_t{op},
      indirect_arg_t{init},
      stream,
      ::cuda::std::identity{},
      *static_cast<cub::detail::reduce::policy_selector*>(build.runtime_policy),
      reduce::reduce_kernel_source{build},
      cub::detail::CudaDriverLauncherFactory{cu_device, build.cc});

    error = static_cast<CUresult>(exec_status);
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_reduce(): %s\n", exc.what());
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

CUresult cccl_device_reduce_nondeterministic(
  cccl_device_reduce_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  cccl_value_t init,
  CUstream stream)
{
  assert(build.determinism == CCCL_NOT_GUARANTEED);

  bool pushed    = false;
  CUresult error = CUDA_SUCCESS;
  try
  {
    pushed = try_push_context();

    CUdevice cu_device;
    check(cuCtxGetDevice(&cu_device));

    auto exec_status = cub::detail::reduce::dispatch_nondeterministic<void>(
      d_temp_storage,
      *temp_storage_bytes,
      indirect_arg_t{d_in}, // could be indirect_iterator_t, but CUB does not need to increment it
      indirect_arg_t{d_out}, // could be indirect_iterator_t, but CUB does not need to increment it
      static_cast<OffsetT>(num_items),
      indirect_arg_t{op},
      indirect_arg_t{init},
      stream,
      ::cuda::std::identity{},
      *static_cast<cub::detail::reduce::policy_selector*>(build.runtime_policy),
      reduce::reduce_kernel_source{build},
      cub::detail::CudaDriverLauncherFactory{cu_device, build.cc});

    error = static_cast<CUresult>(exec_status);
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_reduce(): %s\n", exc.what());
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

CUresult cccl_device_reduce_cleanup(cccl_device_reduce_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  using namespace cub::detail::reduce;
  std::unique_ptr<char[]> cubin(static_cast<char*>(build_ptr->cubin));
  std::unique_ptr<policy_selector> policy(static_cast<policy_selector*>(build_ptr->runtime_policy));
  check(cuLibraryUnload(build_ptr->library));

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_reduce_cleanup(): %s\n", exc.what());
  fflush(stdout);

  return CUDA_ERROR_UNKNOWN;
}

// Backward compatibility wrapper
CUresult cccl_device_reduce_build(
  cccl_device_reduce_build_result_t* build,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  cccl_op_t op,
  cccl_value_t init,
  cccl_determinism_t determinism,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_reduce_build_ex(
    build,
    d_in,
    d_out,
    op,
    init,
    determinism,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    nullptr);
}
