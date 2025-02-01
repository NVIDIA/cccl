#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/detail/choose_offset.cuh>
#include <cub/detail/launcher/cuda_driver.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_temporary_storage.cuh>
#include <cub/util_type.cuh>

#include <format>
#include <iostream>
#include <regex>
#include <string>
#include <type_traits>

#include "cub/util_device.cuh"
#include "kernels/iterators.h"
#include "kernels/operators.h"
#include "util/context.h"
#include "util/errors.h"
#include "util/indirect_arg.h"
#include "util/types.h"
#include <cccl/c/scan.h>
#include <nvrtc.h>
#include <nvrtc/command_list.h>

struct op_wrapper;
struct device_scan_policy;
using OffsetT = unsigned long long;
static_assert(std::is_same_v<cub::detail::choose_offset_t<OffsetT>, OffsetT>, "OffsetT must be size_t");

struct input_iterator_state_t;
struct output_iterator_t;

namespace scan
{

struct scan_runtime_tuning_policy
{
  int block_size;
  int items_per_thread;
  cub::CacheLoadModifier load_modifier;

  scan_runtime_tuning_policy Scan() const
  {
    return *this;
  }

  int ItemsPerThread() const
  {
    return items_per_thread;
  }

  int BlockThreads() const
  {
    return block_size;
  }

  cub::CacheLoadModifier LoadModifier() const
  {
    return load_modifier;
  }

  void CheckLoadModifier() const
  {
    if (LoadModifier() == cub::CacheLoadModifier::LOAD_LDG)
    {
      throw std::runtime_error("The memory consistency model does not apply to texture "
                               "accesses");
    }
  }
};

template <typename Tuning, int N>
Tuning find_tuning(int cc, const Tuning (&tunings)[N])
{
  for (const Tuning& tuning : tunings)
  {
    if (cc >= tuning.cc)
    {
      return tuning;
    }
  }

  return tunings[N - 1];
}

scan_runtime_tuning_policy get_policy(int /*cc*/, cccl_type_info /*accumulator_type*/)
{
  // TODO: we should update this once we figure out a way to reuse
  // tuning logic from C++. Alternately, we should implement
  // something better than a hardcoded default:
  return {128, 4, cub::LOAD_DEFAULT};
}

static cccl_type_info get_accumulator_type(cccl_op_t /*op*/, cccl_iterator_t /*input_it*/, cccl_value_t init)
{
  // TODO Should be decltype(op(init, *input_it)) but haven't implemented type arithmetic yet
  //      so switching back to the old accumulator type logic for now
  return init.type;
}

std::string get_input_iterator_name()
{
  std::string iterator_t;
  check(nvrtcGetTypeName<input_iterator_state_t>(&iterator_t));
  return iterator_t;
}

std::string get_output_iterator_name()
{
  std::string iterator_t;
  check(nvrtcGetTypeName<output_iterator_t>(&iterator_t));
  return iterator_t;
}

std::string
get_init_kernel_name(cccl_iterator_t input_it, cccl_iterator_t /*output_it*/, cccl_op_t op, cccl_value_t init)
{
  const cccl_type_info accum_t  = scan::get_accumulator_type(op, input_it, init);
  const std::string accum_cpp_t = cccl_type_enum_to_name(accum_t.type);
  return std::format("cub::detail::scan::DeviceScanInitKernel<cub::ScanTileState<{0}>>", accum_cpp_t);
}

std::string get_scan_kernel_name(cccl_iterator_t input_it, cccl_iterator_t output_it, cccl_op_t op, cccl_value_t init)
{
  std::string chained_policy_t;
  check(nvrtcGetTypeName<device_scan_policy>(&chained_policy_t));

  const cccl_type_info accum_t  = scan::get_accumulator_type(op, input_it, init);
  const std::string accum_cpp_t = cccl_type_enum_to_name(accum_t.type);
  const std::string input_iterator_t =
    (input_it.type == cccl_iterator_kind_t::pointer //
       ? cccl_type_enum_to_name(input_it.value_type.type, true) //
       : scan::get_input_iterator_name());
  const std::string output_iterator_t =
    output_it.type == cccl_iterator_kind_t::pointer //
      ? cccl_type_enum_to_name(output_it.value_type.type, true) //
      : scan::get_output_iterator_name();
  const std::string init_t = cccl_type_enum_to_name(init.type.type);

  std::string offset_t;
  check(nvrtcGetTypeName<OffsetT>(&offset_t));

  std::string scan_op_t;
  check(nvrtcGetTypeName<op_wrapper>(&scan_op_t));

  auto tile_state_t = std::format("cub::ScanTileState<{0}>", accum_cpp_t);
  return std::format(
    "cub::detail::scan::DeviceScanKernel<{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}>",
    chained_policy_t,
    input_iterator_t,
    output_iterator_t,
    tile_state_t,
    scan_op_t,
    init_t,
    offset_t,
    accum_cpp_t,
    "false", // for now, always exclusive
    init_t);
}

static constexpr auto ptx_u64_assignment_regex = R"(\.visible\s+\.global\s+\.align\s+\d+\s+\.u64\s+{}\s*=\s*(\d+);)";

size_t find_size_t(char* ptx, std::string_view name)
{
  std::regex regex(std::format(ptx_u64_assignment_regex, name));
  std::cmatch match;
  if (std::regex_search(ptx, match, regex))
  {
    auto result = std::stoi(match[1].str());
    return result;
  }
  else
  {
    throw std::runtime_error(std::format("Could not find {} in PTX code", name));
  }
}

size_t find_size_t(char* ptx, std::string_view name, size_t default_value)
{
  std::regex regex(std::format(ptx_u64_assignment_regex, name));
  std::cmatch match;
  if (std::regex_search(ptx, match, regex))
  {
    return std::stoi(match[1].str());
  }
  else
  {
    return default_value;
  }
}

struct scan_tile_state
{
  // scan_tile_state implements the same (host) interface as cub::ScanTileStateT, except
  // that it accepts the acummulator type as a runtime parameter rather than being
  // templated on it.
  //
  // Both specializations ScanTileStateT<T, true> and ScanTileStateT<T, false> - where the
  // bool parameter indicates whether `T` is primitive - are combined into a single type.

  void* d_tile_status; // d_tile_descriptors
  void* d_tile_partial;
  void* d_tile_inclusive;

  size_t description_bytes_per_tile;
  size_t payload_bytes_per_tile;

  scan_tile_state(size_t description_bytes_per_tile, size_t payload_bytes_per_tile)
      : d_tile_status(nullptr)
      , d_tile_partial(nullptr)
      , d_tile_inclusive(nullptr)
      , description_bytes_per_tile(description_bytes_per_tile)
      , payload_bytes_per_tile(payload_bytes_per_tile)
  {}

  cudaError_t Init(int num_tiles, void* d_temp_storage, size_t temp_storage_bytes)
  {
    void* allocations[3] = {};
    auto status          = cub::detail::tile_state_init(
      description_bytes_per_tile, payload_bytes_per_tile, num_tiles, d_temp_storage, temp_storage_bytes, allocations);
    if (status != cudaSuccess)
    {
      return status;
    }
    d_tile_status    = allocations[0];
    d_tile_partial   = allocations[1];
    d_tile_inclusive = allocations[2];
    return cudaSuccess;
  }

  cudaError_t AllocationSize(int num_tiles, size_t& temp_storage_bytes) const
  {
    temp_storage_bytes =
      cub::detail::tile_state_allocation_size(description_bytes_per_tile, payload_bytes_per_tile, num_tiles);
    return cudaSuccess;
  }
};

template <auto* GetPolicy>
struct dynamic_scan_policy_t
{
  using MaxPolicy = dynamic_scan_policy_t;

  template <typename F>
  cudaError_t Invoke(int device_ptx_version, F& op)
  {
    return op.template Invoke<scan_runtime_tuning_policy>(GetPolicy(device_ptx_version, accumulator_type));
  }

  cccl_type_info accumulator_type;
};

struct scan_kernel_source
{
  cccl_device_scan_build_result_t& build;

  std::size_t AccumSize() const
  {
    return build.accumulator_type.size;
  }
  CUkernel InitKernel() const
  {
    return build.init_kernel;
  }
  CUkernel ScanKernel() const
  {
    return build.scan_kernel;
  }
  scan_tile_state TileState()
  {
    auto result = (reinterpret_cast<scan_tile_state*>(build.tile_state));
    return *result;
  }
};

} // namespace scan

extern "C" CCCL_C_API CUresult cccl_device_scan_build(
  cccl_device_scan_build_result_t* build,
  cccl_iterator_t input_it,
  cccl_iterator_t output_it,
  cccl_op_t op,
  cccl_value_t init,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path) noexcept
{
  CUresult error = CUDA_SUCCESS;

  try
  {
    const char* name = "test";

    const int cc                 = cc_major * 10 + cc_minor;
    const cccl_type_info accum_t = scan::get_accumulator_type(op, input_it, init);
    const auto policy            = scan::get_policy(cc, accum_t);
    const auto accum_cpp         = cccl_type_enum_to_string(accum_t.type);
    const auto input_it_value_t  = cccl_type_enum_to_string(input_it.value_type.type);
    const auto offset_t          = cccl_type_enum_to_string(cccl_type_enum::UINT64);

    const std::string input_iterator_src  = make_kernel_input_iterator(offset_t, input_it_value_t, input_it);
    const std::string output_iterator_src = make_kernel_output_iterator(offset_t, accum_cpp, output_it);

    const std::string op_src = make_kernel_user_binary_operator(accum_cpp, op);

    const std::string src = std::format(
      "#include <cub/block/block_scan.cuh>\n"
      "#include <cub/device/dispatch/kernels/scan.cuh>\n"
      "#include <cub/agent/single_pass_scan_operators.cuh>\n"
      "struct __align__({1}) storage_t {{\n"
      "  char data[{0}];\n"
      "}};\n"
      "{4}\n"
      "{5}\n"
      "struct agent_policy_t {{\n"
      "  static constexpr int ITEMS_PER_THREAD = {2};\n"
      "  static constexpr int BLOCK_THREADS = {3};\n"
      "  static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM = cub::BLOCK_LOAD_WARP_TRANSPOSE;\n"
      "  static constexpr cub::CacheLoadModifier LOAD_MODIFIER = cub::LOAD_DEFAULT;\n"
      "  static constexpr cub::BlockStoreAlgorithm STORE_ALGORITHM = cub::BLOCK_STORE_WARP_TRANSPOSE;\n"
      "  static constexpr cub::BlockScanAlgorithm SCAN_ALGORITHM = cub::BLOCK_SCAN_WARP_SCANS;\n"
      "  struct detail {{\n"
      "    using delay_constructor_t = cub::detail::default_delay_constructor_t<{7}>;\n"
      "  }};\n"
      "}};\n"
      "struct device_scan_policy {{\n"
      "  struct ActivePolicy {{\n"
      "    using ScanPolicyT = agent_policy_t;\n"
      "  }};\n"
      "}};\n"
      "{6};\n",
      input_it.value_type.size, // 0
      input_it.value_type.alignment, // 1
      policy.items_per_thread, // 2
      policy.block_size, // 3
      input_iterator_src, // 4
      output_iterator_src, // 5
      op_src, // 6
      accum_cpp); // 7

#if false // CCCL_DEBUGGING_SWITCH
    fflush(stderr);
    printf("\nCODE4NVRTC BEGIN\n%sCODE4NVRTC END\n", src.c_str());
    fflush(stdout);
#endif

    std::string init_kernel_name = scan::get_init_kernel_name(input_it, output_it, op, init);
    std::string scan_kernel_name = scan::get_scan_kernel_name(input_it, output_it, op, init);
    std::string init_kernel_lowered_name;
    std::string scan_kernel_lowered_name;

    const std::string arch = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);

    constexpr size_t num_args  = 7;
    const char* args[num_args] = {arch.c_str(), cub_path, thrust_path, libcudacxx_path, ctk_path, "-rdc=true", "-dlto"};

    constexpr size_t num_lto_args   = 2;
    const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

    // Collect all LTO-IRs to be linked.
    nvrtc_ltoir_list ltoir_list;
    auto ltoir_list_append = [&ltoir_list](nvrtc_ltoir lto) {
      if (lto.ltsz)
      {
        ltoir_list.push_back(std::move(lto));
      }
    };
    ltoir_list_append({op.ltoir, op.ltoir_size});
    if (cccl_iterator_kind_t::iterator == input_it.type)
    {
      ltoir_list_append({input_it.advance.ltoir, input_it.advance.ltoir_size});
      ltoir_list_append({input_it.dereference.ltoir, input_it.dereference.ltoir_size});
    }
    if (cccl_iterator_kind_t::iterator == output_it.type)
    {
      ltoir_list_append({output_it.advance.ltoir, output_it.advance.ltoir_size});
      ltoir_list_append({output_it.dereference.ltoir, output_it.dereference.ltoir_size});
    }

    nvrtc_cubin result =
      make_nvrtc_command_list()
        .add_program(nvrtc_translation_unit{src.c_str(), name})
        .add_expression({init_kernel_name})
        .add_expression({scan_kernel_name})
        .compile_program({args, num_args})
        .get_name({init_kernel_name, init_kernel_lowered_name})
        .get_name({scan_kernel_name, scan_kernel_lowered_name})
        .cleanup_program()
        .add_link_list(ltoir_list)
        .finalize_program(num_lto_args, lopts);

    cuLibraryLoadData(&build->library, result.cubin.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
    check(cuLibraryGetKernel(&build->init_kernel, build->library, init_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(&build->scan_kernel, build->library, scan_kernel_lowered_name.c_str()));

    constexpr size_t num_ptx_args           = 5;
    const char* ptx_args[num_ptx_args]      = {arch.c_str(), cub_path, libcudacxx_path, "-rdc=true", "-dlto"};
    constexpr size_t num_ptx_lto_args       = 3;
    const char* ptx_lopts[num_ptx_lto_args] = {"-lto", arch.c_str(), "-ptx"};

    size_t description_bytes_per_tile;
    size_t payload_bytes_per_tile;
    std::string ptx_src = std::format(
      "#include <cub/agent/single_pass_scan_operators.cuh>\n"
      "#include <cub/util_type.cuh>\n"
      "struct __align__({1}) storage_t {{\n"
      "  char data[{0}];\n"
      "}};\n"
      "__device__ size_t description_bytes_per_tile = cub::ScanTileState<{2}>::description_bytes_per_tile;\n"
      "__device__ size_t payload_bytes_per_tile = cub::ScanTileState<{2}>::payload_bytes_per_tile;\n",
      accum_t.size,
      accum_t.alignment,
      accum_cpp);
    auto compile_result =
      make_nvrtc_command_list()
        .add_program(nvrtc_translation_unit{ptx_src.c_str(), "tile_state_info"})
        .compile_program({ptx_args, num_ptx_args})
        .cleanup_program()
        .finalize_program(num_ptx_lto_args, ptx_lopts);
    auto ptx_code              = compile_result.cubin.get();
    description_bytes_per_tile = scan::find_size_t(ptx_code, "description_bytes_per_tile");
    payload_bytes_per_tile     = scan::find_size_t(ptx_code, "payload_bytes_per_tile", 0);
    auto tile_state = std::make_unique<scan::scan_tile_state>(description_bytes_per_tile, payload_bytes_per_tile);

    build->cc               = cc;
    build->cubin            = (void*) result.cubin.release();
    build->cubin_size       = result.size;
    build->accumulator_type = accum_t;
    build->tile_state       = (void*) tile_state.release();
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_scan_build(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }

  return error;
}

extern "C" CCCL_C_API CUresult cccl_device_scan(
  cccl_device_scan_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  unsigned long long num_items,
  cccl_op_t op,
  cccl_value_t init,
  CUstream stream) noexcept
{
  bool pushed    = false;
  CUresult error = CUDA_SUCCESS;
  try
  {
    pushed = try_push_context();

    CUdevice cu_device;
    check(cuCtxGetDevice(&cu_device));
    auto cuda_error = cub::DispatchScan<
      indirect_arg_t,
      indirect_arg_t,
      indirect_arg_t,
      indirect_arg_t,
      ::cuda::std::size_t,
      void,
      scan::dynamic_scan_policy_t<&scan::get_policy>,
      false,
      scan::scan_kernel_source,
      cub::detail::CudaDriverLauncherFactory>::
      Dispatch(
        d_temp_storage,
        *temp_storage_bytes,
        d_in,
        d_out,
        op,
        init,
        num_items,
        stream,
        {build},
        cub::detail::CudaDriverLauncherFactory{cu_device, build.cc},
        {scan::get_accumulator_type(op, d_in, init)});
    if (cuda_error != cudaSuccess)
    {
      const char* errorString = cudaGetErrorString(cuda_error); // Get the error string
      std::cerr << "CUDA error: " << errorString << std::endl;
    }
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_scan(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }
  if (pushed)
  {
    CUcontext cu_context;
    cuCtxPopCurrent(&cu_context);
  }
  return error;
}

extern "C" CCCL_C_API CUresult cccl_device_scan_cleanup(cccl_device_scan_build_result_t* bld_ptr) noexcept
{
  try
  {
    if (bld_ptr == nullptr)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }

    std::unique_ptr<char[]> cubin(reinterpret_cast<char*>(bld_ptr->cubin));
    check(cuLibraryUnload(bld_ptr->library));

    std::unique_ptr<scan::scan_tile_state> tile_state(reinterpret_cast<scan::scan_tile_state*>(bld_ptr->tile_state));
    tile_state.reset();
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_scan_cleanup(): %s\n", exc.what());
    fflush(stdout);
    return CUDA_ERROR_UNKNOWN;
  }

  return CUDA_SUCCESS;
}
