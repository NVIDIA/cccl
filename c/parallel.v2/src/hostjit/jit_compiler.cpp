#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include <hostjit/jit_compiler.hpp>

#ifdef _WIN32
#  include <process.h>
#else
#  include <unistd.h>
#endif

namespace
{
static constexpr const char* pch_preamble_source =
  "#include <cuda_runtime.h>\n"
  "#include <cuda/std/iterator>\n"
  "#include <cuda/std/functional>\n"
  "#include <cuda/functional>\n"
  "#include <cub/device/device_adjacent_difference.cuh>\n"
  "#include <cub/device/device_copy.cuh>\n"
  "#include <cub/device/device_find.cuh>\n"
  "#include <cub/device/device_for.cuh>\n"
  "#include <cub/device/device_histogram.cuh>\n"
  "#include <cub/device/device_merge.cuh>\n"
  "#include <cub/device/device_merge_sort.cuh>\n"
  "#include <cub/device/device_partition.cuh>\n"
  "#include <cub/device/device_radix_sort.cuh>\n"
  "#include <cub/device/device_reduce.cuh>\n"
  "#include <cub/device/device_scan.cuh>\n"
  "#include <cub/device/device_segmented_radix_sort.cuh>\n"
  "#include <cub/device/device_segmented_scan.cuh>\n"
  "#include <cub/device/device_segmented_sort.cuh>\n"
  "#include <cub/device/device_select.cuh>\n"
  "#include <cub/device/device_transform.cuh>\n";

std::filesystem::path get_pch_cache_dir()
{
  auto dir = std::filesystem::temp_directory_path() / "hostjit_pch";
  std::filesystem::create_directories(dir);
  return dir;
}

std::string get_pch_path(const std::string& kind, int sm_version)
{
  return (get_pch_cache_dir() / (kind + "_sm" + std::to_string(sm_version) + ".pch")).string();
}

std::string get_pch_source_path(const std::string& kind, int sm_version)
{
  return (get_pch_cache_dir() / (kind + "_sm" + std::to_string(sm_version) + "_preamble.cu")).string();
}

bool create_pch_if_needed(
  hostjit::CompilerConfig config,
  libnvccPCHKind kind,
  const std::string& kind_name,
  std::string& diagnostics,
  std::string& pch_path)
{
  pch_path = get_pch_path(kind_name, config.sm_version);
  if (std::filesystem::exists(pch_path))
  {
    return true;
  }

  config.enable_pch = false;
  config.device_pch_path.clear();
  config.host_pch_path.clear();

  std::vector<std::string> options;
  config.appendCommandLineArguments(options);
  auto option_ptrs = hostjit::detail::make_libnvcc_option_ptrs(options);

  hostjit::detail::LibnvccProgramGuard program;
  auto create_result = libnvccCreateProgram(&program.program, pch_preamble_source, "hostjit_preamble.cu");
  if (create_result != LIBNVCC_SUCCESS)
  {
    diagnostics += "Failed to create libnvcc PCH program: ";
    diagnostics += libnvccGetErrorString(create_result);
    diagnostics += "\n";
    pch_path.clear();
    return false;
  }

  auto source_path = get_pch_source_path(kind_name, config.sm_version);
  auto pch_result  = libnvccCreatePCH(
    program.program,
    kind,
    source_path.c_str(),
    pch_path.c_str(),
    static_cast<int>(option_ptrs.size()),
    option_ptrs.empty() ? nullptr : option_ptrs.data());
  if (pch_result != LIBNVCC_SUCCESS)
  {
    diagnostics += kind_name + " PCH generation failed: " + hostjit::detail::get_libnvcc_program_log(program.program);
    diagnostics += "\n";
    pch_path.clear();
    return false;
  }
  return true;
}

hostjit::CompilerConfig prepare_pch_config(const hostjit::CompilerConfig& config, std::string& diagnostics)
{
  hostjit::CompilerConfig prepared = config;
  prepared.device_pch_path.clear();
  prepared.host_pch_path.clear();

  if (!prepared.enable_pch)
  {
    return prepared;
  }

  std::string device_pch_path;
  if (create_pch_if_needed(prepared, LIBNVCC_PCH_DEVICE, "device", diagnostics, device_pch_path))
  {
    prepared.device_pch_path = std::move(device_pch_path);
  }

  std::string host_pch_path;
  if (create_pch_if_needed(prepared, LIBNVCC_PCH_HOST, "host", diagnostics, host_pch_path))
  {
    prepared.host_pch_path = std::move(host_pch_path);
  }

  return prepared;
}

bool read_file(const std::string& path, std::vector<char>& out)
{
  std::ifstream f(path, std::ios::binary);
  if (!f)
  {
    return false;
  }
  out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  return true;
}
} // anonymous namespace

namespace hostjit
{
JITCompiler::JITCompiler()
    : config_(detectDefaultConfig())
{}

JITCompiler::JITCompiler(const CompilerConfig& config)
    : config_(config)
{}

JITCompiler::~JITCompiler()
{
  cleanup();
}

bool JITCompiler::compile(const std::string& source_code)
{
  std::string config_error;
  if (!validateConfig(config_, &config_error))
  {
    last_error_ = "Configuration error: " + config_error;
    return false;
  }

  cleanup();

  temp_dir_ = createTempDirectory();
  if (temp_dir_.empty())
  {
    last_error_ = "Failed to create temporary directory";
    return false;
  }

  std::string pch_diagnostics;
  CompilerConfig libnvcc_config = prepare_pch_config(config_, pch_diagnostics);
  if (config_.verbose && !pch_diagnostics.empty())
  {
    std::cout << pch_diagnostics;
  }

  std::vector<std::string> options;
  libnvcc_config.appendCommandLineArguments(options);
  auto option_ptrs = hostjit::detail::make_libnvcc_option_ptrs(options);

  hostjit::detail::LibnvccProgramGuard program;
  auto create_result = libnvccCreateProgram(&program.program, source_code.c_str(), "input.cu");
  if (create_result != LIBNVCC_SUCCESS)
  {
    last_error_ = std::string("Failed to create libnvcc program: ") + libnvccGetErrorString(create_result);
    removeTempDirectory();
    return false;
  }

  std::string obj_path   = temp_dir_ + "/cuda_code.o";
  std::string cubin_path = temp_dir_ + "/device.cubin";
  auto compile_result    = libnvccCompileProgramToObject(
    program.program,
    obj_path.c_str(),
    cubin_path.c_str(),
    static_cast<int>(option_ptrs.size()),
    option_ptrs.empty() ? nullptr : option_ptrs.data());
  auto compile_log = hostjit::detail::get_libnvcc_program_log(program.program);

  if (compile_result != LIBNVCC_SUCCESS)
  {
    last_error_ = "Compilation failed:\n" + compile_log;
    removeTempDirectory();
    return false;
  }

  cubin_.clear();
  if (!read_file(cubin_path, cubin_))
  {
    last_error_ = "Compilation failed: generated cubin could not be read";
    removeTempDirectory();
    return false;
  }

  if (config_.verbose)
  {
    std::cout << "Compilation diagnostics:\n" << compile_log << "\n";
  }

#ifdef _WIN32
  std::string lib_path = temp_dir_ + "/cuda_code.dll";
#else
  std::string lib_path = temp_dir_ + "/libcuda_code.so";
#endif
  const char* object_files[] = {obj_path.c_str()};
  auto link_result           = libnvccLinkToSharedLibrary(
    program.program,
    1,
    object_files,
    lib_path.c_str(),
    static_cast<int>(option_ptrs.size()),
    option_ptrs.empty() ? nullptr : option_ptrs.data());
  auto link_log = hostjit::detail::get_libnvcc_program_log(program.program);

  if (link_result != LIBNVCC_SUCCESS)
  {
    last_error_ = "Linking failed:\n" + link_log;
    removeTempDirectory();
    return false;
  }

  if (config_.verbose)
  {
    std::cout << "Linking diagnostics:\n" << link_log << "\n";
  }

  if (!library_.load(lib_path))
  {
    last_error_ = "Failed to load library: " + library_.getLastError();
    removeTempDirectory();
    return false;
  }

  if (config_.verbose)
  {
    std::cout << "Successfully loaded library: " << lib_path << "\n";
  }

  last_error_.clear();
  return true;
}

void JITCompiler::cleanup()
{
  library_.unload();

  if (!config_.keep_artifacts)
  {
    removeTempDirectory();
  }

  last_error_.clear();
}

std::string JITCompiler::createTempDirectory()
{
  std::filesystem::path base_tmp_dir;

#ifdef _WIN32
  const char* tmp_dir = std::getenv("TEMP");
  if (!tmp_dir)
  {
    tmp_dir = std::getenv("TMP");
  }
  if (tmp_dir)
  {
    base_tmp_dir = tmp_dir;
  }
  else
  {
    base_tmp_dir = std::filesystem::temp_directory_path();
  }
#else
  const char* tmp_dir = std::getenv("TMPDIR");
  if (tmp_dir)
  {
    base_tmp_dir = tmp_dir;
  }
  else
  {
    base_tmp_dir = "/tmp";
  }
#endif

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 999999);

#ifdef _WIN32
  int pid = _getpid();
#else
  int pid = getpid();
#endif

  for (int attempt = 0; attempt < 10; ++attempt)
  {
    std::string dir_name            = "hostjit_" + std::to_string(pid) + "_" + std::to_string(dis(gen));
    std::filesystem::path full_path = base_tmp_dir / dir_name;

    std::error_code ec;
    if (std::filesystem::create_directories(full_path, ec) && !ec)
    {
      return full_path.string();
    }
  }

  return "";
}

void JITCompiler::removeTempDirectory()
{
  if (temp_dir_.empty())
  {
    return;
  }

  try
  {
    if (std::filesystem::exists(temp_dir_))
    {
      std::filesystem::remove_all(temp_dir_);
    }
  }
  catch (const std::filesystem::filesystem_error& e)
  {
    if (config_.verbose)
    {
      std::cerr << "Warning: Failed to remove temporary directory: " << e.what() << "\n";
    }
  }

  temp_dir_.clear();
}
} // namespace hostjit
