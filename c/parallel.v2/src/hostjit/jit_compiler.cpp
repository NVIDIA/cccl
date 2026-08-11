#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
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

// Parse CCCL_PCH_CACHE_MAXSIZE. Accepts a byte count with an optional binary
// suffix (K/M/G, or KiB/MiB/GiB). 0 disables eviction.
//
// The default is 1 GiB, which holds roughly a dozen configurations. CUDA's
// analogous CUDA_CACHE_MAXSIZE defaults to 256 MiB, but its entries are cubins
// measured in kilobytes; a single PCH here is tens of megabytes, so the same
// default would hold only three configurations and thrash.
std::uintmax_t pch_cache_max_size()
{
  constexpr std::uintmax_t default_bytes = 1024ull * 1024 * 1024;

  const char* v = std::getenv("CCCL_PCH_CACHE_MAXSIZE");
  if (!v || v[0] == '\0')
  {
    return default_bytes;
  }

  errno                 = 0;
  char* end             = nullptr;
  const long long value = std::strtoll(v, &end, 10);
  // strtoll signals overflow with ERANGE and still returns LLONG_MAX, which
  // would otherwise be scaled by the suffix below and wrap.
  if (end == v || value < 0 || errno == ERANGE)
  {
    return default_bytes;
  }

  std::uintmax_t multiplier = 1;
  while (end && *end == ' ')
  {
    ++end;
  }
  if (end && *end != '\0')
  {
    switch (std::tolower(static_cast<unsigned char>(*end)))
    {
      case 'k':
        multiplier = 1024ull;
        break;
      case 'm':
        multiplier = 1024ull * 1024;
        break;
      case 'g':
        multiplier = 1024ull * 1024 * 1024;
        break;
      default:
        return default_bytes;
    }
  }
  // The scaled value must not wrap: a product of exactly 2^64 comes back as 0,
  // which this function's callers read as "eviction disabled" and would let the
  // cache grow without bound. Treat anything that does not fit as a typo and
  // fall back to the default, the same as unparsable input.
  const auto scalar = static_cast<std::uintmax_t>(value);
  if (multiplier > 1 && scalar > std::numeric_limits<std::uintmax_t>::max() / multiplier)
  {
    return default_bytes;
  }
  return scalar * multiplier;
}

// Evict least-recently-used entries until the cache fits under its size cap,
// and drop temp files abandoned by a killed generation.
//
// Bounding disk rather than age follows CUDA's JIT cache (CUDA_CACHE_MAXSIZE)
// and ccache (max_size): size is the resource users care about, and an age
// limit bounds nothing -- a dozen configurations used inside the window still
// costs a gigabyte. Recency is mtime, refreshed on use by touch_pch_entry, so
// this is LRU in the same approximate sense ccache documents.
//
// Called once per build that generated something, never on cache-directory
// resolution. Generation is the only moment the cache grows, and a directory
// scan is free beside the multi-second compile that just finished; doing it per
// process would put a scan in the hot path of every import.
//
// `in_use` holds the entries this build is about to compile against. They must
// be excluded: evicting one after its path has been handed to the in-flight
// build makes that build fail its PCH lookup, fall into the retry path, and
// discard the *other* entry too -- so a cap that should have dropped one entry
// empties the cache instead.
//
// Best-effort throughout: failing to evict is never a reason to fail a build.
void evict_pch_cache(const std::filesystem::path& dir, const std::vector<std::string>& in_use)
{
  // A .pch and the _preamble.cu it was built from are one logical entry: the
  // PCH records the preamble's path, so evicting them separately would leave
  // an entry that clang rejects on the next build.
  struct Entry
  {
    std::filesystem::path pch;
    std::filesystem::path preamble;
    std::filesystem::file_time_type mtime;
    std::uintmax_t bytes;
  };

  std::vector<Entry> entries;
  std::uintmax_t total = 0;

  // Temp files older than an hour belong to a generation that died; a live one
  // is seconds old. Sweeping them here rather than in a separate pass keeps the
  // cache to a single maintenance point.
  const auto temp_cutoff = std::filesystem::file_time_type::clock::now() - std::chrono::hours(1);

  std::error_code ec;
  for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec))
  {
    std::error_code entry_ec;
    if (!it->is_regular_file(entry_ec) || entry_ec)
    {
      continue;
    }
    const auto& path = it->path();

    if (path.extension() == ".tmp")
    {
      const auto mtime = it->last_write_time(entry_ec);
      if (!entry_ec && mtime < temp_cutoff)
      {
        std::error_code remove_ec;
        std::filesystem::remove(path, remove_ec);
      }
      continue;
    }

    if (path.extension() != ".pch")
    {
      continue; // preambles are accounted for with their .pch
    }

    const auto mtime = it->last_write_time(entry_ec);
    if (entry_ec)
    {
      continue;
    }
    const auto size = it->file_size(entry_ec);
    if (entry_ec)
    {
      continue;
    }

    auto preamble = path;
    preamble.replace_extension();
    preamble += "_preamble.cu";

    std::error_code preamble_ec;
    const auto preamble_size   = std::filesystem::file_size(preamble, preamble_ec);
    const std::uintmax_t bytes = size + (preamble_ec ? 0 : preamble_size);

    total += bytes;
    const bool protected_entry = std::find(in_use.begin(), in_use.end(), path.string()) != in_use.end();
    if (!protected_entry)
    {
      entries.push_back(Entry{path, preamble, mtime, bytes});
    }
  }

  const std::uintmax_t cap = pch_cache_max_size();
  if (cap == 0 || total <= cap)
  {
    return;
  }

  std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
    return a.mtime < b.mtime;
  });

  for (const auto& entry : entries)
  {
    if (total <= cap)
    {
      break;
    }
    std::error_code remove_ec;
    std::filesystem::remove(entry.pch, remove_ec);
    std::filesystem::remove(entry.preamble, remove_ec);
    total -= entry.bytes;
  }
}

// Resolve the PCH cache directory, once per process. CCCL_PCH_CACHE_DIR wins on
// every platform and is used verbatim, with no subdirectory appended, so a
// caller (CI, a test's tmp_path) gets exactly the path it asked for. The
// remaining candidates differ by platform.
//
// POSIX:
//   1. $XDG_CACHE_HOME/cccl/hostjit_pch
//   2. $HOME/.cache/cccl/hostjit_pch
//   3. <temp>/hostjit_pch_<uid>  -- uid-scoped, so two users on one machine
//      cannot land on the same directory and fight over its permissions.
//
// Windows (XDG_CACHE_HOME and HOME are not consulted):
//   1. %LOCALAPPDATA%\cccl\hostjit_pch
//   2. <temp>\hostjit_pch  -- no uid suffix; the per-user temp directory
//      already scopes it.
//
// This is a persistent cache of tens of megabytes, not scratch, so the system
// temp directory is a poor default: it is shared, and whoever creates it first
// owns it.
//
// Returns an empty path when nothing is usable. Callers treat that as "PCH
// unavailable" and build without one; this never throws, because an unwritable
// cache location must not be able to fail a build.
const std::filesystem::path& get_pch_cache_dir()
{
  static const std::filesystem::path resolved = [] {
    std::vector<std::filesystem::path> candidates;

    if (const char* explicit_dir = std::getenv("CCCL_PCH_CACHE_DIR"); explicit_dir && explicit_dir[0] != '\0')
    {
      candidates.emplace_back(explicit_dir);
    }
    else
    {
#ifdef _WIN32
      if (const char* local_app = std::getenv("LOCALAPPDATA"); local_app && local_app[0] != '\0')
      {
        candidates.emplace_back(std::filesystem::path(local_app) / "cccl" / "hostjit_pch");
      }
#else
      if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && xdg[0] != '\0')
      {
        candidates.emplace_back(std::filesystem::path(xdg) / "cccl" / "hostjit_pch");
      }
      if (const char* home = std::getenv("HOME"); home && home[0] != '\0')
      {
        candidates.emplace_back(std::filesystem::path(home) / ".cache" / "cccl" / "hostjit_pch");
      }
#endif
      std::error_code temp_ec;
      const auto temp = std::filesystem::temp_directory_path(temp_ec);
      if (!temp_ec)
      {
#ifdef _WIN32
        candidates.emplace_back(temp / "hostjit_pch");
#else
        candidates.emplace_back(temp / ("hostjit_pch_" + std::to_string(static_cast<unsigned>(::getuid()))));
#endif
      }
    }

    for (const auto& dir : candidates)
    {
      std::error_code ec;
      std::filesystem::create_directories(dir, ec);
      // Both overloads must be the non-throwing ones: this runs in a static
      // initializer, so an escaping filesystem_error would both fail the build
      // and leave the static uninitialized for the next caller to retry.
      std::error_code is_dir_ec;
      if (ec && !std::filesystem::is_directory(dir, is_dir_ec))
      {
        continue;
      }
      // create_directories succeeding does not prove we can write into an
      // already-existing directory owned by someone else, so probe.
      const auto probe = dir / ".cccl_write_probe";
      {
        std::ofstream f(probe, std::ios::binary);
        if (!f)
        {
          continue;
        }
      }
      std::error_code remove_ec;
      std::filesystem::remove(probe, remove_ec);
      return dir;
    }
    return std::filesystem::path{};
  }();
  return resolved;
}

// 64-bit FNV-1a over everything that determines the contents of a PCH: the
// preamble text, the PCH kind, and every option handed to libnvcc.
//
// This is the cache key. Two compiles that agree on it can share a PCH; two
// that disagree must not, because clang validates a PCH against the command
// line it was built with. Keying on (kind, sm_version) alone -- as this did
// before -- means a source-tree build and an installed wheel, which differ in
// their include paths, collide on one entry and break each other.
//
// Header *contents* are deliberately not hashed: reading every CUB header to
// digest it would cost about as much as the parse the PCH exists to avoid. An
// in-place CCCL upgrade therefore keeps the same key and leaves a stale PCH
// behind; clang's own size/mtime validation rejects it, and the
// retry-without-PCH path in JITCompiler::compile recovers.
std::string hash_pch_options(const std::vector<std::string>& options, const std::string& kind_name)
{
  std::uint64_t h = 1469598103934665603ull;
  auto mix        = [&h](std::string_view s) {
    for (unsigned char c : s)
    {
      h ^= c;
      h *= 1099511628211ull;
    }
    h ^= '\0';
    h *= 1099511628211ull;
  };

  mix(pch_preamble_source);
  mix(kind_name);
  for (const auto& option : options)
  {
    mix(option);
  }

  // Fixed-width lowercase hex, so cache filenames sort and glob predictably.
  std::string out(16, '0');
  for (int i = 15; i >= 0; --i)
  {
    out[static_cast<size_t>(i)] = "0123456789abcdef"[h & 0xf];
    h >>= 4;
  }
  return out;
}

std::string get_pch_path(const std::filesystem::path& dir, const std::string& kind, const std::string& key)
{
  return (dir / (kind + "_" + key + ".pch")).string();
}

std::string get_pch_source_path(const std::filesystem::path& dir, const std::string& kind, const std::string& key)
{
  return (dir / (kind + "_" + key + "_preamble.cu")).string();
}

// Refresh a cache entry's mtime so evict_pch_cache treats it as recently used,
// but only when it is already more than a day stale -- otherwise every build
// would pay a pointless write. Touching the .pch is safe: clang validates the
// mtimes of the headers a PCH depends on, not of the PCH file itself.
//
// The preamble must be touched alongside it. A PCH records its preamble as an
// input, so pruning the preamble invalidates the PCH; refreshing only the .pch
// would let one in daily use survive while the preamble underneath it aged out.
void touch_pch_entry(const std::string& pch_path, const std::string& preamble_path)
{
  const auto now = std::filesystem::file_time_type::clock::now();
  for (const auto& path : {std::cref(pch_path), std::cref(preamble_path)})
  {
    std::error_code ec;
    const auto mtime = std::filesystem::last_write_time(path.get(), ec);
    if (!ec && now - mtime > std::chrono::hours(24))
    {
      std::filesystem::last_write_time(path.get(), now, ec);
    }
  }
}

// Best-effort exclusive guard around PCH generation, held across processes and
// threads alike.
//
// Generating a PCH costs seconds of CPU and writes tens of megabytes. Without a
// guard, every process that starts against a cold cache generates its own copy
// of the same file -- a test runner fanning out eight workers against a cold
// cache would otherwise do eight identical generations at once.
//
// Deliberately non-blocking: a caller that cannot take the lock builds without
// a PCH instead of waiting for the holder. Waiting would trade one slow build
// for a slow build *plus* a stall, and would need timeout handling for a holder
// that died. Taking the slow path costs exactly what the build would have cost
// with PCH disabled.
//
// A directory is the lock: create_directory is atomic on POSIX and Windows and
// reports whether it did the creating, which is precisely test-and-set.
class PCHGenerationLock
{
public:
  explicit PCHGenerationLock(std::filesystem::path lock_path)
      : path_(std::move(lock_path))
  {
    std::error_code ec;
    held_ = std::filesystem::create_directory(path_, ec) && !ec;
    if (held_)
    {
      return;
    }

    // A lock left behind by a crashed process would otherwise disable PCH
    // generation forever, so treat a sufficiently old one as abandoned. The
    // threshold only has to exceed a legitimate generation (seconds).
    const auto mtime = std::filesystem::last_write_time(path_, ec);
    if (ec)
    {
      return;
    }
    if (std::filesystem::file_time_type::clock::now() - mtime > std::chrono::minutes(10))
    {
      std::error_code remove_ec;
      std::filesystem::remove(path_, remove_ec);
      std::error_code retry_ec;
      held_ = std::filesystem::create_directory(path_, retry_ec) && !retry_ec;
    }
  }

  ~PCHGenerationLock()
  {
    if (held_)
    {
      std::error_code ec;
      std::filesystem::remove(path_, ec);
    }
  }

  PCHGenerationLock(const PCHGenerationLock&)            = delete;
  PCHGenerationLock& operator=(const PCHGenerationLock&) = delete;

  bool held() const
  {
    return held_;
  }

private:
  std::filesystem::path path_;
  bool held_ = false;
};

bool create_pch_if_needed(
  hostjit::CompilerConfig config,
  libnvccPCHKind kind,
  const std::string& kind_name,
  std::string& diagnostics,
  std::string& pch_path,
  bool& generated)
{
  pch_path.clear();

  const auto& cache_dir = get_pch_cache_dir();
  if (cache_dir.empty())
  {
    diagnostics += "PCH cache directory unavailable; building without PCH\n";
    return false;
  }

  // Generate against the same options a later compile will use, minus the PCH
  // settings themselves -- those are what we are producing.
  config.enable_pch = false;
  config.device_pch_path.clear();
  config.host_pch_path.clear();

  // Strip everything that varies per build. None of it can affect a PCH: the
  // preamble is headers only, the user's operator is linked as bitcode after
  // the frontend runs, and the entry point name only drives post-compile
  // passes. Leaving them in would put `--device-bitcode=<temp path>` and
  // `--entry-point=<algo>` into the cache key, so every build would generate
  // its own PCH -- tens of them per test run, none ever reused.
  config.device_bitcode_files.clear();
  config.device_ltoir_files.clear();
  config.entry_point_name.clear();

  std::vector<std::string> options;
  config.appendCommandLineArguments(options);

  // The arch appears in the filename for readability only -- the hash already
  // covers it, because libnvcc takes one option list for both PCH kinds and
  // that list carries the target. (Clang's *host* frontend arguments contain no
  // arch, so host PCHs could in principle be shared across GPUs; that is not
  // exploitable here without assuming what libnvcc does with the option
  // internally, and guessing wrong would mean handing a compile the wrong PCH.)
  const std::string label = kind_name + "_sm" + std::to_string(config.sm_version);
  const std::string key   = hash_pch_options(options, kind_name);
  pch_path                = get_pch_path(cache_dir, label, key);
  const auto source_path  = get_pch_source_path(cache_dir, label, key);

  auto present = [&] {
    std::error_code ec;
    return std::filesystem::exists(pch_path, ec) && !ec;
  };

  if (present())
  {
    touch_pch_entry(pch_path, source_path);
    return true;
  }

  // Only one generator at a time; losers build without a PCH rather than
  // waiting. See PCHGenerationLock.
  PCHGenerationLock lock(cache_dir / (label + "_" + key + ".lock"));
  if (!lock.held())
  {
    // The holder may have finished between the check above and here.
    if (present())
    {
      touch_pch_entry(pch_path, source_path);
      return true;
    }
    diagnostics += kind_name + " PCH generation already in progress elsewhere; building without PCH\n";
    pch_path.clear();
    return false;
  }

  // Re-check under the lock: another generator may have landed the file while
  // we were acquiring it.
  if (present())
  {
    touch_pch_entry(pch_path, source_path);
    return true;
  }

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

  auto pch_result = libnvccCreatePCH(
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

  generated = true;
  return true;
}

// Discard a PCH the compiler rejected, so the next build regenerates it instead
// of tripping over the same entry forever. Best-effort by design: another
// process may be reading or replacing it concurrently.
void discard_pch(const std::string& pch_path)
{
  if (pch_path.empty())
  {
    return;
  }
  std::error_code ec;
  std::filesystem::remove(pch_path, ec);
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

  bool generated = false;

  std::string device_pch_path;
  if (create_pch_if_needed(prepared, LIBNVCC_PCH_DEVICE, "device", diagnostics, device_pch_path, generated))
  {
    prepared.device_pch_path = std::move(device_pch_path);
  }

  std::string host_pch_path;
  if (create_pch_if_needed(prepared, LIBNVCC_PCH_HOST, "host", diagnostics, host_pch_path, generated))
  {
    prepared.host_pch_path = std::move(host_pch_path);
  }

  // Evict only when this build actually added to the cache, and only once both
  // entries are known, so neither can be evicted out from under the compile
  // that is about to use them.
  if (generated)
  {
    const auto& cache_dir = get_pch_cache_dir();
    if (!cache_dir.empty())
    {
      evict_pch_cache(cache_dir, {prepared.device_pch_path, prepared.host_pch_path});
    }
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
std::string pchCacheDir()
{
  return get_pch_cache_dir().string();
}

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

  // A PCH the compiler rejects -- stale after an in-place header upgrade, or
  // written by a build whose options happened to hash the same -- must not be
  // able to fail the build. Drop it and compile the honest way. Clang validates
  // a PCH against both the command line and the size/mtime of every header it
  // covers, so a rejected PCH is a routine possibility rather than a defect.
  const bool used_pch = !libnvcc_config.device_pch_path.empty() || !libnvcc_config.host_pch_path.empty();
  if (compile_result != LIBNVCC_SUCCESS && used_pch)
  {
    discard_pch(libnvcc_config.device_pch_path);
    discard_pch(libnvcc_config.host_pch_path);
    if (config_.verbose)
    {
      std::cout << "PCH rejected by the compiler; retrying without it\n";
    }

    libnvcc_config.enable_pch = false;
    libnvcc_config.device_pch_path.clear();
    libnvcc_config.host_pch_path.clear();

    options.clear();
    libnvcc_config.appendCommandLineArguments(options);
    option_ptrs = hostjit::detail::make_libnvcc_option_ptrs(options);

    // A fresh program: the failed attempt's log would otherwise be reported
    // alongside the retry's, so a genuine compile error would appear twice.
    hostjit::detail::LibnvccProgramGuard retry_program;
    auto retry_create = libnvccCreateProgram(&retry_program.program, source_code.c_str(), "input.cu");
    if (retry_create != LIBNVCC_SUCCESS)
    {
      last_error_ = std::string("Failed to create libnvcc program: ") + libnvccGetErrorString(retry_create);
      removeTempDirectory();
      return false;
    }

    compile_result = libnvccCompileProgramToObject(
      retry_program.program,
      obj_path.c_str(),
      cubin_path.c_str(),
      static_cast<int>(option_ptrs.size()),
      option_ptrs.empty() ? nullptr : option_ptrs.data());
    compile_log = hostjit::detail::get_libnvcc_program_log(retry_program.program);

    // The retry owns the rest of this build: linking below reuses `program`'s
    // handle for its log, so hand ownership over.
    std::swap(program.program, retry_program.program);
  }

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
