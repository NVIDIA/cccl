#include <cstdio>
#include <filesystem>
#include <fstream>

#include <hostjit/codegen/bitcode.hpp>
#include <hostjit/compiler.hpp>

namespace hostjit::codegen
{
namespace
{
bool write_file(const char* data, size_t size, const std::string& path)
{
  std::ofstream f(path, std::ios::binary);
  if (!f)
  {
    return false;
  }
  f.write(data, static_cast<std::streamsize>(size));
  return f.good();
}

std::string make_temp_path(const std::string& prefix, uintptr_t id, const std::string& ext)
{
  return (std::filesystem::temp_directory_path() / (prefix + std::to_string(id) + ext)).string();
}
} // anonymous namespace

BitcodeCollector::BitcodeCollector(CompilerConfig& config, uintptr_t unique_id)
    : config_(config)
    , unique_id_(unique_id)
{}

bool BitcodeCollector::is_bitcode_op(cccl_op_t op)
{
  return (op.code_type == CCCL_OP_LLVM_IR || op.code_type == CCCL_OP_LTOIR) && op.code != nullptr && op.code_size > 0;
}

void BitcodeCollector::add_raw_bitcode(const char* data, size_t size, const std::string& name)
{
  if (data && size > 0)
  {
    auto path = make_temp_path("cccl_" + name + "_", unique_id_, ".bc");
    if (write_file(data, size, path))
    {
      config_.device_bitcode_files.push_back(path);
      temp_paths_.push_back(path);
    }
  }
}

bool BitcodeCollector::compile_and_add(const char* source, size_t source_size, const std::string& name)
{
  hostjit::CUDACompiler compiler;
  std::string src(source, source_size);
  auto result = compiler.compileToDeviceBitcode(src, config_);
  if (!result.success)
  {
    fprintf(stderr, "\nERROR compiling %s to bitcode: %s\n", name.c_str(), result.diagnostics.c_str());
    return false;
  }
  auto path = make_temp_path("cccl_" + name + "_", unique_id_, ".bc");
  if (write_file(result.bitcode.data(), result.bitcode.size(), path))
  {
    config_.device_bitcode_files.push_back(path);
    temp_paths_.push_back(path);
    return true;
  }
  return false;
}

void BitcodeCollector::add_op_code(cccl_op_t& op, const std::string& name)
{
  if (!op.code || op.code_size == 0)
  {
    return;
  }

  // Deduplicate: if two iterators share the same symbol (e.g. two CountingIterators
  // of the same type), only compile/link the bitcode once.
  if (op.name && op.name[0])
  {
    if (!added_symbols_.insert(std::string(op.name)).second)
    {
      return; // already added
    }
  }

  if (op.code_type == CCCL_OP_CPP_SOURCE)
  {
    compile_and_add(op.code, op.code_size, name);
  }
  else
  {
    add_raw_bitcode(op.code, op.code_size, name);
  }

  // Also link any extra modules (child iterator ops, numba-compiled ops).
  int extra_counter = 0;
  for (size_t i = 0; i < op.num_extra_ltoirs; ++i)
  {
    if (op.extra_ltoirs[i] && op.extra_ltoir_sizes[i] > 0)
    {
      auto extra_name    = name + "_extra" + std::to_string(extra_counter++);
      const auto* data   = op.extra_ltoirs[i];
      const auto data_sz = op.extra_ltoir_sizes[i];
      // Detect LLVM bitcode magic bytes (0x42 0x43 = "BC")
      const bool is_bitcode_data =
        data_sz >= 2 && static_cast<unsigned char>(data[0]) == 0x42 && static_cast<unsigned char>(data[1]) == 0x43;
      if (!is_bitcode_data)
      {
        // Treat as C++ source
        compile_and_add(data, data_sz, extra_name);
      }
      else
      {
        add_raw_bitcode(data, data_sz, extra_name);
      }
    }
  }
}

void BitcodeCollector::add_op(cccl_op_t op, const std::string& label)
{
  // Only add bitcode for LTOIR/LLVM_IR ops (CPP_SOURCE is embedded inline in the generated source)
  if (is_bitcode_op(op))
  {
    add_raw_bitcode(op.code, op.code_size, label);
  }

  // Always process extra ltoirs
  int extra_counter = 0;
  for (size_t i = 0; i < op.num_extra_ltoirs; ++i)
  {
    if (op.extra_ltoirs[i] && op.extra_ltoir_sizes[i] > 0)
    {
      auto extra_name    = label + "_extra" + std::to_string(extra_counter++);
      const auto* data   = op.extra_ltoirs[i];
      const auto data_sz = op.extra_ltoir_sizes[i];
      const bool is_bitcode_data =
        data_sz >= 2 && static_cast<unsigned char>(data[0]) == 0x42 && static_cast<unsigned char>(data[1]) == 0x43;
      if (!is_bitcode_data)
      {
        compile_and_add(data, data_sz, extra_name);
      }
      else
      {
        add_raw_bitcode(data, data_sz, extra_name);
      }
    }
  }
}

void BitcodeCollector::add_iterator(cccl_iterator_t it, const std::string& label_prefix)
{
  if (it.type != CCCL_ITERATOR)
  {
    return;
  }
  add_op_code(it.advance, label_prefix + "_adv");
  add_op_code(it.dereference, label_prefix + "_deref");
}

void BitcodeCollector::cleanup()
{
  for (const auto& p : temp_paths_)
  {
    std::filesystem::remove(p);
  }
  temp_paths_.clear();
}
} // namespace hostjit::codegen
