#pragma once

#include <cstdint>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include <cccl/c/types.h>
#include <hostjit/config.hpp>

namespace hostjit::codegen
{
// Manages bitcode files needed for linking. Collects LTOIR, LLVM IR,
// and C++ source (compiling the latter to bitcode on the fly).
// Tracks temp file paths for cleanup.
class BitcodeCollector
{
public:
  explicit BitcodeCollector(CompilerConfig& config, uintptr_t unique_id);

  // Add bitcode from an operator (handles LTOIR, LLVM_IR, CPP_SOURCE,
  // and extra modules).
  void add_op(cccl_op_t op, const std::string& label);

  // Add bitcode from a custom iterator's advance/dereference ops.
  void add_iterator(cccl_iterator_t it, const std::string& label_prefix);

  // Returns true if the op has linked bitcode (LTOIR or LLVM_IR).
  static bool is_bitcode_op(cccl_op_t op);

  // Clean up all temporary files.
  void cleanup();

private:
  void add_raw_bitcode(const char* data, size_t size, const std::string& name);
  bool compile_and_add(const char* source, size_t source_size, const std::string& name);
  void add_op_code(cccl_op_t& op, const std::string& name);

  CompilerConfig& config_;
  uintptr_t unique_id_;
  std::vector<std::string> temp_paths_;
  std::set<std::string> added_symbols_; // dedup by op.name (when present)
  std::unordered_set<std::uint64_t> added_content_hashes_; // dedup by content hash for unnamed extras
};
} // namespace hostjit::codegen
