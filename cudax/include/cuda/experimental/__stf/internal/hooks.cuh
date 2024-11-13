//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/internal/task_dep.cuh>

#include <vector>

namespace cuda::experimental::stf
{

namespace reserved
{

/*
 * When we dump the content of the logical data, we put them in files which are
 * automatically named according to this counter. By having such a determinitic
 * counter shared by all contexts, we can compare the content of files with the
 * same index during different executions.
 */
class dump_hook_cnt : public reserved::meyers_singleton<dump_hook_cnt>
{
protected:
  dump_hook_cnt()
  {
    cnt = 0;
  }

  ~dump_hook_cnt() = default;

public:
  static int get()
  {
    return instance().cnt++;
  }

private:
  int cnt;
};
} // namespace reserved

template <typename Unknown, size_t... i>
void data_dump(Unknown, ::std::ostream& file = ::std::cerr)
{
  file << "Dunno how to dump object of type " << type_name<Unknown> << ".\n";
}

template <typename Unknown, size_t... i>
size_t data_hash(Unknown)
{
  return 0;
}

namespace reserved
{
inline void create_dump_dir(const ::std::string& dump_dir)
{
  // Create the directory (no-op if it already exists)
  if (::std::filesystem::create_directories(dump_dir))
  {
    //::std::cout << "Directory \"" << dump_dir << "\" was created successfully." << ::std::endl;
  }
  else
  {
    if (!::std::filesystem::exists(dump_dir))
    {
      ::std::cerr << "An error occurred while trying to create the dump_dir \"" << dump_dir << "\"." << ::std::endl;
      abort();
    }
  }
}

inline void ensure_directory_exists(const ::std::string& dir_path)
{
  // Check if the directory exists
  if (!::std::filesystem::exists(dir_path))
  {
    ::std::cerr << "Directory \"" << dir_path << "\" does not exist." << ::std::endl;
    abort();
  }
}

/* Compute a vector of hooks to dump modified logical data (using
 * typed-erased hooks). This will generate one host_launch task for each
 * modified logical data after task submission. */
template <typename ctxt_t, typename... Deps>
static ::std::vector<::std::function<void()>> get_dump_hooks(ctxt_t* ctx, task_dep<Deps>... deps)
{
  ::std::vector<::std::function<void()>> hooks;

  // If the CUDASTF_AUTO_DUMP is not set, or set to 0, we don't save the content
  const char* dump_str = ::std::getenv("CUDASTF_AUTO_DUMP");
  bool dump            = dump_str && atoi(dump_str) != 0;

  const char* compare_str = ::std::getenv("CUDASTF_AUTO_COMPARE");
  bool compare            = compare_str && atoi(compare_str) != 0;

  if (!dump && !compare)
  {
    return hooks;
  }

  bool hash_only = ::std::getenv("CUDASTF_AUTO_DUMP_ONLY_HASH");

  // Where do we write dumped content ? We postpone the creation of this
  // directory to the first time we need to create a directory to avoid
  // creating an empty dir if no data was dumped
  const char* dump_dir_env = ::std::getenv("CUDASTF_AUTO_DUMP_DIR");
  ::std::string dump_dir   = (dump_dir_env != nullptr) ? dump_dir_env : "dump/";

  // For every dependency, we create a hook to dump the content of the
  // logical data if it was modified.
  auto dump_dep = [&, dump_dir](auto dep) {
    auto dep_ld = dep.get_data();
    if (dep.get_access_mode() != access_mode::read && dep_ld.get_auto_dump())
    {
      auto ro_dep = dep.as_read_mode();

      /* We either make sure the directory exists or lazily create it if
       * we need to add content when dumping data */
      if (compare)
      {
        ensure_directory_exists(dump_dir);
      }
      else
      {
        create_dump_dir(dump_dir);
      }

      // Create a hook that will be executed after the submission of the
      // tasks: this will submit a host callback to write the content in
      // a file
      auto h = [ctx, ro_dep, dump_dir, hash_only, compare]() {
        // Get the next counter (to have a repeatable order)
        int cnt                = reserved::dump_hook_cnt::get();
        ::std::string filePath = dump_dir + "/" + ::std::to_string(cnt);

        if (compare)
        {
          // Instead of using a host callback which might have had
          // better performance, we use a task and a synchronization
          // because it is easier to break on errors with a debugger
          // when a mismatch is found.
          ctx->task(exec_place::host, ro_dep).set_symbol("compare " + ::std::to_string(cnt))
              ->*[filePath](cudaStream_t stream, auto s) {
                    cuda_safe_call(cudaStreamSynchronize(stream));
                    ::std::ifstream f(filePath);
                    if (!f.is_open())
                    {
                      ::std::cerr << "Failed to open " << filePath << ::std::endl;
                      abort();
                    }

                    size_t saved_hash;
                    f >> saved_hash;
                    f.close();

                    size_t computed_hash = data_hash(s);
                    if (computed_hash != saved_hash)
                    {
                      ::std::cerr << "Hash mismatch : computed = " << computed_hash << ", saved = " << saved_hash
                                  << " in " << filePath << ::std::endl;
                      if (getenv("CUDASTF_AUTO_COMPARE_ABORT_ON_ERRORS"))
                      {
                        abort();
                      }
                    }
                  };
        }
        else
        {
          ctx->host_launch(ro_dep).set_symbol("dump " + ::std::to_string(cnt))->*[filePath, hash_only](auto s) {
            ::std::ofstream f(filePath);
            if (!f.is_open())
            {
              ::std::cerr << "Failed to open " << filePath << ::std::endl;
              abort();
            }
            // Compute a hash of the content, to easily compare equality
            size_t hsh = data_hash(s);
            f << hsh << ::std::endl;

            if (!hash_only)
            {
              // Dump the actual data content (may be very large)
              data_dump(s, f);
            }
            f.close();
          };
        }
      };
      hooks.push_back(h);
    }
  };
  ::std::ignore = dump_dep;

  (dump_dep(deps), ...); // Call dump_dep on every dependency

  return hooks;
}

} // end namespace reserved

} // end namespace cuda::experimental::stf
