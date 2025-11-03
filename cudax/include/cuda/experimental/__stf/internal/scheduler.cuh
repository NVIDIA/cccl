//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Implements automatic task scheduling
 *
 * CUDASTF_SCHEDULE
 * CUDASTF_SCHEDULE_FILE
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/internal/task.cuh> // scheduler uses task
#include <cuda/experimental/__stf/internal/task_statistics.cuh> // heft_scheduler uses statistics_t

#include <cstdlib> // rand()
#include <filesystem> // ::std::filesystem::exists
#include <fstream> // ::std::ifstream
#include <iostream>
#include <limits> // ::std::numeric_limits<double>::max()
#include <random> // random_scheduler uses rng
#include <sstream> // ::std::stringstream
#include <string>
#include <unordered_map>
#include <vector>

namespace cuda::experimental::stf::reserved
{
/**
 * @brief The scheduler class defines the interface that all schedulers must follow to assign tasks to devices
 */
class scheduler
{
public:
  scheduler()
  {
    cuda_safe_call(cudaGetDeviceCount(&num_devices));
    assert(num_devices > 0);
  }

  /**
   * @brief Assign a task to a device
   *
   * @param mapping_id The mapping ID of the task
   * @return The device ID of the assigned device and a boolean whether this task still needs calibration
   */
  virtual ::std::pair<exec_place, bool> schedule_task(const task& t) = 0;

  /// @brief Destructor for the scheduler
  virtual ~scheduler() = default;

  static ::std::unique_ptr<scheduler> make(const char* schedule_type);

protected:
  int num_devices = 0;

  // Map from task id to device
  using schedule_t = ::std::unordered_map<mapping_id_t, int, hash<mapping_id_t>>;
};

class random_scheduler : public scheduler
{
public:
  ::std::pair<exec_place, bool> schedule_task(const task&) override
  {
    return {exec_place::device(dist(gen)), false};
  }

private:
  ::std::mt19937 gen                     = ::std::mt19937(::std::random_device()());
  ::std::uniform_int_distribution<> dist = ::std::uniform_int_distribution<>(0, num_devices - 1);
};

class round_robin_scheduler : public scheduler
{
public:
  round_robin_scheduler() = default;

  ::std::pair<exec_place, bool> schedule_task(const task&) override
  {
    return {exec_place::device(current_device++ % num_devices), false};
  }

private:
  int current_device = 0;
};

class post_mortem_scheduler : public scheduler
{
public:
  post_mortem_scheduler(const char* schedule_file)
  {
    read_schedule_file(schedule_file);
  }

  ::std::pair<exec_place, bool> schedule_task(const task& t) override
  {
    return {exec_place::device(schedule[t.get_mapping_id()]), false};
  }

private:
  schedule_t schedule;

  /* Read the csv schedule file mapping tasks to devices */
  void read_schedule_file(const char* filename)
  {
    ::std::ifstream file(filename);
    EXPECT(file, "Failed to open schedule file: '", filename, "'.");

    int current_line = 0;
    for (::std::string line; ::std::getline(file, line); ++current_line)
    {
      ::std::stringstream ss(line);

      int mapping_id = -1;
      int dev_id     = -1;

      int column = 0;
      for (::std::string cell; ::std::getline(ss, cell, ','); ++column)
      {
        if (column == 0)
        {
          dev_id = ::std::stoi(cell);
        }
        else if (column == 1)
        {
          mapping_id = ::std::stoi(cell);
        }
      }

      EXPECT(dev_id >= 0, "Invalid device id value '", dev_id, "' provided on line '", current_line, "'.");
      EXPECT(mapping_id >= 0, "Invalid mapping id value '", mapping_id, "' provided on line '", current_line, "'.");

      schedule[mapping_id] = dev_id;
    }
  }
};

class heft_scheduler : public scheduler
{
public:
  heft_scheduler()
      : gpu_loads(num_devices, 0.0)
      , msi(num_devices)
  {
    const char* filename = getenv("CUDASTF_TASK_STATISTICS");

    if (filename)
    {
      statistics.read_statistics_file(filename);
    }
    else
    {
      statistics.enable_calibration();
    }
  }

  ::std::pair<exec_place, bool> schedule_task(const task& t) override
  {
    auto [task_cost, num_calls] = statistics.get_task_stats(t);

    if (num_calls == 0)
    {
      task_cost = default_cost;
    }

    double best_end = ::std::numeric_limits<double>::max();
    int best_device = -1;

    for (int i = 0; i < num_devices; i++)
    {
      int current_device = i;

      double total_cost = cost_on_device<task>(t, current_device, task_cost, gpu_loads[current_device]);
      if (total_cost < best_end)
      {
        best_device = current_device;
        best_end    = total_cost;
      }
    }

    gpu_loads[best_device]       = best_end;
    schedule[t.get_mapping_id()] = best_device;

    auto& deps = t.get_task_deps();
    for (const auto& dep : deps)
    {
      msi.update_msi_for_dep(best_device, dep, best_end);
    }

    bool needs_calibration = num_calls < num_samples;

    return {exec_place::device(best_device), needs_calibration};
  }

  ~heft_scheduler()
  {
    const char* schedule_file = getenv("CUDASTF_HEFT_SCHEDULE");
    if (schedule_file)
    {
      write_schedule_file(schedule_file);
    }
  }

private:
  class msi_protocol
  {
  public:
    msi_protocol(int num_devices)
        : num_devices(num_devices)
    {}

    double when_available(int device_id, const task_dep_untyped& dep)
    {
      auto& info                                        = get_symbol_info(dep.get_symbol());
      const ::std::pair<msi_state, double>& device_info = info[device_id];

      msi_state device_state = device_info.first;
      switch (device_state)
      {
        case msi_state::modified:
        case msi_state::shared:
          return device_info.second;
        default:
          break;
      }

      double earliest = get_earliest(dep);
      return earliest;
    }

    void update_msi_for_dep(int device_id, const task_dep_untyped& dep, double task_end)
    {
      const ::std::string& symbol = dep.get_symbol();
      const access_mode mode      = dep.get_access_mode();

      auto& info                                  = get_symbol_info(symbol);
      ::std::pair<msi_state, double>& device_info = info[device_id];

      // Update local state first
      switch (device_info.first)
      {
        case msi_state::modified:
          device_info.second = task_end;
          break;
        case msi_state::shared:
          if (mode != access_mode::read)
          {
            device_info.first  = msi_state::modified;
            device_info.second = task_end;
          }
          break;
        case msi_state::invalid:
          if (mode == access_mode::read)
          {
            device_info.first  = msi_state::shared;
            device_info.second = get_earliest(dep);
          }
          else
          {
            device_info.first  = msi_state::modified;
            device_info.second = task_end;
          }
          break;
      }

      for (int i = 0; i < num_devices; i++)
      {
        if (i == device_id) // already updated
        {
          continue;
        }

        ::std::pair<msi_state, double>& i_info = info[i];

        msi_state state = i_info.first;
        switch (state)
        {
          case msi_state::modified:
            if (mode == access_mode::read)
            {
              i_info.first = msi_state::shared;
            }
            else
            {
              i_info.first  = msi_state::invalid;
              i_info.second = -1.0;
            }
            break;
          case msi_state::shared:
            if (mode != access_mode::read)
            {
              i_info.first  = msi_state::invalid;
              i_info.second = -1.0;
            }
            break;
          default:
            assert(state == msi_state::invalid);
            break;
        }
      }
    }

  private:
    enum class msi_state : unsigned int
    {
      modified = 0,
      shared   = 1,
      invalid  = 2
    };

    int num_devices;
    const double bandwidth = 250 * 1e5; // Bytes/ms, Obtained by running
                                        // cuda-samples/Samples/5_Domain_Specific/p2pBandwidthLatencyTest
    using cache_state = ::std::vector<::std::pair<msi_state, double>>;
    ::std::unordered_map<::std::string, cache_state> cache;

    cache_state& get_symbol_info(const ::std::string& symbol)
    {
      auto it = cache.find(symbol);
      if (it == cache.end())
      {
        cache_state state(num_devices, ::std::make_pair(msi_state::invalid, -1.0));
        cache.emplace(symbol, mv(state));
        it = cache.find(symbol);
      }

      return it->second;
    }

    double get_earliest(const task_dep_untyped& dep) const
    {
      const ::std::string& symbol = dep.get_symbol();
      const auto& info            = cache.at(symbol); // need to use at() to keep method const
      double earliest             = ::std::numeric_limits<double>::max();

      bool found_one = false;

      double comm_cost = dep.get_data_footprint() / bandwidth;

      for (int i = 0; i < num_devices; i++)
      {
        const ::std::pair<msi_state, double>& device_info = info[i];
        if (device_info.first != msi_state::invalid)
        {
          found_one = true;
          earliest  = ::std::min(earliest, device_info.second + comm_cost);
        }
      }

      if (!found_one) // If the data is not on any device
      {
        earliest = comm_cost;
      }

      return earliest;
    }
  };

  template <typename task_type>
  double cost_on_device(const task_type& t, int device_id, double task_cost, double when_can_start)
  {
    double data_available = 0.0;

    const auto& deps = t.get_task_deps();
    for (const auto& dep : deps)
    {
      access_mode mode = dep.get_access_mode();
      switch (mode)
      {
        case access_mode::read:
        case access_mode::rw:
          data_available = ::std::max(data_available, msi.when_available(device_id, dep));
          break;
        default:
          break;
      }
    }

    double possible_start = ::std::max(when_can_start, data_available);
    double end            = possible_start + task_cost;

    return end;
  }

  void write_schedule_file(const char* schedule_file) const
  {
    ::std::ofstream file(schedule_file);
    if (!file)
    {
      ::std::cerr << "Failed to write to heft schedule file '" << schedule_file << "'.\n";
      return;
    }

    for (const auto& [device_id, mapping_id] : schedule)
    {
      file << mapping_id << "," << device_id << '\n';
    }
  }

  ::std::vector<double> gpu_loads; // TODO: is it better to use a ::std::array<double, 8>?
  task_statistics& statistics = task_statistics::instance();
  msi_protocol msi;
  schedule_t schedule;
  const int num_samples     = 5;
  const double default_cost = 0.5;
};

inline ::std::unique_ptr<scheduler> scheduler::make(const char* schedule_type)
{
  if (!schedule_type)
  {
    return nullptr;
  }

  const auto schedule_type_s = ::std::string(schedule_type);

  if (schedule_type_s == "post_mortem")
  {
    const char* schedule_file = getenv("CUDASTF_SCHEDULE_FILE");

    EXPECT(schedule_file, "CUDASTF_SCHEDULE set to 'post_mortem' but CUDASTF_SCHEDULE_FILE is unset.");
    EXPECT(::std::filesystem::exists(schedule_file), "CUDASTF_SCHEDULE_FILE '", schedule_file, "' does not exist");

    return ::std::make_unique<post_mortem_scheduler>(schedule_file);
  }

  if (schedule_type_s == "random")
  {
    return ::std::make_unique<random_scheduler>();
  }

  if (schedule_type_s == "round_robin")
  {
    return ::std::make_unique<round_robin_scheduler>();
  }

  if (schedule_type_s == "heft")
  {
    return ::std::make_unique<heft_scheduler>();
  }

  ::std::cerr << "Invalid CUDASTF_SCHEDULE value '" << schedule_type << "'\n";
  exit(EXIT_FAILURE);
}
} // namespace cuda::experimental::stf::reserved
