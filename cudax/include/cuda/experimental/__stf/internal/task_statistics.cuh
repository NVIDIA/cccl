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
 * @brief Implements the tracking and recording of task statistics. Each task is uniquely
 * identified by the user set task symbol and the size of its dependencies. Can be enabled
 * by setting the CUDASTF_CALIBRATION_FILE environment variable to point to the file which
 * will store the results.
 *
 *
 * CUDASTF_CALIBRATION_FILE
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

#include <cuda/experimental/__stf/utility/hash.cuh>
#include <cuda/experimental/__stf/utility/traits.cuh>
#include <cuda/experimental/__stf/utility/unittest.cuh>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

namespace cuda::experimental::stf::reserved
{
/**
 * @brief This class stores statistics about task execution time
 */
class task_statistics : public reserved::meyers_singleton<task_statistics>
{
protected:
  task_statistics()
  {
    const char* filename = ::std::getenv("CUDASTF_CALIBRATION_FILE");

    if (!filename)
    {
      calibrating = false;
      return;
    }

    calibrating      = true;
    calibration_file = ::std::string(filename);
  }

  ~task_statistics()
  {
    if (!calibration_file.empty())
    {
      write_stats();
    }
  }

public:
  bool is_calibrating() const
  {
    return calibrating;
  }

  bool is_calibrating_to_file() const
  {
    return !calibration_file.empty();
  }

  /**
   * @brief Enable online calibration
   */
  void enable_calibration()
  {
    calibrating = true;
  }

  template <typename task_type>
  void log_task_time(const task_type& t, double time)
  {
    auto key = ::std::pair{t.get_symbol(), get_data_footprint(t)};

    auto it = statistics.find(key);
    if (it == statistics.end())
    {
      statistics.emplace(key, statistic(time));
    }
    else
    {
      it->second.update(time);
    }
  }

  class statistic
  {
  public:
    statistic() = delete;
    explicit statistic(double initial_time)
        : num_calls(1)
        , mean(initial_time)
        , squares(0.0)
    {}

    // Constructor when reading from file
    statistic(int num_calls, double mean, double stddev)
        : num_calls(num_calls)
        , mean(mean)
        , stddev(stddev)
    {}

    void update(double new_time)
    {
      // Use Welford's method to compute mean https://stackoverflow.com/a/15638726
      num_calls++;

      double delta = new_time - mean;

      mean += delta / num_calls;
      squares += delta * (new_time - mean);

      double variance = (num_calls == 1 ? 0 : squares / (num_calls - 1));
      stddev          = ::std::sqrt(variance);
    }

    int get_num_calls() const
    {
      return num_calls;
    }
    double get_mean() const
    {
      return mean;
    }
    double get_stddev() const
    {
      return stddev;
    }

  private:
    int num_calls  = 0;
    double mean    = 0.0;
    double squares = 0.0;

    // Only used when reading file
    double stddev = 0.0;
  };

  using statistics_map_key_t = ::std::pair<::std::string, size_t>;
  using statistics_map_t =
    ::std::unordered_map<statistics_map_key_t, statistic, cuda::experimental::stf::hash<statistics_map_key_t>>;

  void read_statistics_file(const char* filename)
  {
    _CCCL_ASSERT(!is_calibrating(), "Cannot calibrate if we read task statistics from a file.");

    ::std::ifstream file(filename);
    EXPECT(file, "Failed to read statistics file '", filename, "'.");

    int current_line = 0;
    for (::std::string line; ::std::getline(file, line); ++current_line)
    {
      if (current_line == 0) // This is the csv header
      {
        continue;
      }

      ::std::stringstream ss(line);

      ::std::string task_name;
      size_t size   = 0;
      double time   = -1.0;
      int num_calls = -1;
      double stddev = -1.0;

      int column = 0;
      for (::std::string cell; ::std::getline(ss, cell, ','); ++column)
      {
        EXPECT(column < 5, "Invalid number of columns in statistics file ", column + 1, " (expected 5).");

        if (column == 0)
        {
          task_name = cell;
        }
        else if (column == 1)
        {
          size = ::std::stoul(cell);
        }
        else if (column == 2)
        {
          num_calls = ::std::stoi(cell);
        }
        else if (column == 3)
        {
          time = ::std::stod(cell);
        }
        else if (column == 4)
        {
          stddev = ::std::stod(cell);
        }
      }

      EXPECT(time >= 0, "Invalid time value '", time, "' provided on line '", current_line, "'.");
      EXPECT(num_calls >= 0, "Invalid num_calls value '", num_calls, "' provided on line '", current_line, "'.");
      EXPECT(stddev >= 0, "Invalid stddev value '", stddev, "' provided on line '", current_line, "'.");

      ::std::pair<::std::string, size_t> key(task_name, size);
      statistics.emplace(key, statistic(num_calls, time, stddev));
    }
  }

  /**
   * @brief Get statistics associated with a specific task
   *
   * @tparam Type of task
   * @param The specified task
   * @return A pair of the task time and the number of calls so far
   */
  template <typename task_type>
  ::std::pair<double, int> get_task_stats(const task_type& t)
  {
    const ::std::string& task_name = t.get_symbol();
    size_t data_footprint          = get_data_footprint(t);

    ::std::pair<::std::string, size_t> key(task_name, data_footprint);
    auto it = statistics.find(key);
    if (it != statistics.end())
    {
      const statistic& s = it->second;
      return {s.get_mean(), s.get_num_calls()};
    }

    // If we do not have the task in the map, this means we have to be calibrating online.
    // A missing task implies an incomplete stats file was provided
    EXPECT(is_calibrating(), "Task '", task_name, "' with size ", data_footprint, " not provided in stats file.");

    return {0.0, 0};
  }

private:
  ::std::string calibration_file;
  bool calibrating = false;

  template <typename task_type>
  size_t get_data_footprint(const task_type& t) const
  {
    size_t data_footprint = 0;
    const auto deps       = t.get_task_deps();

    for (auto it = deps.begin(); it < deps.end(); it++)
    {
      size_t new_size = it->get_data_footprint();
      data_footprint += new_size;
      assert(data_footprint >= new_size); // Check for overflow
    }

    return data_footprint;
  }

  void write_stats() const
  {
    ::std::ofstream file(calibration_file);
    if (!file)
    {
      ::std::cerr << "Failed to write to calibration file '" << calibration_file << "'.\n";
      return;
    }

    file << "task,size,num_calls,mean,stddev\n";
    for (const auto& [key, value] : statistics)
    {
      double stddev = value.get_stddev();
      file << key.first << "," << key.second << "," << value.get_num_calls() << "," << value.get_mean() << "," << stddev
           << '\n';
    }

    file.close();
    if (file.rdstate() & ::std::ifstream::failbit)
    {
      ::std::cerr << "ERROR: Closing calibration file failed.\n";
    }
  }

  statistics_map_t statistics;
};
} // namespace cuda::experimental::stf::reserved
