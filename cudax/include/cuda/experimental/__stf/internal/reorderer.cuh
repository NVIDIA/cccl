//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/** @file
 *
 * @brief Implements automatic task reordering
 *
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

#include <cuda/experimental/__stf/internal/task_dep.cuh> // reorderer_payload uses task_dep_vector_untyped
#include <cuda/experimental/__stf/internal/task_statistics.cuh> // heft_scheduler uses statistics_t

#include <algorithm> // ::std::shuffle
#include <functional> // ::std::function
#include <memory> // ::std::unique_ptr
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace cuda::experimental::stf::reserved
{
/**
 * @brief We cannot pass deferred_stream_task<> to the reorderer due to circular
 * dependencies, so we store all the necessary info in this struct instead.
 */
struct reorderer_payload
{
  reorderer_payload(
    ::std::string s, int id, ::std::unordered_set<int> succ, ::std::unordered_set<int> pred, task_dep_vector_untyped d)
      : symbol(mv(s))
      , mapping_id(id)
      , successors(mv(succ))
      , predecessors(mv(pred))
      , deps(mv(d))
  {}

  reorderer_payload() = delete;

  bool done           = false;
  bool done_execution = false;
  double upward_rank  = -1.0;
  int device          = -1;

  size_t num_successors() const
  {
    return successors.size();
  }
  size_t num_predecessors() const
  {
    return predecessors.size();
  }

  /// Needed for task_statistics
  const ::std::string& get_symbol() const
  {
    return symbol;
  }

  /// Needed for task_statistics
  const task_dep_vector_untyped& get_task_deps() const
  {
    return deps;
  }

  ::std::string symbol;
  int mapping_id;
  ::std::unordered_set<int> successors;
  ::std::unordered_set<int> predecessors;
  task_dep_vector_untyped deps;
};

/**
 * @brief The reorderer class defines the interface for all reorderers
 */
class reorderer
{
public:
  /**
   * @brief Reorder a vector of tasks
   *
   * @param tasks The vector of tasks to be reordered
   */
  virtual void reorder_tasks(::std::vector<int>& tasks, ::std::unordered_map<int, reorderer_payload>& task_map) = 0;

  /// @brief Destructor for the reorderer
  virtual ~reorderer() = default;

  static ::std::unique_ptr<reorderer> make(const char* reorderer_type);

protected:
  reorderer() = default;

  const int num_devices = cuda_try<cudaGetDeviceCount>();
};

class random_reorderer : public reorderer
{
public:
  random_reorderer() = default;

  void reorder_tasks(::std::vector<int>& tasks, ::std::unordered_map<int, reorderer_payload>&) override
  {
    ::std::shuffle(::std::begin(tasks), ::std::end(tasks), gen);
  }

private:
  ::std::mt19937 gen = ::std::mt19937(::std::random_device()());
};

class heft_reorderer : public reorderer
{
public:
  heft_reorderer()
      : reorderer()
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

  void reorder_tasks(::std::vector<int>& tasks, ::std::unordered_map<int, reorderer_payload>& task_map) override
  {
    calculate_upward_ranks(tasks, task_map);
    rearrange_tasks(tasks, task_map);
  }

private:
  void calculate_upward_ranks(const ::std::vector<int>& tasks,
                              ::std::unordered_map<int, reorderer_payload>& task_map) const
  {
    ::std::queue<int> work_list; // queue of mapping ids
    ::std::unordered_set<int> tasks_done;

    double comm_cost = 0.2;

    // Initialize the work_list with the leaf tasks
    for (int id : tasks)
    {
      auto& t = task_map.at(id);
      if (t.num_successors() == 0)
      {
        work_list.push(t.mapping_id);
      }
    }

    while (work_list.size() > 0)
    {
      auto& current_task = task_map.at(work_list.front());
      work_list.pop();

      tasks_done.insert(current_task.mapping_id);
      current_task.done = true;

      // The second term in the upward_rank equation that gets added to the task cost
      double second_term = 0.0;
      for (int s : current_task.successors)
      {
        const auto& succ = task_map.at(s);
        assert(succ.upward_rank != -1);
        second_term = ::std::max(second_term, comm_cost + succ.upward_rank);
      }

      ::std::pair<double, int> stats;
      double task_cost;
      if (current_task.get_symbol().rfind("task ", 0) == 0)
      {
        task_cost = 0;
      }
      else
      {
        stats     = statistics.get_task_stats(current_task);
        task_cost = ::std::get<0>(stats);
      }
      current_task.upward_rank = task_cost + second_term;

      for (int p : current_task.predecessors)
      {
        const auto& pred = task_map.at(p);
        if (tasks_done.count(p))
        {
          continue;
        }

        bool add_it = true;
        for (int s : pred.successors)
        {
          const auto& succ = task_map.at(s);
          if (succ.upward_rank == -1)
          {
            add_it = false;
            break;
          }
        }

        if (add_it)
        {
          work_list.push(pred.mapping_id);
        }
      }
    }
  }

  /**
   * @brief Now that we've calculated the upward ranks, we need to rearrange
   * the tasks. This isn't as simple as sorting the vector of tasks according
   * to the upward rank, as we need to take into account when tasks are ready.
   */
  void rearrange_tasks(::std::vector<int>& tasks, ::std::unordered_map<int, reorderer_payload>& task_map) const
  {
    using task_priority = ::std::pair<int,
                                      double>; // the double is the upward rank, needed for sorting. Should I just use
                                               // the reorderer_payload
    auto cmp = [](const task_priority& p1, const task_priority& p2) {
      return p1.second < p2.second;
    };
    ::std::priority_queue<task_priority, ::std::vector<task_priority>, decltype(cmp)> ready_tasks(cmp);

    for (int id : tasks)
    {
      const auto& t = task_map.at(id);
      if (t.num_predecessors() == 0)
      {
        ready_tasks.emplace(t.mapping_id, t.upward_rank);
      }
    }

    ::std::unordered_set<int> tasks_done; // shouldn't be necessary but we need it now
    ::std::vector<int> actual_order;

    while (ready_tasks.size() > 0)
    {
      auto [id, upward_rank] = ready_tasks.top();
      ready_tasks.pop();

      auto& current_task          = task_map.at(id);
      current_task.done_execution = true;
      tasks_done.insert(id);
      actual_order.push_back(id);

      for (int succ_id : current_task.successors)
      {
        if (tasks_done.count(succ_id))
        {
          continue;
        }

        const auto& succ  = task_map.at(succ_id);
        bool is_ready_now = true;

        for (int pred_id : succ.predecessors)
        {
          const auto& pred = task_map.at(pred_id);

          if (!pred.done_execution)
          {
            is_ready_now = false;
          }
        }

        if (is_ready_now)
        {
          ready_tasks.emplace(succ_id, succ.upward_rank);
        }
      }
    }

    tasks = mv(actual_order);
  }

  task_statistics& statistics = task_statistics::instance();
};

class post_mortem_reorderer : public reorderer
{
public:
  post_mortem_reorderer(const char* order_file)
      : reorderer()
  {
    read_order_file(order_file);
  }

private:
  void reorder_tasks(::std::vector<int>& tasks, ::std::unordered_map<int, reorderer_payload>&) override
  {
    tasks = file_order;
  }

  /* Read the csv schedule file mapping tasks to devices */
  void read_order_file(const char* filename)
  {
    ::std::ifstream file(filename);
    EXPECT(file, "Failed to open order file: '", filename, "'.");

    int current_line = 0;
    for (::std::string line; ::std::getline(file, line); ++current_line)
    {
      ::std::stringstream ss(line);

      int mapping_id = -1;

      int column = 0;
      for (::std::string cell; ::std::getline(ss, cell, ','); ++column)
      {
        if (column == 1)
        {
          mapping_id = ::std::stoi(cell);
        }
      }

      EXPECT(mapping_id >= 0, "Invalid mapping id value '", mapping_id, "' provided on line '", current_line, "'.");

      file_order.push_back(mapping_id);
    }
  }

  ::std::vector<int> file_order;
};

inline ::std::unique_ptr<reorderer> reorderer::make(const char* reorderer_type)
{
  if (!reorderer_type)
  {
    return nullptr;
  }

  const auto reorderer_type_s = ::std::string(reorderer_type);

  if (reorderer_type_s == "random")
  {
    return ::std::make_unique<random_reorderer>();
  }

  if (reorderer_type_s == "heft")
  {
    return ::std::make_unique<heft_reorderer>();
  }

  if (reorderer_type_s == "post_mortem")
  {
    const char* order_file = getenv("CUDASTF_ORDER_FILE");

    EXPECT(order_file, "CUDASTF_TASK_ORDER set to 'post_mortem' but CUDASTF_SCHEDULE_FILE is unset.");
    EXPECT(::std::filesystem::exists(order_file), "CUDASTF_ORDER_FILE '", order_file, "' does not exist");

    return ::std::make_unique<post_mortem_reorderer>(order_file);
  }

  fprintf(stderr, "Invalid CUDASTF_TASK_ORDER value '%s'\n", reorderer_type);
  abort();
}
} // namespace cuda::experimental::stf::reserved
