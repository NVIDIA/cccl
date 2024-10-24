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
 * @brief Implements the generation of a DOT file to visualize the task graph
 *
 * CUDASTF_DOT_FILE
 * CUDASTF_DOT_IGNORE_PREREQS
 * CUDASTF_DOT_COLOR_BY_DEVICE
 * CUDASTF_DOT_REMOVE_DATA_DEPS
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

#include <cuda/experimental/__stf/internal/constants.cuh>
#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>
#include <cuda/experimental/__stf/utility/hash.cuh>
#include <cuda/experimental/__stf/utility/threads.cuh>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_set>

namespace cuda::experimental::stf::reserved
{

/**
 * @brief Some helper type
 */
using IntPairSet = ::std::unordered_set<::std::pair<int, int>, cuda::experimental::stf::hash<::std::pair<int, int>>>;

class dot;

class per_ctx_dot
{
public:
  per_ctx_dot(bool _is_tracing, bool _is_tracing_prereqs)
      : _is_tracing(_is_tracing)
      , _is_tracing_prereqs(_is_tracing_prereqs)
  {}
  ~per_ctx_dot() = default;

  void finish()
  {
    prev_oss.push_back(mv(oss));
  }

  const auto& get_streams() const
  {
    return prev_oss;
  }

  void set_ctx_symbol(::std::string s)
  {
    ctx_symbol = mv(s);
  }

  void add_fence_vertex(int unique_id)
  {
    if (getenv("CUDASTF_DOT_NO_FENCE"))
    {
      return;
    }

    ::std::lock_guard<::std::mutex> guard(mtx);

    vertices.push_back(unique_id);

    // Add a node in the DOT file
    oss << "\"NODE_" << unique_id << "\" [style=\"filled\" fillcolor=\"red\" label=\"task fence\"]\n";
  }

  void add_prereq_vertex(const ::std::string& symbol, int prereq_unique_id)
  {
    if (!is_tracing_prereqs())
    {
      return;
    }

    ::std::lock_guard<::std::mutex> guard(mtx);

    vertices.push_back(prereq_unique_id);

    set_current_color_by_device(guard);
    // Add an entry in the DOT file
    oss << "\"NODE_" << prereq_unique_id << "\" [label=\"" << symbol << "\", style=dashed]\n";
  }

  // Edges are not rendered directly, so that we can decide to filter out
  // some of them later. They are rendered when the graph is finalized.
  //
  // style = 0 => plain
  // style = 1 => dashed
  //
  // Note that while tasks are topologically ordered, when we generate graphs
  // which includes internal async events, we may have (prereq) nodes which
  // are not ordered, so we cannot expect "id_from < id_to"
  void add_edge(int id_from, int id_to, int style = 0)
  {
    if (!is_tracing())
    {
      return;
    }

    ::std::lock_guard<::std::mutex> guard(mtx);

    if (is_discarded(id_from, guard) || is_discarded(id_to, guard))
    {
      return;
    }

    if (style == 1 && getenv("CUDASTF_DOT_NO_FENCE"))
    {
      return;
    }

    auto p = ::std::pair(id_from, id_to);

    // Note that since this is a set, there is no need to check if that
    // was already in the set.
    existing_edges.insert(p);
  }

  template <typename task_type, typename data_type>
  void add_vertex(task_type t)
  {
    // Do this work outside the critical section
    const auto remove_deps = getenv("CUDASTF_DOT_REMOVE_DATA_DEPS");

    ::std::lock_guard<::std::mutex> guard(mtx);

    if (!tracing_enabled)
    {
      discard(t.get_unique_id());
      return;
    }

    set_current_color_by_device(guard);

    vertices.push_back(t.get_unique_id());

    // Add an entry in the DOT file
    oss << "\"NODE_" << t.get_unique_id() << "\" [style=\"filled\" fillcolor=\"" << get_current_color() << "\" label=\""
        << t.get_symbol();

    // Append the text with a list of accessed logical data, and the corresponding access modes
    if (!remove_deps)
    {
      const auto deps = t.get_task_deps();
      for (auto& e : deps)
      {
        data_type d      = e.get_data();
        access_mode mode = e.get_access_mode();
        size_t size      = d.get_data_interface().data_footprint();
        oss << "\\n" << d.get_symbol() << "(" << access_mode_string(mode) << ")(" << size << ") ";
      }
    }

    oss << "\"]\n";
  }

  template <typename task_type>
  void add_vertex_timing(task_type t, float time_ms, int device = -1)
  {
    ::std::lock_guard<::std::mutex> guard(mtx);

    if (!tracing_enabled)
    {
      return;
    }

    oss << "// " << t.get_unique_id() << " : mapping_id=" << t.get_mapping_id() << " time=" << time_ms
        << " device=" << device << "\n";
  }

  // Take a reference to an (unused) `::std::lock_guard<::std::mutex>` to make sure someone ddid take a lock.
  void set_current_color_by_device(::std::lock_guard<::std::mutex>&)
  {
    if (getenv("CUDASTF_DOT_COLOR_BY_DEVICE"))
    {
      int dev;
      cuda_safe_call(cudaGetDevice(&dev));
      EXPECT(dev < sizeof(colors) / sizeof(*colors));
      current_color = colors[dev];
    }
  }

  void set_current_color(const char* color)
  {
    if (!is_tracing())
    {
      return;
    }
    ::std::lock_guard<::std::mutex> guard(mtx);
    current_color = color;
  }

  const char* get_current_color()
  {
    return current_color;
  }

  void change_epoch()
  {
    if (getenv("CUDASTF_DOT_DISPLAY_EPOCHS"))
    {
      ::std::lock_guard<::std::mutex> guard(mtx);
      prev_oss.push_back(mv(oss));
      oss.clear();
    }
  }

  // Change the current tracing mode to enable or disable it dynamically
  void set_tracing(bool enable)
  {
    tracing_enabled = enable;
  }

  void discard(int id)
  {
    single_threaded_section guard(mtx);
    discarded_tasks.insert(id);
  }
  bool is_discarded(int id, ::std::lock_guard<::std::mutex>&) const
  {
    return discarded_tasks.find(id) != discarded_tasks.end();
  }

  static void set_parent_ctx(::std::shared_ptr<per_ctx_dot> parent_dot, ::std::shared_ptr<per_ctx_dot> child_dot)
  {
    parent_dot->children.push_back(child_dot);
    child_dot->parent = mv(parent_dot);
  }

  ::std::shared_ptr<per_ctx_dot> parent;
  ::std::vector<::std::shared_ptr<per_ctx_dot>> children;

  const ::std::string get_ctx_symbol() const
  {
    return ctx_symbol;
  }

private:
  mutable ::std::string ctx_symbol;

  mutable ::std::mutex mtx;

  ::std::vector<int> vertices;

  ::std::unordered_set<int> discarded_tasks;

public:
  // Keep track of existing edges, to make the output possibly look better
  IntPairSet existing_edges;

  bool is_tracing() const
  {
    return _is_tracing;
  }
  bool _is_tracing;
  bool is_tracing_prereqs() const
  {
    return _is_tracing_prereqs;
  }
  bool _is_tracing_prereqs;

  // We may temporarily discard some tasks
  bool tracing_enabled = true;

  // A palette of colors
  const char* const colors[8] = {"#ff5500", "#66ccff", "#9933cc", "#00cc66", "#ffcc00", "#00b3e6", "#cc0066", "#009933"};

  const char* current_color = "white";

public: // XXX protected, friend : dot
  // string for the current epoch
  mutable ::std::ostringstream oss;
  // strings of the previous epochs
  mutable ::std::vector<::std::ostringstream> prev_oss;
};

class dot : public reserved::meyers_singleton<dot>
{
protected:
  dot()
  {
    const char* filename = getenv("CUDASTF_DOT_FILE");
    if (!filename)
    {
      return;
    }

    dot_filename = filename;
    //::std::cout << "Creating a DOT file in " << filename << ::std::endl;

    const char* ignore_prereqs_str = getenv("CUDASTF_DOT_IGNORE_PREREQS");
    tracing_prereqs                = ignore_prereqs_str && atoi(ignore_prereqs_str) == 0;
  }

  ~dot()
  {
    finish();
  }

public:
  void
  print_one_context(::std::ofstream& outFile, size_t& ctx_cnt, bool display_clusters, ::std::shared_ptr<per_ctx_dot> pc)
  {
    // Pick up an identifier in DOT (we may update this value later)
    size_t ctx_id = ctx_cnt++;
    if (display_clusters)
    {
      outFile << "subgraph cluster_" << ctx_id << " {\n";
    }
    size_t epoch_cnt = pc->prev_oss.size();
    for (size_t epoch_id = 0; epoch_id < epoch_cnt; epoch_id++)
    {
      if (epoch_cnt > 1)
      {
        outFile << "subgraph cluster_" << epoch_id << "_" << ctx_id << " {\n";
        outFile << "label=\"epoch " << epoch_id << "\"\n";
      }

      outFile << pc->prev_oss[epoch_id].str();

      if (epoch_cnt > 1)
      {
        outFile << "} // end subgraph cluster_" << epoch_id << "_" << ctx_id << "\n";
      }

      if (!getenv("CUDASTF_DOT_SKIP_CHILDREN"))
      {
        for (auto& child_pc : pc->children)
        {
          print_one_context(outFile, ctx_cnt, display_clusters, child_pc);
        }
      }
    }
    if (display_clusters)
    {
      if (pc->get_ctx_symbol().empty())
      {
        outFile << "label=\"cluster_" << ctx_id << "\"\n";
      }
      else
      {
        outFile << "label=\"" << pc->get_ctx_symbol() << "\"\n";
      }
      outFile << "} // end subgraph cluster_" << ctx_id << "\n";
    }

    for (const auto& e : pc->existing_edges)
    {
      existing_edges.insert(e);
    }
  }

  // This should not need to be called explicitly, unless we are doing some automatic tests for example
  void finish()
  {
    single_threaded_section guard(mtx);

    if (dot_filename.empty())
    {
      return;
    }

    for (const auto& pc : per_ctx)
    {
      pc->finish();
    }

    ::std::ofstream outFile(dot_filename);
    if (outFile.is_open())
    {
      outFile << "digraph {\n";
      size_t ctx_cnt        = 0;
      bool display_clusters = (per_ctx.size() > 1);
      /*
       * For every context, we write the description of the DAG per
       * epoch. Then we write the edges after removing redundant ones.
       */
      for (const auto& pc : per_ctx)
      {
        // If the context has a parent, it will be printed by this parent itself
        if (!pc->parent)
        {
          print_one_context(outFile, ctx_cnt, display_clusters, pc);
        }
      }

      if (!getenv("CUDASTF_DOT_KEEP_REDUNDANT"))
      {
        remove_redundant_edges(existing_edges);
      }

      /* Edges do not have to belong to the cluster (Vertices do) */
      for (const auto& [from, to] : existing_edges)
      {
        outFile << "\"NODE_" << from << "\" -> \"NODE_" << to << "\"\n";
      }

      outFile << "}\n";

      outFile.close();
    }
    else
    {
      ::std::cerr << "Unable to open file: " << dot_filename << ::std::endl;
    }

    dot_filename.clear();
  }

  bool is_tracing() const
  {
    return !dot_filename.empty();
  }

  bool is_tracing_prereqs()
  {
    return tracing_prereqs;
  }

  ::std::vector<::std::shared_ptr<per_ctx_dot>> per_ctx;

private:
  bool reachable(int from, int to, ::std::unordered_set<int>& visited)
  {
    visited.insert(to);

    for (auto p : predecessors[to])
    {
      if (p == from)
      {
        return true;
      }

      if (visited.find(p) == visited.end())
      {
        if (reachable(from, p, visited))
        {
          return true;
        }
      }
    }

    return false;
  }

  // This method will check whether an edge can be removed when there is already a direct path
  void remove_redundant_edges(IntPairSet& edges)
  {
    single_threaded_section guard(mtx);
    // We first dump the set of edges into a map of predecessors per node
    for (const auto& [from, to] : edges)
    {
      predecessors[to].push_back(from);
    }

    // Maybe this is not the most efficient, we could use a vector of
    // pair<int, int> if efficiency matters here.
    IntPairSet edges_cpy = edges;

    // We will put back only those needed
    edges.clear();

    for (const auto& [from, to] : edges_cpy)
    {
      bool keep   = true;
      auto& preds = predecessors[to];
      ::std::sort(preds.begin(), preds.end());

      // To avoid checking for the same pairs many times, we keep track
      // of nodes already tested
      ::std::unordered_set<int> visited;
      for (auto p : preds)
      {
        if (reachable(from, p, visited))
        {
          // Put this edge back in the set of edges
          keep = false;
          break;
        }
      }

      if (keep)
      {
        edges.insert(::std::pair(from, to));
      }
    }
  }

  bool tracing_prereqs = false;

  // Keep track of existing edges, to make the output possibly look better
  IntPairSet existing_edges;

  ::std::unordered_map<int, ::std::vector<int>> predecessors;

  mutable ::std::mutex mtx;

  ::std::string dot_filename;
};

} // namespace cuda::experimental::stf::reserved
