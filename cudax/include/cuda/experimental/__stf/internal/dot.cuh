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
 * CUDASTF_DOT_TIMING
 * CUDASTF_DOT_MAX_DEPTH
 * CUDASTF_DOT_SHOW_FENCE
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
#include <cuda/experimental/__stf/utility/nvtx.cuh>
#include <cuda/experimental/__stf/utility/threads.cuh>
#include <cuda/experimental/__stf/utility/unique_id.cuh>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <stack>
#include <unordered_set>

/**
 * @file
 * @brief Generation of the DOT file to visualize task DAGs
 */

namespace cuda::experimental::stf::reserved
{

/**
 * @brief Some helper type
 */
using IntPairSet = ::std::unordered_set<::std::pair<int, int>, cuda::experimental::stf::hash<::std::pair<int, int>>>;

class dot;

// edge represent dependencies, but we sometimes want to visualize differently
// dependencies which are related to actual task dependencies, and "internal"
// dependencies between asynchronous operations (eg. a task depends on an
// allocation) which are not necessarily useful to visualize.
enum edge_type
{
  plain   = 0,
  prereqs = 1,
  fence   = 2
};

// We have different types of nodes in the graph
enum vertex_type
{
  task_vertex,
  prereq_vertex,
  fence_vertex,
  section_vertex, // collapsed sections depicted as a single node
  freeze_vertex, // freeze and unfreeze operations
  cluster_proxy_vertex // start or end of clusters (invisible nodes)
};

// Information for every vertex (task, prereq, ...), so that we can eventually generate a node for the DAG
struct per_vertex_info
{
  // This color can for example be computed according to the duration of the task, if measured
  ::std::string color;
  ::std::string label;
  ::std::optional<float> timing;
  // is that a task, fence or prereq ?
  vertex_type type;

  int dot_section_id;
};

/**
 * @brief Store dot-related information per STF context.
 *
 * If multiple contexts are created, the specific state of each context is
 * saved here, and common state is stored in the dot singleton class
 */
class per_ctx_dot
{
public:
  per_ctx_dot(bool _is_tracing, bool _is_tracing_prereqs, bool _is_timing)
      : _is_tracing(_is_tracing)
      , _is_tracing_prereqs(_is_tracing_prereqs)
      , _is_timing(_is_timing)
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
    if (!getenv("CUDASTF_DOT_SHOW_FENCE"))
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
  // style = 2 => dashed to fence
  //
  // Note that while tasks are topologically ordered, when we generate graphs
  // which includes internal async events, we may have (prereq) nodes which
  // are not ordered, so we cannot expect "id_from < id_to"
  void add_edge(int id_from, int id_to, edge_type style = edge_type::plain)
  {
    if (!is_tracing())
    {
      return;
    }

    if (!tracing_enabled)
    {
      return;
    }

    ::std::lock_guard<::std::mutex> guard(mtx);

    if (is_discarded(id_from, guard) || is_discarded(id_to, guard))
    {
      return;
    }

    if (style == edge_type::fence && !getenv("CUDASTF_DOT_SHOW_FENCE"))
    {
      return;
    }

    auto p = ::std::pair(id_from, id_to);

    // Note that since this is a set, there is no need to check if that
    // was already in the set.
    existing_edges.insert(p);
  }

  // Used to avoid cyclic dependencies, defined later
  static int get_current_section_id();

  // Save the ID of the task in the "vertices" vector, and associate this ID to
  // the metadata of the task, so that we can generate a node for the task
  // later.
  template <typename task_type, typename data_type>
  void add_vertex_internal(const task_type& t, vertex_type type)
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

    auto& task_metadata = metadata[t.get_unique_id()];

    task_metadata.color = get_current_color();

    task_metadata.dot_section_id = get_current_section_id();

    // We here create the label of the task, which we may augment later with
    // timing information for example
    ::std::ostringstream task_oss;
    task_oss << t.get_symbol();

    // Append the text with a list of accessed logical data, and the corresponding access modes
    if (!remove_deps)
    {
      const auto deps = t.get_task_deps();
      for (auto& e : deps)
      {
        data_type d      = e.get_data();
        access_mode mode = e.get_access_mode();
        size_t size      = d.get_data_interface().data_footprint();
        task_oss << "\\n" << d.get_symbol() << "(" << access_mode_string(mode) << ")(" << size << ") ";
      }
    }

    task_metadata.label = task_oss.str();
    task_metadata.type  = type;
  }

  // Save the ID of the task in the "vertices" vector, and associate this ID to
  // the metadata of the task, so that we can generate a node for the task
  // later.
  template <typename task_type, typename data_type>
  void add_vertex(const task_type& t)
  {
    add_vertex_internal<task_type, data_type>(t, task_vertex);
  }

  // internal freeze indicates if this is a freeze/unfreeze from the user or
  // not and if we need to display an edge, or if it is unnecessary because
  // other dependencies will imply that edge between freeze and unfreeze
  template <typename task_type, typename data_type>
  void add_freeze_vertices(const task_type& freeze_fake_task, const task_type& unfreeze_fake_task, bool user_freeze)
  {
    add_vertex_internal<task_type, data_type>(freeze_fake_task, freeze_vertex);
    add_vertex_internal<task_type, data_type>(unfreeze_fake_task, freeze_vertex);

    if (user_freeze)
    {
      add_edge(freeze_fake_task.get_unique_id(), unfreeze_fake_task.get_unique_id(), edge_type::plain);
    }
  }

  template <typename task_type>
  void add_vertex_timing(const task_type& t, float time_ms, [[maybe_unused]] int device = -1)
  {
    ::std::lock_guard<::std::mutex> guard(mtx);

    if (!tracing_enabled)
    {
      return;
    }

    oss << "// " << t.get_unique_id() << " : mapping_id=" << t.get_mapping_id() << " time=" << time_ms
        << " device=" << device << "\n";

    // Save timing information for this task
    metadata[t.get_unique_id()].timing = time_ms;
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

  void change_stage()
  {
    if (getenv("CUDASTF_DOT_DISPLAY_STAGES"))
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

  const ::std::string& get_ctx_symbol() const
  {
    return ctx_symbol;
  }

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

  bool _is_timing;
  bool is_timing() const
  {
    return _is_timing;
  }

  // We may temporarily discard some tasks
  bool tracing_enabled = true;

  // A palette of colors
  const char* const colors[8] = {"#ff5500", "#66ccff", "#9933cc", "#00cc66", "#ffcc00", "#00b3e6", "#cc0066", "#009933"};

  const char* current_color = "white";

  // Non copyable unique id, ok because we use shared pointers
  unique_id<per_ctx_dot> id;

public: // XXX protected, friend : dot
  // string for the current stage
  mutable ::std::ostringstream oss;
  // strings of the previous stages
  mutable ::std::vector<::std::ostringstream> prev_oss;
  ::std::unordered_map<int /* id */, per_vertex_info> metadata;

private:
  mutable ::std::string ctx_symbol;

  mutable ::std::mutex mtx;

  ::std::vector<int> vertices;

  ::std::unordered_set<int> discarded_tasks;
};

class dot : public reserved::meyers_singleton<dot>
{
public:
  /**
   * @brief A named section in the DOT output to potentially collapse multiple nodes in the same section
   */
  class section
  {
  public:
    // Constructor to initialize symbol and children
    section(::std::string sym)
        : symbol(mv(sym))
        , r(symbol.c_str())
    {
      static_assert(::std::is_move_constructible_v<section>, "section must be move constructible");
      static_assert(::std::is_move_assignable_v<section>, "section must be move assignable");
    }

    class guard
    {
    public:
      guard(::std::string symbol)
      {
        section::push(mv(symbol));
      }

      // Move constructor: transfer ownership and disable the moved-from guard.
      guard(guard&& other) noexcept
          : active(cuda::std::exchange(other.active, false))
      {}

      // Move assignment, disable the moved-from guard
      guard& operator=(guard&& other) noexcept
      {
        if (this != &other)
        {
          // Clean up current resource if needed. We must call the pop method
          // of the existing guard before overwriting it with a new guard.
          if (active)
          {
            section::pop();
          }
          // Transfer ownership.
          active       = other.active;
          other.active = false;
        }
        return *this;
      }

      // Non-copyable
      guard(const guard&)            = delete;
      guard& operator=(const guard&) = delete;

      void end()
      {
        _CCCL_ASSERT(active, "Attempting to end the same section twice.");
        section::pop();
        active = false;
      }

      ~guard()
      {
        if (active)
        {
          section::pop();
        }
      }

    private:
      // Have we called end() ?
      bool active = true;
    };

    static auto& current()
    {
      thread_local ::std::stack<int> s;
      return s;
    }

    static void push(::std::string symbol)
    {
      // We first create a section object, with its unique id
      auto sec = ::std::make_shared<section>(mv(symbol));
      int id   = sec->get_id();

      int parent_id  = current().empty() ? 0 : current().top();
      sec->parent_id = parent_id;

      // Save the section in the map
      dot::instance().map[id] = sec;

      // Add the section to the children of its parent if that was not the root
      if (parent_id > 0)
      {
        dot::instance().map[parent_id]->children_ids.push_back(id);
      }

      // Push the id in the current stack
      current().push(id);

      // The size of the stack is the recursion level of the section
      sec->depth = current().size();
    }

    static void pop()
    {
      _CCCL_ASSERT(current().size() > 0, "Cannot pop, no section was pushed.");
      current().pop();
    }

    /**
     * @brief Get the unique ID of the section
     *
     * Note that returned values start at 1, not 0, we use 0 to designate the
     * lack of a section.
     */
    int get_id() const
    {
      return 1 + int(id);
    }

    const ::std::string& get_symbol() const
    {
      return symbol;
    }

    int get_depth() const
    {
      return depth;
    }

    int parent_id;

    ::std::vector<int> children_ids;

  private:
    int depth = ::std::numeric_limits<int>::min();

    ::std::string symbol;

    // An annotation that has the lifetime of the section
    nvtx_range r;

    // An identifier for that section. This is movable, but non
    // copyable, but we manipulate section by the means of shared_ptr.
    unique_id<section> id;
  };

protected:
  dot()
  {
    ::std::lock_guard<::std::mutex> lock(mtx);

    const char* filename = getenv("CUDASTF_DOT_FILE");
    if (!filename)
    {
      return;
    }

    dot_filename = filename;
    //::std::cout << "Creating a DOT file in " << filename << ::std::endl;

    const char* ignore_prereqs_str = getenv("CUDASTF_DOT_IGNORE_PREREQS");
    tracing_prereqs                = ignore_prereqs_str && atoi(ignore_prereqs_str) == 0;

    const char* dot_timing_str = ::std::getenv("CUDASTF_DOT_TIMING");
    enable_timing              = (dot_timing_str && atoi(dot_timing_str) != 0);
  }

  ~dot()
  {
    finish();
  }

public:
  bool is_tracing() const
  {
    return !dot_filename.empty();
  }

  bool is_tracing_prereqs()
  {
    return tracing_prereqs;
  }

  bool is_timing() const
  {
    return enable_timing;
  }

  // Add a context to the vector of contexts we track
  void track_ctx(::std::shared_ptr<per_ctx_dot> pc)
  {
    ::std::lock_guard<::std::mutex> lock(mtx);

    per_ctx.push_back(mv(pc));
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

    collapse_sections();

    // Now we have executed all tasks, so we can compute the average execution
    // times, and update the colors appropriately if needed.
    update_colors_with_timing();

    ::std::ofstream outFile(dot_filename);
    if (outFile.is_open())
    {
      outFile << "digraph {\n";
      size_t ctx_cnt        = 0;
      bool display_clusters = (per_ctx.size() > 1);
      /*
       * For every context, we write the description of the DAG per
       * stage. Then we write the edges after removing redundant ones.
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

      compute_critical_path(outFile);

      /* Edges do not have to belong to the cluster (Vertices do) */
      for (const auto& [from, to] : existing_edges)
      {
        outFile << "\"NODE_" << from << "\" -> \"NODE_" << to << "\"\n";
      }

      // Update node properties such as labels and colors now that we have all information
      vertex_count = 0;
      for (const auto& pc : per_ctx)
      {
        for (const auto& p : pc->metadata)
        {
          outFile << "\"NODE_" << p.first << "\" [style=\"filled\" fillcolor=\"" << p.second.color << "\" label=\""
                  << p.second.label << "\"]\n";
          vertex_count++;
        }
      }

      edge_count = existing_edges.size();

      outFile << "// Edge   count : " << edge_count << "\n";
      outFile << "// Vertex count : " << vertex_count << "\n";

      outFile << "}\n";

      outFile.close();
    }
    else
    {
      ::std::cerr << "Unable to open file: " << dot_filename << ::std::endl;
    }

    const char* stats_filename_str = getenv("CUDASTF_DOT_STATS_FILE");
    if (stats_filename_str)
    {
      ::std::string stats_filename = stats_filename_str;
      ::std::ofstream statsFile(stats_filename);
      if (statsFile.is_open())
      {
        statsFile << "#nedges,nvertices,total_work,critical_path\n";

        // to display an optional value or NA
        auto formatOptional = [](const ::std::optional<float>& opt) -> ::std::string {
          return opt ? ::std::to_string(*opt) : "NA";
        };

        statsFile << edge_count << "," << vertex_count << "," << formatOptional(total_work) << ","
                  << formatOptional(critical_path) << "\n";

        statsFile.close();
      }
      else
      {
        ::std::cerr << "Unable to open file: " << stats_filename << ::std::endl;
      }
    }

    dot_filename.clear();
  }

private:
  void
  print_one_context(::std::ofstream& outFile, size_t& ctx_cnt, bool display_clusters, ::std::shared_ptr<per_ctx_dot> pc)
  {
    // Pick up an identifier in DOT (we may update this value later)
    size_t ctx_id = ctx_cnt++;
    if (display_clusters)
    {
      outFile << "subgraph cluster_" << ctx_id << " {\n";
    }
    size_t stage_cnt = pc->prev_oss.size();
    for (size_t stage_id = 0; stage_id < stage_cnt; stage_id++)
    {
      if (stage_cnt > 1)
      {
        outFile << "subgraph cluster_" << stage_id << "_" << ctx_id << " {\n";
        outFile << "label=\"stage " << stage_id << "\"\n";
      }

      outFile << pc->prev_oss[stage_id].str();

      if (stage_cnt > 1)
      {
        outFile << "} // end subgraph cluster_" << stage_id << "_" << ctx_id << "\n";
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

    /* Put nodes which belong to a section into their clusters */
    ::std::unordered_map<int, ::std::vector<int>> section_id_to_nodes;
    for (auto& p : pc->metadata)
    {
      // p.first task id, p.second metadata
      int dot_section_id = p.second.dot_section_id;
      if (dot_section_id > 0)
      {
        section_id_to_nodes[dot_section_id].push_back(p.first);
      }
    }

    /* Display all sections recursively */
    for (auto [id, sec_ptr] : map)
    {
      // Select root nodes only
      if (sec_ptr->parent_id == 0)
      {
        print_section(outFile, id, section_id_to_nodes);
      }
    }
  }

  /**
   * @brief Add a dashed box around a section and its children
   */
  void
  print_section(::std::ofstream& outFile, int id, ::std::unordered_map<int, ::std::vector<int>>& section_id_to_nodes)
  {
    // Stop printing sections if they are deeper than the max depth (if defined)
    const char* env_max_depth = getenv("CUDASTF_DOT_MAX_DEPTH");
    if (env_max_depth && (atoi(env_max_depth) < map[id]->get_depth()))
    {
      return;
    }

    outFile << "subgraph cluster_section_" << ::std::to_string(id) << " {\n ";

    // Display all children too to have nested boxes
    for (int children_ids : map[id]->children_ids)
    {
      print_section(outFile, children_ids, section_id_to_nodes);
    }

    // style of the box
    outFile << "    color=black;\n";
    outFile << "    style=dashed\n";
    outFile << "    label=\"" + map[id]->get_symbol() + "\"\n";

    // Put all nodes which belong to this section
    for (auto i : section_id_to_nodes[id])
    {
      outFile << "    \"NODE_" + ::std::to_string(i) + "\"\n";
    }

    outFile << "} // end subgraph cluster_section_" << ::std::to_string(id) << "\n ";
  }

  // This will update colors if necessary
  void update_colors_with_timing()
  {
    if (!enable_timing)
    {
      return;
    }

    float sum  = 0.0;
    size_t cnt = 0;
    // First go over all tasks which have some timing and compute the average duration
    for (const auto& pc : per_ctx)
    {
      for (const auto& p : pc->metadata)
      {
        if (p.second.timing.has_value())
        {
          float ms = p.second.timing.value();
          cnt++;
          sum += ms;
        }
      }
    }

    if (cnt > 0)
    {
      float avg = sum / cnt;

      // Update colors associated to tasks with timing now in order to
      // illustrate how long they take to execute relative to the average
      for (auto& pc : per_ctx)
      {
        for (auto& p : pc->metadata)
        {
          if (p.second.timing.has_value())
          {
            float ms       = p.second.timing.value();
            p.second.color = get_color_for_duration(ms, avg);
            p.second.label += "\ntiming: " + ::std::to_string(ms) + " ms\n";
          }
        }
      }
    }
  }

  /**
   * @brief Combine two nodes identified by "src_id" and "dst_id" into a single
   * updated one ("dst_id"), redirecting edges and combining timing if
   * necessary
   */
  void merge_nodes(per_ctx_dot& pc, int dst_id, int src_id)
  {
    // ::std::unordered_map<int /* id */, per_vertex_info> metadata;

    // Get src_id from the map and remove it
    auto it = pc.metadata.find(src_id);
    assert(it != pc.metadata.end());
    per_vertex_info src = mv(it->second);
    pc.metadata.erase(it);

    // If there was some timing associated to either src or dst, update timing
    auto& dst = pc.metadata[dst_id];
    if (dst.timing.has_value() || src.timing.has_value())
    {
      dst.timing =
        (src.timing.has_value() ? src.timing.value() : 0.0f) + (dst.timing.has_value() ? dst.timing.value() : 0.0f);
    }

    // Replace edges if necessary
    IntPairSet new_edges;
    for (auto& [from, to] : pc.existing_edges)
    {
      int new_from = (from == src_id) ? dst_id : from;
      int new_to   = (to == src_id) ? dst_id : to;

      if (new_from != new_to)
      {
        _CCCL_ASSERT(new_from < new_to, "invalid edge");
        new_edges.insert(std::make_pair(new_from, new_to));
      }
    }
    ::std::swap(new_edges, pc.existing_edges);
  }

  /**
   * @brief Collapse nodes which are parts of sections deeper than the value
   *        specified in CUDASTF_DOT_MAX_DEPTH
   */
  void collapse_sections()
  {
    const char* env_depth = getenv("CUDASTF_DOT_MAX_DEPTH");
    if (!env_depth)
    {
      return;
    }

    const int max_depth = atoi(env_depth);

    // First go over all tasks which have some timing and compute the average duration
    for (auto& pc : per_ctx)
    {
      // key : section id, value : vector of IDs which should be condensed into a node
      ::std::unordered_map<int, ::std::vector<int>> to_condense;

      for (auto& p : pc->metadata)
      {
        // p.first task id, p.second metadata
        auto& task_info    = p.second;
        int dot_section_id = task_info.dot_section_id;
        if (dot_section_id > 0)
        {
          // Note we do not use a reference here, because we are maybe going to
          // update the pointer, and we do not want to update its content
          // instead when setting sec.
          ::std::shared_ptr<section> sec = dot::instance().map[dot_section_id];
          assert(sec);

          int depth      = sec->get_depth();
          int section_id = task_info.dot_section_id;

          if (depth >= max_depth + 1)
          {
            /* Find the parent at depth (max_depth + 1)*/
            while (depth > max_depth + 1)
            {
              section_id = sec->parent_id;
              _CCCL_ASSERT(section_id != 0, "invalid value");
              sec = dot::instance().map[section_id];
              depth--;
            }

            _CCCL_ASSERT(depth == max_depth + 1, "invalid value");

            /* Add this node to the list of nodes which are "equivalent" to section_id */
            to_condense[section_id].push_back(p.first);
          }
        }
      }

      for (auto& p : to_condense)
      {
        ::std::sort(p.second.begin(), p.second.end());

        // For every section ID, we get the vector of nodes to condense. We pick the first node, and "merge" other nodes
        // with it.

        // Merge all tasks in the vector with the first entry, and then rename
        // the first entry to take the name of the section.
        for (size_t i = 1; i < p.second.size(); i++)
        {
          // Fuse i-th entry with the first one
          merge_nodes(*pc, p.second[0], p.second[i]);
        }

        // Rename the task that remains to have the label of the section
        ::std::shared_ptr<section> sec  = dot::instance().map[p.first];
        pc->metadata[p.second[0]].label = sec->get_symbol();

        // Assign the node to the parent of the section it corresponds to
        pc->metadata[p.second[0]].dot_section_id = sec->parent_id;
      }
    }
  }

private:
  // Function to get a color based on task duration relative to the average
  ::std::string get_color_for_duration(double duration, double avg_duration)
  {
    // Define thresholds relative to the average duration
    const double very_short_threshold = 0.5 * avg_duration; // < 50% of avg
    const double short_threshold      = 0.8 * avg_duration; // < 80% of avg
    const double long_threshold       = 1.5 * avg_duration; // > 150% of avg
    const double very_long_threshold  = 2.0 * avg_duration; // > 200% of avg

    // Return color based on duration thresholds
    if (duration < very_short_threshold)
    {
      return "#b6e3b6"; // Light Green for Very Short tasks
    }
    else if (duration < short_threshold)
    {
      return "#69b369"; // Green for Short tasks
    }
    else if (duration <= long_threshold)
    {
      return "#ffd966"; // Yellow for Around Average tasks
    }
    else if (duration <= very_long_threshold)
    {
      return "#ffb84d"; // Orange for Long tasks
    }
    else
    {
      return "#ff6666"; // Red for Very Long tasks
    }
  }

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

  void compute_critical_path(::std::ofstream& outFile)
  {
    if (!enable_timing)
    {
      return;
    }

    single_threaded_section guard(mtx);

    // Total Work (T1) in Cilk terminology
    float t1 = 0.0f;

    ::std::unordered_map<int, ::std::vector<int>> predecessors;
    ::std::unordered_map<int, ::std::vector<int>> successors;

    ::std::unordered_map<int, float> dist;
    ::std::unordered_map<int, int> indegree;

    ::std::unordered_map<int, float> durations; // there might be missing entries for non-timed nodes (eg. events which
                                                // are not tasks)

    ::std::unordered_map<int, int> path_predecessor;

    // Gather durations
    for (const auto& pc : per_ctx)
    {
      // Per task
      for (const auto& p : pc->metadata)
      {
        if (p.second.timing.has_value())
        {
          float ms = p.second.timing.value();
          // Total work is simply the sum of all work
          t1 += ms;
          durations[p.first] = ms;
        }
      }

      // We go through edges to find the predecessors of every node
      for (const auto& [from, to] : pc->existing_edges)
      {
        predecessors[to].push_back(from);
        successors[from].push_back(to);

        // For nodes which don't have timing information, we assume a 0.0 duration. This will make a simpler algorithm
        if (durations.find(from) == durations.end())
        {
          durations[from] = 0.0f;
        }

        if (durations.find(to) == durations.end())
        {
          durations[to] = 0.0f;
        }
      }
    }

    // Topological sort using Kahn's algorithm
    ::std::queue<int> q;
    for (const auto& p : durations)
    {
      int id = p.first;

      // how many input deps for that node ?
      indegree[id] = predecessors[id].size();
      // how much time is needed to compute that node (and also its predecessors)
      dist[id] = p.second;
      // we will backtrack which were the tasks in the critical path
      path_predecessor[p.first] = -1;

      if (indegree[id] == 0)
      {
        q.push(id);
      }
    }

    // Process each node in topological order
    while (!q.empty())
    {
      int u = q.front();
      q.pop();

      for (int v : successors[u])
      {
        if (dist[u] + durations[v] > dist[v])
        {
          dist[v]             = dist[u] + durations[v];
          path_predecessor[v] = u;
        }
        indegree[v]--;
        if (indegree[v] == 0)
        {
          q.push(v);
        }
      }
    }

    // The longest path in dist gives the critical path (Tinfinity)
    float max_dist = 0.0f;
    int max_ind    = -1;
    for (const auto& pair : dist)
    {
      if (pair.second > max_dist)
      {
        max_dist = pair.second;
        max_ind  = pair.first;
      }
    }

    // Highlight the critical path in the DAG by backtracking in the path
    // until we reach the first node
    int next = max_ind;
    while (next != -1)
    {
      outFile << "\"NODE_" << next << "\" [color=red, penwidth=10]\n";
      next = path_predecessor[next];
    }

    outFile << "// T1 = " << t1 << ::std::endl;
    outFile << "// Tinf = " << max_dist << ::std::endl;

    critical_path = max_dist;
    total_work    = t1;
  }

  // Are we tracing asynchronous events in addition to tasks ? (eg. copies, allocations, ...)
  bool tracing_prereqs = false;

  // Are we measuring the duration of tasks ?
  bool enable_timing = false;

  // Keep track of existing edges, to make the output possibly look better
  IntPairSet existing_edges;

  ::std::unordered_map<int, ::std::vector<int>> predecessors;

  ::std::string dot_filename;

  // Map to get dot sections from their ID
  ::std::unordered_map<int, ::std::shared_ptr<section>> map;

  // Stats
  ::std::optional<float> critical_path; // Tinf
  ::std::optional<float> total_work; // T1
  size_t edge_count;
  size_t vertex_count;

private:
  mutable ::std::mutex mtx;

  // A vector that keeps track of all per context stored data
  ::std::vector<::std::shared_ptr<per_ctx_dot>> per_ctx;
};

inline int per_ctx_dot::get_current_section_id()
{
  // Get the stack of IDs, if it's empty return 0, otherwise the id of the section (which are numbered starting from 1)
  auto& s = dot::section::current();
  return s.size() == 0 ? 0 : s.top();
}

} // namespace cuda::experimental::stf::reserved
