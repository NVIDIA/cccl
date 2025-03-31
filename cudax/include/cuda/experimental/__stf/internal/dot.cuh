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

int get_next_prereq_unique_id();

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

  // id of the vertex
  int original_id;

  // id of the vertex after collapsing : this is the id of the vertex that
  // represents all nodes that were merged in a section.
  int representative_id;

  int dot_section_id;
  int ctx_id;
};

class dot_section;

inline ::std::shared_ptr<dot_section>& dot_get_section_by_id(int id);

class per_ctx_dot;

/**
 * @brief A named section in the DOT output to potentially collapse multiple
 * nodes in the same section, this can also be created automatically when there
 * are nested contexts.
 */
class dot_section
{
public:
  // Constructor to initialize symbol and children
  dot_section(::std::string sym)
      : symbol(mv(sym))
  {
    static_assert(::std::is_move_constructible_v<dot_section>, "dot_section must be move constructible");
    static_assert(::std::is_move_assignable_v<dot_section>, "dot_section must be move assignable");
  }

  class guard
  {
  public:
    guard(::std::shared_ptr<per_ctx_dot> pc_, ::std::string symbol)
        : pc(mv(pc_))
    {
      dot_section::push(pc, mv(symbol));
    }

    // Move constructor: transfer ownership and disable the moved-from guard.
    guard(guard&& other) noexcept
        : pc(other.pc)
        , active(cuda::std::exchange(other.active, false))
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
          dot_section::pop(pc);
        }
        // Transfer ownership.
        pc           = other.pc;
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
      dot_section::pop(pc);
      active = false;
    }

    ~guard()
    {
      if (active)
      {
        dot_section::pop(pc);
      }
    }

  private:
    // per context dot object of the context where this
    // section was created
    ::std::shared_ptr<per_ctx_dot> pc;

    // Have we called end() ?
    bool active = true;
  };

  static void push(::std::shared_ptr<per_ctx_dot>& pc, ::std::string symbol);
  static void pop(::std::shared_ptr<per_ctx_dot>& pc);

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

  // rename to to_string()
  const ::std::string& get_symbol() const
  {
    return symbol;
  }

  int get_depth() const
  {
    return depth;
  }

  void set_depth(int d)
  {
    depth = d;
  }

  // id of the parent section (0 if this is a root)
  int parent_id;

  // ids of the children sections (if any)
  ::std::vector<int> children_ids;

  ::std::string symbol;

private:
  int depth = ::std::numeric_limits<int>::min();

  // An identifier for that section. This is movable, but non
  // copyable, but we manipulate section by the means of shared_ptr.
  unique_id<dot_section> id;
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
  {
    auto sec       = ::std::make_shared<dot_section>("context");
    int id         = sec->get_id();
    sec->parent_id = 0; // until we call set_parent_ctx, this is a root node

    fprintf(stderr, "Creating per_ctx_dot section id %d ctx id %d\n", id, get_unique_id());

    // Save the section in the map
    dot_get_section_by_id(id) = sec;

    // Push on the stack associated to this context
    section_id_stack.push_back(id);

    // We could implement a destructor that pops the section of the context
    // but this is done automatically
  }

  void set_ctx_symbol(::std::string s)
  {
    ctx_symbol = mv(s);

    // When creating a per-context dot structure, the first entry of the section stack is for the ctx itself
    int ctx_section_id                            = section_id_stack[0];
    dot_get_section_by_id(ctx_section_id)->symbol = ctx_symbol;
  }

  void add_fence_vertex(int unique_id)
  {
    if (!getenv("CUDASTF_DOT_SHOW_FENCE"))
    {
      return;
    }

    ::std::lock_guard<::std::mutex> guard(mtx);

    vertices.push_back(unique_id);

    auto& m = metadata[unique_id];
    m.color = "red";

    m.dot_section_id = section_id_stack.back();
    m.label          = "task fence";

    m.type = fence_vertex;

    m.original_id       = unique_id;
    m.representative_id = unique_id;
  }

  // To connect a subcontext and its parent context, we can say which events a context depends on
  //
  // This is managed by introducing extra edges, and one extra "fake" vertex,
  // so there is nothing specific to do when collapsing.
  void ctx_add_input_id(int prereq_unique_id)
  {
    if (!is_tracing())
    {
      return;
    }

    // If this is the first input we select the ID of the proxy start
    if (!proxy_start_unique_id.has_value())
    {
      proxy_start_unique_id = get_next_prereq_unique_id();
      add_ctx_proxy_vertex(proxy_start_unique_id.value());

      // We create a dependency between in/out nodes (invisible) so that we can
      // compute the critical path even with empty ctx
      // do this when both start and end are defined
      if (proxy_end_unique_id.has_value())
      {
        add_edge(proxy_start_unique_id.value(), proxy_end_unique_id.value(), edge_type::prereqs);
      }
    }

    add_edge(prereq_unique_id, proxy_start_unique_id.value(), edge_type::prereqs);
  }

  // Same as ctx_add_input_id for output dependencies : these are the events which depend on that context
  void ctx_add_output_id(int prereq_unique_id)
  {
    if (!is_tracing())
    {
      return;
    }

    if (!proxy_end_unique_id.has_value())
    {
      proxy_end_unique_id = get_next_prereq_unique_id();
      add_ctx_proxy_vertex(proxy_end_unique_id.value());

      // We create a dependency between in/out nodes (invisible) so that we can
      // compute the critical path even with empty ctx
      // do this when both start and end are defined
      if (proxy_start_unique_id.has_value())
      {
        add_edge(proxy_start_unique_id.value(), proxy_end_unique_id.value(), edge_type::prereqs);
      }
    }

    add_edge(proxy_end_unique_id.value(), prereq_unique_id, edge_type::prereqs);
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

    auto& m = metadata[prereq_unique_id];
    m.color = get_current_color();

    m.dot_section_id = section_id_stack.back();
    m.ctx_id         = get_unique_id();

    m.label = symbol;

    m.type = prereq_vertex;

    m.original_id       = prereq_unique_id;
    m.representative_id = prereq_unique_id;
  }

  // Define vertices which represent the beginning or the end of a (nested) context
  void add_ctx_proxy_vertex(int prereq_unique_id)
  {
    if (!is_tracing())
    {
      return;
    }

    ::std::lock_guard<::std::mutex> guard(mtx);

    auto& m          = metadata[prereq_unique_id];
    m.color          = get_current_color();
    m.dot_section_id = section_id_stack[0];
    m.ctx_id         = get_unique_id();
    m.label          = "proxy";
    m.type           = cluster_proxy_vertex;

    m.original_id       = prereq_unique_id;
    m.representative_id = prereq_unique_id;
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

    task_metadata.dot_section_id = section_id_stack.back();
    task_metadata.ctx_id         = get_unique_id();

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

    task_metadata.original_id       = t.get_unique_id();
    task_metadata.representative_id = t.get_unique_id();
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

    // Save timing information for this task
    metadata[t.get_unique_id()].timing = time_ms;
  }

  // Take a reference to an (unused) `::std::lock_guard<::std::mutex>` to make sure someone did take a lock.
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

  // Change the current tracing mode to enable or disable it dynamically
  void set_tracing(bool enable)
  {
    tracing_enabled = enable;
  }

  void discard(int id)
  {
    discarded_tasks.insert(id);
  }

  bool is_discarded(int id, ::std::lock_guard<::std::mutex>&) const
  {
    return discarded_tasks.find(id) != discarded_tasks.end();
  }

  static void set_parent_ctx(::std::shared_ptr<per_ctx_dot> parent_dot, ::std::shared_ptr<per_ctx_dot> child_dot)
  {
    // Save the ID of the current section in the parent context (this id may
    // describe an actual user-defined section or a context)
    int parent_section_id = parent_dot->section_id_stack.back();

    // Section automatically associated with the child context : at the bottom of the stack of the ctx
    int child_ctx_section_id = child_dot->section_id_stack[0];
    dot_get_section_by_id(parent_section_id)->children_ids.push_back(child_ctx_section_id);
    dot_get_section_by_id(child_ctx_section_id)->parent_id = parent_section_id;

    fprintf(stderr, "set_parent_ctx sec %d is parent of sec %d\n", parent_section_id, child_ctx_section_id);
  }

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

  // Get the unique id of this per-context descriptor
  int get_unique_id() const
  {
    return int(id);
  }

public:
  // per-context vertices
  ::std::unordered_map<int /* id */, per_vertex_info> metadata;

  // IDs of the sections in this context
  ::std::vector<int> section_id_stack;

private:
  mutable ::std::string ctx_symbol;

  mutable ::std::mutex mtx;

  ::std::vector<int> vertices;

  ::std::unordered_set<int> discarded_tasks;

  ::std::optional<int> proxy_start_unique_id;
  ::std::optional<int> proxy_end_unique_id;
};

class dot : public reserved::meyers_singleton<dot>
{
public:

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

  // Add a context to the vector of contexts we need to depict in DOT
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

    /* Collect all edges and vertices from all contexts */
    for (const auto& pc : per_ctx)
    {
      for (const auto& e : pc->existing_edges)
      {
        all_edges.insert(e);
      }

      for (const auto& v : pc->metadata)
      {
        all_vertices[v.first] = v.second;
      }
    }

    // Find root sections (those with no parents)
    ::std::vector<::std::shared_ptr<dot_section>> root_sections;
    for (auto [id, sec] : section_map)
    {
      if (sec->parent_id == 0)
      {
        root_sections.push_back(sec);
      }
    }

    // Recursively compute the depth of the different sections starting from roots
    for (auto& sec : root_sections)
    {
      compute_section_depth(sec, 0);
    }

    collapse_sections();

    remove_freeze_nodes();

    // Now we have executed all tasks, so we can compute the average execution
    // times, and update the colors appropriately if needed.
    update_colors_with_timing();

    if (!getenv("CUDASTF_DOT_KEEP_REDUNDANT"))
    {
      remove_redundant_edges();
    }

    ::std::ofstream outFile(dot_filename);
    if (outFile.is_open())
    {
      outFile << "digraph {\n";

      compute_critical_path(outFile);

      // We only put the root level (context) in a box if there are more than
      // one root sections
      bool display_top_cluster = root_sections.size() > 1;

      for (auto& sec : root_sections)
      {
        print_section(outFile, sec, display_top_cluster);
      }

      /* Edges do not have to belong to the cluster (Vertices do) */
      for (const auto& [from, to] : all_edges)
      {
        outFile << "\"NODE_" << from << "\" -> \"NODE_" << to << "\"\n";
      }

      edge_count = all_edges.size();

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
  // Recursively compute depth
  void compute_section_depth(::std::shared_ptr<dot_section>& sec, int current_depth)
  {
    sec->set_depth(current_depth);
    for (int child_id : sec->children_ids)
    {
      compute_section_depth(section_map[child_id], current_depth + 1);
    }
  }

  void print_section(::std::ofstream& outFile, ::std::shared_ptr<dot_section> sec, bool display_cluster, int depth = 0)
  {
    int section_id = sec->get_id();
    fprintf(stderr, "print section %d, depth %d\n", section_id, depth);

    if (display_cluster)
    {
      outFile << "subgraph cluster_" << section_id << " {\n";

      _CCCL_ASSERT(!sec->symbol.empty(), "no symbol for section");

      outFile << "label=\"" << sec->symbol << "\"\n";
    }

    // Put all nodes which belong to this section
    for (auto& v : all_vertices)
    {
      if (v.second.dot_section_id == section_id)
      {
        // Select the display style of the node based on the type of vertex
        ::std::string style;
        ::std::string shape;
        switch (v.second.type)
        {
          case task_vertex:
          case fence_vertex:
          case section_vertex:
            style = "filled";
            break;
          case freeze_vertex:
          case prereq_vertex:
            style = "dashed";
            break;
          case cluster_proxy_vertex:
            style = "dashed"; //"invis";
            shape = "point";
            break;
          default:
            fprintf(stderr, "error: unknown vertex type\n");
            abort();
        };

        outFile << "\"NODE_" << v.first << "\" [style=\"" << style;

        if (!shape.empty())
        {
          outFile << "\" shape=\"" << shape;
        }

        outFile << "\" fillcolor=\"" << v.second.color << "\" label=\"" << v.second.label << "\"]\n";
      }
    }

    for (int child_id : sec->children_ids)
    {
      print_section(outFile, section_map[child_id], true, depth + 1);
    }

    if (display_cluster)
    {
      outFile << "} // end subgraph cluster_" << section_id << "\n";
    }
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
    for (const auto& p : all_vertices)
    {
      if (p.second.timing.has_value())
      {
        float ms = p.second.timing.value();
        cnt++;
        sum += ms;
      }
    }

    if (cnt > 0)
    {
      float avg = sum / cnt;

      // Update colors associated to tasks with timing now in order to
      // illustrate how long they take to execute relative to the average
      for (auto& p : all_vertices)
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

  /**
   * @brief Combine two nodes identified by "src_id" and "dst_id" into a single
   * updated one ("dst_id"), redirecting edges and combining timing if
   * necessary
   */
  void merge_nodes(int dst_id, int src_id)
  {
    // If there was some timing associated to either src or dst, update timing
    auto& src = all_vertices[src_id];
    auto& dst = all_vertices[dst_id];
    if (dst.timing.has_value() || src.timing.has_value())
    {
      dst.timing =
        (src.timing.has_value() ? src.timing.value() : 0.0f) + (dst.timing.has_value() ? dst.timing.value() : 0.0f);
    }
  }

  // Remove one vertex and replace incoming/outgoing edges
  void collapse_vertex(int vertex_id)
  {
    /* We look for incoming edges, and outgoing edges */
    ::std::vector<int> in;
    ::std::vector<int> out;

    IntPairSet new_edges;

    for (auto& [from, to] : all_edges)
    {
      bool keep_edge = true;
      if (from == vertex_id)
      {
        out.push_back(to);
        keep_edge = false;
      }

      if (to == vertex_id)
      {
        in.push_back(from);
        keep_edge = false;
      }

      if (keep_edge)
      {
        // Create a copy of the existing edge
        new_edges.insert(::std::make_pair(from, to));
      }
    }

    // Create new edges between all pairs of in/out
    for (auto& from : in)
    {
      for (auto& to : out)
      {
        new_edges.insert(::std::make_pair(from, to));
      }
    }

    ::std::swap(new_edges, all_edges);
  }

  void remove_freeze_nodes()
  {
    if (getenv("CUDASTF_DOT_KEEP_FREEZE"))
    {
      return;
    }

    // Since we are going to potentially remove nodes, we create a copy first
    // to iterate on it, and remove from the original structure if necessary
    decltype(all_vertices) copy_metadata = all_vertices;

    // p.first = vertex id, p.second = per_vertex_info
    for (auto& p : copy_metadata)
    {
      if (p.second.type == freeze_vertex)
      {
        collapse_vertex(p.first);

        // remove this collapsed vertex from the unordered_map
        all_vertices.erase(p.first);
      }
    }
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

    // key : section id, value : vector of IDs which should be condensed into a node
    ::std::unordered_map<int, ::std::vector<int>> to_condense;

    // Go over all vertices (tasks, prereqs, ..). If they are in a section
    // which level is below the threshold, add the vertex id in the
    // to_condense map.
    for (auto& p : all_vertices)
    {
      // p.first task id, p.second metadata
      auto& task_info = p.second;

      int dot_section_id = task_info.dot_section_id;
      if (dot_section_id > 0)
      {
        // Note we do not use a reference here, because we are maybe going to
        // update the pointer, and we do not want to update its content
        // instead when setting sec.
        ::std::shared_ptr<dot_section> sec = dot::instance().section_map[dot_section_id];
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
            sec = dot::instance().section_map[section_id];
            depth--;
          }

          _CCCL_ASSERT(depth == max_depth + 1, "invalid value");

          /* Add this node to the list of nodes which are "equivalent" to section_id */
          to_condense[section_id].push_back(p.first);
        }
      }
    }

    // For each group of tasks
    for (auto& p : to_condense)
    {
      ::std::sort(p.second.begin(), p.second.end());

      // For every section ID, we get the vector of nodes to condense. We pick the first node, and "merge" other nodes
      // with it.
      for (size_t i = 0; i < p.second.size(); i++)
      {
        all_vertices[p.second[i]].representative_id = p.second[0];
      }

      // Merge all tasks in the vector with the first entry, and then rename
      // the first entry to take the name of the section.
      for (size_t i = 1; i < p.second.size(); i++)
      {
        // Fuse the content (eg. timing) of the i-th entry with the first one
        merge_nodes(p.second[0], p.second[i]);
      }

      // Condense edges

      // Rename the task that remains to have the label of the section
      ::std::shared_ptr<dot_section> sec = dot::instance().section_map[p.first];
      all_vertices[p.second[0]].label    = sec->get_symbol();

      // Assign the node to the parent of the section it corresponds to
      all_vertices[p.second[0]].dot_section_id = sec->parent_id;

      all_vertices[p.second[0]].type = section_vertex;
    }

    // Replace or condense edges
    IntPairSet new_edges;
    for (auto& [from, to] : all_edges)
    {
      _CCCL_ASSERT(all_vertices.find(from) != all_vertices.end(), "edge invalid from");
      _CCCL_ASSERT(all_vertices.find(to) != all_vertices.end(), "edge invalid to");

      int new_from = all_vertices[from].representative_id;
      int new_to   = all_vertices[to].representative_id;

      // Remove edges internal to a section
      if (new_from != new_to)
      {
        // XXX_CCCL_ASSERT(new_from < new_to, "invalid edge");
        // insert the edge (if it does not exist already)
        new_edges.insert(std::make_pair(new_from, new_to));
      }
    }
    ::std::swap(new_edges, all_edges);

    // Remove vertices which have been collapsed
    for (auto it = all_vertices.begin(); it != all_vertices.end();)
    {
      const auto& info = it->second;
      if (info.representative_id != info.original_id)
      {
        it = all_vertices.erase(it); // erase returns the next valid iterator
      }
      else
      {
        ++it;
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

  // check if there exists a path between "from" and "to"
  bool reachable(
    int from, int to, ::std::unordered_map<int, ::std::vector<int>>& predecessors, ::std::unordered_set<int>& visited)
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
        if (reachable(from, p, predecessors, visited))
        {
          return true;
        }
      }
    }

    return false;
  }

  // This method will check whether an edge can be removed when there is already a direct path
  void remove_redundant_edges()
  {
    single_threaded_section guard(mtx);

    ::std::unordered_map<int, ::std::vector<int>> predecessors;

    // We first dump the set of edges into a map of predecessors per node
    for (const auto& [from, to] : all_edges)
    {
      predecessors[to].push_back(from);
    }

    IntPairSet new_edges;

    for (const auto& [from, to] : all_edges)
    {
      bool keep   = true;
      auto& preds = predecessors[to];
      ::std::sort(preds.begin(), preds.end());

      // To avoid checking for the same pairs many times, we keep track
      // of nodes already tested
      ::std::unordered_set<int> visited;
      for (auto p : preds)
      {
        // Note that it is based on predecessor lists before pruning nodes!
        if (reachable(from, p, predecessors, visited))
        {
          // Put this edge back in the set of edges
          keep = false;
          break;
        }
      }

      if (keep)
      {
        new_edges.insert(::std::pair(from, to));
      }
    }

    ::std::swap(new_edges, all_edges);
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

    // Per vertex (task, prereqs, ...)
    for (const auto& p : all_vertices)
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
    for (const auto& [from, to] : all_edges)
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

  // all_edges and all_vertices gather edges and vertices from all contexts
  IntPairSet all_edges;
  ::std::unordered_map<int /* id */, per_vertex_info> all_vertices;

  ::std::string dot_filename;

  // Stats
  ::std::optional<float> critical_path; // Tinf
  ::std::optional<float> total_work; // T1
  size_t edge_count;
  size_t vertex_count;

private:
  mutable ::std::mutex mtx;

  // A vector that keeps track of all per context stored data
  ::std::vector<::std::shared_ptr<per_ctx_dot>> per_ctx;

protected:
  friend ::std::shared_ptr<dot_section>& dot_get_section_by_id(int id);
  // Map to get dot sections from their ID
  ::std::unordered_map<int, ::std::shared_ptr<dot_section>> section_map;
};

inline ::std::shared_ptr<dot_section>& dot_get_section_by_id(int id)
{
  // Note that it may populate the map
  return dot::instance().section_map[id];
}

inline void dot_section::push(::std::shared_ptr<per_ctx_dot>& pc, ::std::string symbol)
{
#if _CCCL_HAS_INCLUDE(<nvtx3/nvToolsExt.h>) && (!_CCCL_COMPILER(NVHPC) || _CCCL_STD_VER <= 2017)
  nvtxRangePushA(symbol.c_str());
#endif

  // We first create a section object, with its unique id
  auto sec = ::std::make_shared<dot_section>(mv(symbol));
  int id   = sec->get_id();

  // This must at least contain the section of the context
  auto& section_stack = pc->section_id_stack;
  sec->parent_id      = section_stack.back();

  // Save the section in the map
  dot_get_section_by_id(id) = sec;

  // Add the section to the children of its parent if that was not the root
  dot_get_section_by_id(sec->parent_id)->children_ids.push_back(id);

  // Push the id in the current stack
  section_stack.push_back(id);
}

inline void dot_section::pop(::std::shared_ptr<per_ctx_dot>& pc)
{
  pc->section_id_stack.pop_back();

#if _CCCL_HAS_INCLUDE(<nvtx3/nvToolsExt.h>) && (!_CCCL_COMPILER(NVHPC) || _CCCL_STD_VER <= 2017)
  nvtxRangePop();
#endif
}

} // namespace cuda::experimental::stf::reserved
