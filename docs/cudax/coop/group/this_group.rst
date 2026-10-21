.. _cudax-coop-group-this-group:

``cuda::experimental::coop::this_{level}``
==========================================

.. code:: cuda

  namespace cuda::experimental::coop {

  template <typename Level, typename Hierarchy>
  class /*this-group*/ : public /*group-interface-for*/</*this-group*/>
  {
  public:
    using unit_type       = Level;
    using level_type      = Level;
    using hierarchy_type  = Hierarchy;

    /*this-group*/() = delete;

    template <typename HierarchyLike>
      requires cuda::std::same_as<Hierarchy, /*hierarchy-type-of*/<HierarchyLike>>
    __device__ /*this-group*/(const HierarchyLike& hierarchy_like) noexcept;

    /*this-group*/(const /*this-group*/&) = delete;

    /*this-group*/(/*this-group*/&&) = delete;

    /*this-group*/& operator=(const /*this-group*/&) = delete;

    /*this-group*/& operator=(/*this-group*/&&) = delete;
  };

  template <typename Hierarchy>
  class this_thread : public /*this-group*/<thread_level, Hierarchy> {};
  template <typename Hierarchy>
  class this_warp : public /*this-group*/<warp_level, Hierarchy> {};
  template <typename Hierarchy>
  class this_block : public /*this-group*/<block_level, Hierarchy> {};
  template <typename Hierarchy>
  class this_cluster : public /*this-group*/<cluster_level, Hierarchy> {};
  template <typename Hierarchy>
  class this_grid : public /*this-group*/<grid_level, Hierarchy> {};

  template </*hierarchy-level-type*/ Level, /*hierarchy-like-type*/ HierarchyLike>
  [[nodiscard]] __device__
  auto make_this_group(const Level& level, const HierarchyLike& hierarchy_like) noexcept;

  } // namespace cuda::experimental::coop

Overview
--------

*This* groups are the simplest groups that contain a single unit within the same level. They are the building blocks for all other groups. They are implemented by several classes named ``cuda::experimental::coop::this_{level}`` where ``level`` is any of the predefined CUDA levels. They can be explicitly constructed from a hierarchy-like object.

Because every this group is a separate type, the ``cuda::experimental::coop::make_this_group(...)`` factory function is provided to construct any this group based on the parameters.

*This* groups are always contiguous.

*This* groups use the :ref:`cuda::experimental::coop::level_synchronizer <cudax-coop-group-synchronizer-level-synchronizer>` synchronizer for synchronization.

Queries
-------

Sub-unit queries work as with any other group type; however, since *this* groups don't have a parent group, a native hierarchy level must be used in the super-level queries as the upper bound. For example, ``cuda::experimental::coop::this_warp{...}.count(cuda::block)`` queries the number of warps within a block.

Examples
--------

.. code:: cuda

  #include <cuda/devices>
  #include <cuda/stream>
  #include <cuda/hierarchy>
  #include <cuda/launch>
  #include <cuda/std/cstdint>
  #include <cuda/std/cassert>

  #include <cuda/experimental/coop/group>

  #include <cstdio>

  namespace cudax = cuda::experimental;

  __device__ void print_info(auto group, const char* group_name)
  {
    // Query thread and group information.
    const auto thread_count = cuda::gpu_thread.count_as<cuda::std::uint32_t>(group);
    const auto thread_rank  = cuda::gpu_thread.rank_as<cuda::std::uint32_t>(group);
    const auto group_count  = group.template count_as<cuda::std::uint32_t>(cuda::grid);
    const auto group_rank   = group.template rank_as<cuda::std::uint32_t>(cuda::grid);

    // When the launch dimensions are known at compile time, the static_count query returns a valid count;
    // otherwise, cuda::std::dynamic_extent is returned.
    constexpr auto static_thread_count = cuda::gpu_thread.static_count(group);
    constexpr auto static_group_count  = group.static_count(cuda::grid);

    if (thread_rank == 0 && group_rank == 0)
    {
      printf("%s contains %u threads%s. There are %u groups within the grid%s.\n",
            group_name,
            thread_count,
            (static_thread_count != cuda::std::dynamic_extent) ? " (statically known)" : "",
            group_count,
            (static_group_count != cuda::std::dynamic_extent) ? " (statically known)" : "");
    }

    // Ranks are always in the range `[0;count)`.
    assert(thread_rank < thread_count);
    assert(group_rank < group_count);

    // The .sync() method synchronizes the whole group. Synchronizing this_grid requires a cooperative launch.
    group.sync();
  }

  struct Kernel
  {
    __device__ void operator()(auto config) const
    {
      print_info(cudax::coop::this_thread{config}, "this_thread");
      print_info(cudax::coop::this_warp{config}, "this_warp");
      print_info(cudax::coop::this_block{config}, "this_block");
      print_info(cudax::coop::this_cluster{config}, "this_cluster");
      print_info(cudax::coop::this_grid{config}, "this_grid");
    }
  };

  int main()
  {
    // Select a device and create a stream for it.
    const auto device = cuda::devices[0];
    cuda::stream stream{device};

    // Launch the test kernel on the stream. Use the cooperative launch tag to make sure we can synchronize the whole grid.
    const auto config = cuda::make_config(cuda::grid_dims(2), cuda::block_dims<128>(), cuda::cooperative_launch{});
    cuda::launch(stream, config, Kernel{});

    // Wait until the kernel finishes.
    stream.sync();
  }

`See it on Godbolt 🔗 <https://godbolt.org/z/E1KPs8jTr>`__
