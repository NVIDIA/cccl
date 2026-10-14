.. _cudax-coop-group-generic-group:

``cuda::experimental::coop::generic_group``
===========================================

.. code:: cuda

  namespace cuda::experimental::coop {

  template <typename Unit,
            typename ParentGroup,
            typename MappingResult,
            typename SynchronizerInstance>
  class generic_group : public /*group-interface-for*/<generic_group>
  {
  public:
    using unit_type       = Unit;
    using level_type      = typename ParentGroup::level_type;
    using hierarchy_type  = typename ParentGroup::hierarchy_type;

    template <typename Mapping, typename Synchronizer>
      requires cuda::std::same_as</*mapping-result-of*/<Unit, ParentGroup, Mapping>, MappingResult>
        && cuda::std::same_as</*synchronizer-instance-for*/<Unit, ParentGroup, MappingResult, Synchronizer>,
                              SynchronizerInstance>
    __device__ explicit generic_group(const Unit& unit,
                                     const ParentGroup& parent_group,
                                     Mapping&& mapping,
                                     Synchronizer&& synchronizer) noexcept;

    generic_group(const generic_group&) = delete;

    generic_group(generic_group&&) = delete;

    generic_group& operator=(const generic_group&) = delete;

    generic_group& operator=(generic_group&&) = delete;
  };

  template </*hierarchy-level-type*/ Unit,
            group ParentGroup,
            typename Mapping,
            typename Synchronizer,
            typename MappingResult = /*mapping-result-of*/<Unit, ParentGroup, Mapping>>
    requires /*unit-same-as-or-below*/<Unit, typename ParentGroup::unit_type>
  generic_group(const Unit&, const ParentGroup&, Mapping&&, Synchronizer&&)
    -> generic_group<Unit,
                      ParentGroup,
                      MappingResult,
                      /*synchronizer-instance-for*/<Unit, ParentGroup, MappingResult, Synchronizer>>;

  } // namespace cuda::experimental::coop

Overview
--------

``cuda::experimental::coop::generic_group`` is a group type that splits the parent group into subgroups according to the ``mapping``. Every subgroup is instantiated with a synchronization mechanism provided by the ``synchronizer``, which makes the subgroups completely independent of each other. The group's constructor is the only statement that all of the units from the parent group are required to execute uniformly.

Queries
-------

Sub-unit queries work as with any other group type. The group may be queried for rank and (static) count within the parent group.

Examples
--------

.. code:: cuda

  #include <cuda/devices>
  #include <cuda/stream>
  #include <cuda/hierarchy>
  #include <cuda/launch>
  #include <cuda/std/cstdint>
  #include <cuda/std/ranges>

  #include <cuda/experimental/coop/group>

  #include <cstdio>

  namespace cudax = cuda::experimental;

  __device__ void print_info(const auto& work_group, const char* group_name)
  {
    // Print a hello world message from the root thread in the group.
    if (cuda::gpu_thread.is_root_rank(work_group))
    {
      // Get the warp count in the group.
      const auto warp_count = cuda::warp.count(work_group);

      // Print the message.
      printf("Hello world from group %s of %u %s!\n", group_name, warp_count, (warp_count > 1) ? "warps" : "warp");
    }
  }

  __device__ void demo(auto config)
  {
    // An enumeration helper for every group rank.
    enum : cuda::std::uint32_t
    {
      group_A,
      group_B,
      group_C,
    };

    // The parent group for the generic_group.
    cudax::coop::this_block block{config};

    // An array for group_as mapping that specifies how many units every subgroup should get.
    const cuda::std::uint32_t warp_counts[]{/*group_A*/ 2,
                                            /*group_B*/ 3,
                                            /*group_C*/ 1};

    // Create a generic_group of warps within a block that uses interwarp_synchronizer as the synchronization
    // mechanism for every subgroup.
    cudax::coop::generic_group work_group{cuda::warp,
                                          block,
                                          cudax::coop::group_as{warp_counts},
                                          cudax::coop::interwarp_synchronizer{cuda::std::ranges::views::iota(1)}};

    // Dispatch work for every group based on its rank within the parent group.
    switch (work_group.rank(block))
    {
      // Note that all groups are independent of each other and thus can synchronize in different places using
      // different synchronization kinds.
      case group_A:
        print_info(work_group, "group_A");
        break;
      case group_B:
        work_group.sync();
        print_info(work_group, "group_B");
        work_group.sync();
        break;
      case group_C:
        print_info(work_group, "group_C");
        work_group.sync();
        work_group.sync_aligned();
        break;
    }
  }

  struct Kernel
  {
    __device__ void operator()(auto config) const
    {
      demo(config);
    }
  };

  int main()
  {
    // Select a device and create a stream for it.
    const auto device = cuda::devices[0];
    cuda::stream stream{device};

    // Launch the test kernel on the stream.
    const auto config = cuda::make_config(cuda::grid_dims(1), cuda::block_dims<192>());
    cuda::launch(stream, config, Kernel{});

    // Wait until the kernel finishes.
    stream.sync();
  }

`See it on Godbolt 🔗 <https://godbolt.org/z/sahvrPc5b>`__
