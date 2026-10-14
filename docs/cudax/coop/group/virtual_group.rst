.. _cudax-coop-group-virtual-group:

``cuda::experimental::coop::virtual_group``
===========================================

.. code:: cuda

  namespace cuda::experimental::coop {

  template <typename Unit, typename ParentGroup, typename MappingResult>
  class virtual_group : public /*group-interface-for*/<virtual_group>
  {
  public:
    using unit_type       = Unit;
    using level_type      = typename ParentGroup::level_type;
    using hierarchy_type  = typename ParentGroup::hierarchy_type;

    template <typename Mapping>
      requires cuda::std::same_as</*mapping-result-of*/<Unit, ParentGroup, Mapping>, MappingResult>
    __device__ explicit virtual_group(const Unit& unit,
                                      const ParentGroup& parent_group,
                                      Mapping&& mapping) noexcept;

    virtual_group(const virtual_group&) = delete;

    virtual_group(virtual_group&&) = delete;

    virtual_group& operator=(const virtual_group&) = delete;

    virtual_group& operator=(virtual_group&&) = delete;
  };

  template </*hierarchy-level-type*/ Unit, group ParentGroup, typename Mapping>
    requires /*unit-same-as-or-below*/<Unit, typename ParentGroup::unit_type>
  virtual_group(const Unit&, const ParentGroup&, const Mapping&)
    -> virtual_group<Unit,
                      ParentGroup,
                      /*mapping-result-of*/<Unit, ParentGroup, Mapping>>;

  } // namespace cuda::experimental::coop

Overview
--------

``cuda::experimental::coop::virtual_group`` is a group type that provides an abstraction layer over a group. It allows you to take an existing group and split it into subgroups while still using the parent group's synchronization mechanism. This means that the created subgroups are not truly independent of each other and are required to share synchronization points.

Because of this limitation, a virtual group doesn't support optional participation in the group. If the input mapping outputs a unit without a valid mapping result, the behaviour is undefined.

The type can be constructed by specifying what is being grouped (``unit``) within what (``parent_group``) and how (``mapping``). The type is not copyable, movable, or assignable. The user should always use deduction guides to deduce the template parameters.

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

    // The parent group for the virtual_group.
    cudax::coop::this_block block{config};

    // An array for group_as mapping that specifies how many units every subgroup should get.
    const cuda::std::uint32_t warp_counts[]{/*group_A*/ 2,
                                            /*group_B*/ 3,
                                            /*group_C*/ 1};

    // Create a virtual_group of warps within a block.
    cudax::coop::virtual_group work_group{cuda::warp,
                                          block,
                                          cudax::coop::group_as{warp_counts}};

    // Dispatch work for every group based on its rank within the parent group.
    switch (work_group.rank(block))
    {
      // Note that every group can do different things, but the number of synchronization points must match.
      case group_A:
        print_info(work_group, "group_A");
        work_group.sync();
        break;
      case group_B:
        work_group.sync();
        print_info(work_group, "group_B");
        break;
      case group_C:
        print_info(work_group, "group_C");
        if (cuda::warp.rank(work_group) < 100) // always true
        {
          work_group.sync();
        }
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

`See it on Godbolt 🔗 <https://godbolt.org/z/ozfs4xKfj>`__
