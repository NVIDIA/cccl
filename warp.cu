// clang-format off

struct WarpInBlockGroupsKernel
{
  template <class Config>
  __device__ void operator()(const Config& config) const noexcept
  {
    // 1. equivalent to warp_group grouped by 1
    {
      cuda::this_warp group{config};

      group.sync();
    }

    // 2. warps in block grouped by 2
    {
      constexpr auto ngroups = cuda::warp.count(cuda::block, config) / 2;

      __shared__ cuda::barrier<cuda::thread_scope_thread> bars[ngroups];
      // or allocate dynamic number of barriers in dynamic shared memory

      // Needs to store:
      // - [barrier*] pointer to the group's barrier
      cuda::warp_group group{
        cuda::block,       // in-level
        cuda::group_by<2>, // object determining the grouping
        bars,              // span of barriers
        config             // hierarchy-like object
      };

      group.sync();
    }

    // 3. warps in block split by their rank / 4 groups
    {
      constexpr auto ngroups = cuda::warp.count(cuda::block, config) / 4;

      __shared__ cuda::barrier<cuda::thread_scope_thread> bars[ngroups];
      // or allocate dynamic number of barriers in dynamic shared memory

      // Needs to store:
      // - [unsigned] ngroups
      // - [unsigned] group rank (or the invocable)
      // - [unsigned] rank in group (or the invocable)
      // - [barrier*] pointer to the group's barrier
      cuda::warp_group group{
        cuda::block,                        // in-level
        ngroups,                            // how many groups per block will be created
        [](auto rank) { return rank / 4; }, // warp rank in group to group rank mapping, can be just passed as value?
        [](auto rank) { return rank % 4; }, // warp rank in group to warp rank in group mapping, can be just passed as value?
        bars,                               // span of barriers
        config                              // hierarchy-like object
      };

      group.sync();
    }

    // 4. warps in block split into 3 groups with each 4-th warp not being part of the group
    {
      constexpr auto ngroups = 3;

      __shared__ cuda::barrier<cuda::thread_scope_block> bars[ngroups];
      // or allocate dynamic number of barriers in dynamic shared memory

      // Needs to store:
      // - [unsigned] ngroups
      // - [unsigned] group rank (or the invocable)
      // - [unsigned] rank in group (or the invocable)
      // - [barrier*] pointer to the group's barrier
      cuda::warp_group group{
        cuda::block,                                               // in-level
        ngroups,                                                   // how many groups per block will be created, not including the disabled group
        [](auto rank) { return (rank % 4 != 0)
          ? cuda::std::optional{rank % 4} : cuda::std::nullopt; }, // warp rank in group to group rank mapping (optional), can be just passed as value?
        [](auto rank) { return rank / 4; },                        // warp rank in group to warp rank in group mapping, can be just passed as value?
        bars,                                                      // span of barriers
        config                                                     // hierarchy-like object
      };

      group.sync();
    }

    // 5. warps grouped as
    {
      enum : unsigned
      {
        reduce_squad,
        scan_store_squad,
        load_squad,
        sched_squad,
        loockback_squad,
        nsquads,
      };

      unsigned nwarps_per_squad[nsquads];
      nwarps_per_squad[reduce_squad]     = 4;
      nwarps_per_squad[scan_store_squad] = 4;
      nwarps_per_squad[load_squad]       = 1;
      nwarps_per_squad[sched_squad]      = 1;
      nwarps_per_squad[loockback_squad]  = 1;

      __shared__ cuda::barrier<cuda::thread_scope_block> bars[nsquads];
      // or allocate dynamic number of barriers in dynamic shared memory

      cuda::warp_group group{cuda::block, cuda::group_as{nwarps_per_squad}, bars, config};

      group.sync();

      switch (group.rank(cuda::block))
      {
      case reduce_squad:
      case scan_store_squad:
      case load_squad:
        return (cuda::gpu_thread.rank(group) == 0) ? 2 : 3;
      case sched_squad:
      case loockback_squad:
        return 1;
      default:
        return 0;
      }
    }
  }
};
