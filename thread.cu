// clang-format off

struct ThreadInWarpGroupsKernel
{
  template <class Config>
  __device__ void operator()(const Config& config) const noexcept
  {
    // 1. equivalent to thread_group grouped by 1
    {
      cuda::this_thread group{config};

      group.sync();
    }

    // 2. threads in warp grouped by 2
    {
      // Requires to store:
      // - [unsigned] mask of lanes in warp
      cuda::thread_group group{
        cuda::warp,        // in-level
        cuda::group_by<2>, // object determining the grouping
        config             // hierarchy-like object
      };

      group.sync();
    }

    // 3. threads in warp split by their rank / 4 groups
    {
      // Needs to store:
      // - [unsigned] mask of lanes in warp
      // - [unsigned] ngroups
      // - [unsigned] group rank (or the invocable)
      // - [unsigned] rank in group (or the invocalb, can be deduced from the warp mask?)
      cuda::thread_group group{
        cuda::warp,                         // in-level
        4,                                  // how many groups per block will be created (not needed, saves one collective operation)
        [](auto rank) { return rank / 4; }, // thread rank in group to group rank mapping, can be just passed as value?
        [](auto rank) { return rank % 4; }, // thread rank in group to thread rank in group mapping, can be just passed as value?
        config                              // hierarchy-like object
      };

      group.sync();
    }

    // 4. threads in warp split into 3 groups with each 4-th thread not being part of the group
    {
      // Needs to store:
      // - [unsigned] mask of lanes in warp (disabled thread store all 0)
      // - [unsigned] ngroups
      // - [unsigned] group rank (or the invocable)
      // - [unsigned] rank in group (or the invocalb, can be deduced from the warp mask?)
      cuda::thread_group group{
        cuda::warp,                                                // in-level
        3,                                                         // how many groups per block will be created, not including the disabled group (not needed, saves one collective operation)
        [](auto rank) { return (rank % 4 != 0)
          ? cuda::std::optional{rank % 4} : cuda::std::nullopt; }, // thread rank in group to group rank mapping (optional), can be just passed as value?
        [](auto rank) { return rank / 4; },                        // thread rank in group to thread rank in group mapping, can be just passed as value?
        config                                                     // hierarchy-like object
      };

      group.sync();
    }
  }
};

struct ThreadInBlockGroupsKernel
{
  template <class Config>
  __device__ void operator()(const Config& config) const noexcept
  {
    // 1. threads in block grouped by 2 (less than 32 can be optimized and don't need the barrier)
    {
      // Needs to store:
      // - [unsigned] mask of lanes in warp
      cuda::thread_group group{
        cuda::block,       // in-level
        cuda::group_by<2>, // object determining the grouping
        config             // hierarchy-like object
      };

      group.sync();
    }

    // 2. threads in block grouped by 64
    {
      constexpr auto nthread_in_group = 64;
      constexpr auto nbarriers        = cuda::thread.count(cuda::warp, config) / nthread_in_group;

      __shared__ cuda::barrier<cuda::thread_scope_thread> bars[nbarriers];
      // or allocate dynamic number of barriers in dynamic shared memory

      // Needs to store:
      // - [unsigned] mask of lanes in warp
      // - [barrier*] pointer to the group's barrier
      cuda::thread_group group{
        cuda::block,                      // in-level
        cuda::group_by<nthread_in_group>, // object determining the grouping
        bars,                             // span of barriers
        config                            // hierarchy-like object
      };

      group.sync();
    }

    // 3. threads in block split by their rank / 4 groups
    {
      constexpr auto ngroups = cuda::gpu_thread.count(cuda::block, config) / 4;

      __shared__ cuda::barrier<cuda::thread_scope_thread> bars[ngroups];
      // or allocate dynamic number of barriers in dynamic shared memory

      // Needs to store:
      // - [unsigned] mask of lanes in warp
      // - [unsigned] ngroups
      // - [unsigned] group rank (or the invocable)
      // - [unsigned] rank in group (or the invocable)
      // - [barrier*] pointer to the group's barrier
      cuda::thread_group group{
        cuda::block,                        // in-level
        ngroups,                            // how many groups per block will be created
        [](auto rank) { return rank / 4; }, // thread rank in group to group rank mapping, can be just passed as value?
        [](auto rank) { return rank % 4; }, // thread rank in group to thread rank in group mapping, can be just passed as value?
        bars,                               // span of barriers
        config                              // hierarchy-like object
      };

      group.sync();
    }

    // 4. threads in block split into 3 groups with each 4-th thread not being part of the group
    {
      constexpr auto ngroups = 3;

      __shared__ cuda::barrier<cuda::thread_scope_thread> bars[ngroups];
      // or allocate dynamic number of barriers in dynamic shared memory

      // Needs to store:
      // - [unsigned] mask of lanes in warp
      // - [unsigned] ngroups
      // - [unsigned] group rank (or the invocable)
      // - [unsigned] rank in group (or the invocable)
      // - [barrier*] pointer to the group's barrier
      cuda::thread_group group{
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
  }
};
