# CUB Tuning Policies

## Policy struct shapes

Each device algorithm defines its policy in `cub/cub/device/dispatch/tuning/tuning_<algo>.cuh`.
A typical policy aggregates sub-structs, one per kernel phase. Example for reduce:

```cpp
namespace detail::reduce {

struct agent_reduce_policy {
  int threads_per_block;
  int items_per_thread;
  int vec_size;
  BlockReduceAlgorithm block_algorithm;
  CacheLoadModifier load_modifier;
};

struct reduce_policy {
  agent_reduce_policy reduce;
  agent_reduce_policy single_tile;
};

} // namespace detail::reduce
```

The legacy style uses nested `struct`s with static constants (`ReducePolicy`,
`SingleTilePolicy`, etc.) accessed via `CUB_DEFINE_SUB_POLICY_GETTER`. New code uses
the aggregate style above. The dispatch layer handles both via `ReducePolicyWrapper`.

## Policy selector pattern

The selector is a `constexpr` function that maps runtime-detected
`compute_capability` + type-level traits to a `reduce_policy`:

```cpp
template <class AccumT, class OffsetT>
_CCCL_HOST_DEVICE constexpr reduce_policy get_policy(
  compute_capability cc, ...) noexcept;
```

Type-level classification helpers (`classify_accum_size<T>()`,
`classify_offset_size<T>()`, `op_type`) turn the template arguments into
discriminants so the selector stays `constexpr`.

## Authoring a custom policy hub

A custom policy hub is a struct with the same nested policy types the dispatch
layer queries. Pass it as the `PolicyHub` template argument to the dispatch type:

```cpp
struct MyReduceHub {
  struct MaxPolicy {
    struct ReducePolicy : cub::AgentReducePolicy<256, 16, int, 4,
      cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_LDG> {};
    struct SingleTilePolicy : ReducePolicy {};
  };
};

// Invoke with a custom hub:
cub::DeviceReduce::DispatchReduce<...>::Dispatch<MyReduceHub>(...);
```

Tests named `catch2_test_device_<algo>_custom_policy_hub.cu` demonstrate the
full pattern for each algorithm. Read those before writing a new hub.

## Where policies live per algorithm

| Algorithm            | Tuning file                                                                              |
|----------------------|------------------------------------------------------------------------------------------|
| Reduce               | `tuning/tuning_reduce.cuh` + `tuning_reduce_deterministic.cuh` / `_nondeterministic.cuh` |
| Scan                 | `tuning/tuning_scan.cuh`, `tuning_scan_by_key.cuh`                                       |
| Sort (radix)         | `tuning/tuning_radix_sort.cuh`                                                           |
| Sort (merge)         | `tuning/tuning_merge_sort.cuh`                                                           |
| Histogram            | `tuning/tuning_histogram.cuh`                                                            |
| Select / Partition   | `tuning/tuning_select_if.cuh`, `tuning_three_way_partition.cuh`                          |
| Run-length encode    | `tuning/tuning_rle_encode.cuh`, `tuning_rle_non_trivial_runs.cuh`                        |
| Others               | `tuning/tuning_<algo>.cuh` — pattern is uniform                                          |

## Compute-capability dispatch

Selectors branch on `compute_capability` values (e.g., `sm_80`, `sm_90`). The
`cuda/__device/compute_capability.h` header provides the comparison operators.
Selectors are evaluated at device-function instantiation time, so the compiler
sees the final constant folded policy — no runtime branching.
