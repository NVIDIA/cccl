# elect.sync

- PTX ISA: [`elect.sync`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-elect-sync)

*Implementation note:* Since C++ does not support returning multiple values, the
variant of the instruction that returns both a predicate and an updated
membermask is not supported.

| C++ | PTX |
| [(0)](#0-elect_sync) `cuda::ptx::elect_sync`| `elect.sync` |


### [(0)](#0-elect_sync) `elect_sync`
{: .no_toc }
```cuda
// elect.sync _|is_elected, membermask; // PTX ISA 80, SM_90
template <typename=void>
__device__ static inline bool elect_sync(
  const uint32_t& membermask);
```
