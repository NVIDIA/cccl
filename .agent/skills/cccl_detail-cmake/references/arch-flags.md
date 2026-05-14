# Architecture flag translation

Source: `cmake/CCCLCheckCudaArchitectures.cmake`

## Recognized pseudo-values

| Input value      | Expansion                                                                                   |
|------------------|-----------------------------------------------------------------------------------------------|
| `all-cccl`       | All arches reported by `nvcc --help` matching `compute_NNN`, filtered to ≥ `minimum_cccl_arch` |
| `all-major-cccl` | Major-only subset of `all-cccl`: one entry per SM generation (`(arch / 10) * 10`), with the minimum clamped to `minimum_cccl_arch` |

All other values (`native`, `all`, `all-major`, numeric lists, `XX-real`, `XX-virtual`) pass through CMake's standard handling unchanged.

## Tag application

After filtering, both pseudo-values apply `-real` to every arch and `-real` + `-virtual` to the last arch in the list. This ensures PTX is emitted for the latest SM (forward-compat JIT) while all others compile to SASS only.

Example on CTK 12.9, `all-major-cccl` → `75-real;80-real;90-real;100-real;120-real;120-virtual`

## Internal call chain

```
cccl_check_cuda_architectures()         ← called from top-level CMakeLists.txt
  _cccl_detect_nvcc_arch_support()      ← runs nvcc --help, extracts compute_NNN
  _cccl_filter_to_supported_arches()    ← removes arches < minimum_cccl_arch
  _cccl_filter_to_all_major_cccl()      ← (all-major-cccl only) keeps one per SM gen
  _cccl_add_real_virtual_arch_tags()    ← tags -real / -virtual, pops last element for -virtual
```

Result is written back to `CMAKE_CUDA_ARCHITECTURES` cache with `FORCE`.

## minimum_cccl_arch history

| Value | Reason                          |
|-------|----------------------------------|
| 75    | CTK 13.x dropped pre-Turing support |

The variable lives in `CCCLCheckCudaArchitectures.cmake` and must be bumped whenever the minimum supported CTK drops an architecture tier.

## Pitfall: must call before `enable_language(CUDA)`

`cccl_check_cuda_architectures()` must be called before `enable_language(CUDA)` if you want the expanded value to influence the CUDA compiler setup. Once CMake has processed the language, `CMAKE_CUDA_ARCHITECTURES` changes may not propagate to all compiler invocations.
