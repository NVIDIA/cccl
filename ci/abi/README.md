# ABI diff

CCCL is header-only, so every library that uses it carries its own copies of CCCL's inline
functions, function-local statics, vtables and class layouts as weak, COMDAT or
`STB_GNU_UNIQUE` symbols. When two static libraries built against different CCCL releases are
linked into one binary, the linker keeps one definition per mangled name, even with hidden
visibility and `-Wl,--exclude-libs`. If a symbol keeps its mangled name but changes its
definition, one release's code runs on the other release's data.

`abi_diff.sh` compares `HEAD` against the newest `vX.Y.Z` tag and fails when such a collision
is found. The fix is to move the changed entities into a new inline namespace, for example
`_CCCL_BEGIN_NAMESPACE_ABI_VER4_BUMP`.

## Usage

```bash
ci/abi/abi_diff.sh                       # HEAD against the newest vX.Y.Z tag
ci/abi/abi_diff.sh -base-ref v3.5.0      # HEAD against a given release
ci/abi/cccl_abi_diff.py v3.5.0 HEAD      # the underlying tool, without exclusions
```

The tool needs `g++`, `readelf`, `c++filt`, `git` and the CUDA Toolkit headers. It uses
`nvcc` for headers that only compile as CUDA. The CI job runs in the default devcontainer.

## How it works

1. `git archive` each revision's `libcudacxx/include`, `cub`, `thrust` and `cudax/include`.
2. Compile every public header with `-O0 -g -fkeep-inline-functions`, so that every inline
   function and its statics are emitted with DWARF. Headers that compile with neither `g++`
   nor `nvcc` are skipped.
3. For each emitted function and object, read the return and parameter types, the object's
   type, and the layout of every class reachable from them.
4. For each mangled name present in both revisions, report:
   - **A** function: a return or parameter type changed size, kind or calling convention, for
     example a scalar return that became a struct returned through a hidden pointer.
   - **B** object: `st_size` or type changed (function-local statics, vtables, variable
     templates).
   - **C** class: `sizeof`, alignment, calling convention, member offsets or member types, or
     non-empty bases changed for a class reachable from a common symbol.

Any difference fails the check. Adding or removing an empty non-virtual base is not a
difference: it changes neither the layout nor any mangled name.

## Limitations

- Only signatures and layouts are compared. A function whose body changes in an incompatible
  way while its signature stays the same is not detected.
- Only host code is compared, and only templates that the public headers instantiate.
- `cuda::experimental` is excluded because it has no ABI guarantee. CUB and Thrust are
  excluded because their symbols live in an inline namespace that encodes their version.
