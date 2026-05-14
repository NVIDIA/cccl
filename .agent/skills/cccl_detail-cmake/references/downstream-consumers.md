# Downstream consumer surface

## Entry point

```cmake
find_package(CCCL CONFIG REQUIRED
  HINTS /path/to/cccl/lib/cmake/cccl/
)
target_link_libraries(my_target PRIVATE CCCL::CCCL)
```

`cccl-config.cmake` at `lib/cmake/cccl/` transitively locates and configures all enabled components via sibling `lib/cmake/<name>/` directories. No separate `find_package` calls for sub-libraries are needed when using the umbrella target.

## Exported targets

| Target              | Type                       | Provides                                                                                      |
|---------------------|----------------------------|-----------------------------------------------------------------------------------------------|
| `CCCL::CCCL`        | INTERFACE IMPORTED GLOBAL  | All components via single link                                                              |
| `CCCL::libcudacxx`  | ALIAS → `_libcudacxx_libcudacxx` | libcudacxx headers                                                                    |
| `CCCL::CUB`         | INTERFACE IMPORTED GLOBAL  | CUB (wraps `CUB::CUB`; IMPORTED not ALIAS to allow downstream export sets)                 |
| `CCCL::Thrust`      | Created via `thrust_create_target` | Thrust with configurable host/device systems                                    |
| `CCCL::cudax`       | ALIAS → `cudax::cudax`     | Experimental features (only with `CCCL_ENABLE_UNSTABLE`)                                    |

Sub-library targets also exported directly: `libcudacxx::libcudacxx`, `CUB::CUB`, `Thrust::Thrust`, `cudax::cudax`.

## Thrust host/device customization

When `CCCL::Thrust` is created, `thrust_create_target` reads two options:

| Option                     | Default | Controls       |
|----------------------------|---------|----------------|
| `CCCL_THRUST_HOST_SYSTEM`  | `CPP`   | Host backend   |
| `CCCL_THRUST_DEVICE_SYSTEM` | `CUDA`  | Device backend |

Set `CCCL_ENABLE_DEFAULT_THRUST_TARGET=OFF` to suppress `CCCL::Thrust` creation and call `thrust_create_target` manually.

## Component selection

```cmake
find_package(CCCL CONFIG REQUIRED COMPONENTS libcudacxx CUB)
```

Only the requested components are configured. Omitting `COMPONENTS` enables all three (Thrust, CUB, libcudacxx). `cudax` is never included automatically — it requires both an explicit `COMPONENTS cudax` and `CCCL_ENABLE_UNSTABLE=ON`.

## Internal vs. external find_package

Within the CCCL source tree, sub-libraries are never brought in via `find_package` at the CMake top level. `cccl_add_subdir_helper` directly `include()`s the package config files from `CCCL_SOURCE_DIR/lib/cmake/<name>/`, bypassing the find_package machinery to avoid re-configuration inconsistencies under CPM. See `cmake/CCCLAddSubdirHelper.cmake` comment referencing NVIDIA/libcudacxx#242 for details.

## Install layout

After `cmake --install`:

```
<prefix>/
  include/                 ← all sub-library headers
  lib/cmake/
    cccl/                  ← cccl-config.cmake, cccl-config-version.cmake
    libcudacxx/            ← libcudacxx-config.cmake, libcudacxx-header-search.cmake
    cub/
    thrust/
    cudax/
```

Each `*-header-search.cmake` is a configured file that records the relative path from the cmake package dir to the include dir, allowing the package to locate its headers without hardcoded paths.
