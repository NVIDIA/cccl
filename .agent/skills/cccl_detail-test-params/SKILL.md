---
description: |
  CCCL %PARAM% test parameterization — `cmake/CCCLTestParams.cmake`, comment syntax,
  cartesian-product axis expansion, generated CTest names, and `VAR_IDX` preprocessor
  injection. Currently used in CUB test trees; not yet adopted by Thrust or cudax.
  See `cccl-test` to run the generated test variants.
  Triggers: "%PARAM% test", "CCCLTestParams.cmake", "test variant expansion",
  "VAR_IDX", "cccl_parse_variant_params".
---

Parameterize a single `.cu`/`.cpp` test source into multiple CTest executables via
`%PARAM%` source comments. `cmake/CCCLTestParams.cmake` parses the comments at configure
time and expands them into the cartesian product of all axis values.

## Syntax

Each `%PARAM%` line follows this form:

```
// %PARAM% <DEFINITION> <label> <val0>:<val1>:...
```

- **DEFINITION** — preprocessor macro injected into each variant build. Convention: `TEST_` prefix.
- **label** — short token included in the generated executable name.
- **values** — colon-separated list. Only numeric values are tested in practice.

Example with two axes:

```cpp
// %PARAM% TEST_FOO foo 0:1:2
// %PARAM% TEST_LAUNCH lid 0:1
```

## Expansion

`cccl_parse_variant_params` builds the cartesian product. For the example above, six
variants are generated:

| Executable name      | Compile definitions                      |
|----------------------|------------------------------------------|
| `<base>.foo_0.lid_0` | `TEST_FOO=0 TEST_LAUNCH=0 VAR_IDX=0`     |
| `<base>.foo_0.lid_1` | `TEST_FOO=0 TEST_LAUNCH=1 VAR_IDX=1`     |
| `<base>.foo_1.lid_0` | `TEST_FOO=1 TEST_LAUNCH=0 VAR_IDX=2`     |
| `<base>.foo_1.lid_1` | `TEST_FOO=1 TEST_LAUNCH=1 VAR_IDX=3`     |
| `<base>.foo_2.lid_0` | `TEST_FOO=2 TEST_LAUNCH=0 VAR_IDX=4`     |
| `<base>.foo_2.lid_1` | `TEST_FOO=2 TEST_LAUNCH=1 VAR_IDX=5`     |

`VAR_IDX` is always injected alongside the declared definitions; it identifies the
variant's position in the cartesian product.

## Name convention

Variant suffix: `.label_value` per axis, dot-separated, appended to the base test name.
Base name is derived from the source filename by the consuming `CMakeLists.txt`.

In CUB, a `lid` axis (`lid 0:1:2`) also controls the launch mode:
- `lid_0` — host launch, RDC off
- `lid_1` — device launch (CDP), RDC on
- `lid_2` — graph capture, RDC off

## CMake API

Three public functions in `cmake/CCCLTestParams.cmake`:

| Function                                       | Purpose                                  |
|------------------------------------------------|------------------------------------------|
| `cccl_parse_variant_params(src num_var labels defs)` | Parse source; populate label and definition lists |
| `cccl_get_variant_data(labels defs idx label_var defs_var)` | Extract label + defs for one variant index |
| `cccl_log_variant_params(base num labels defs)` | Emit detected variant info at `VERBOSE` log level |

## Reconfiguration note

CMake does not track source file changes for reconfiguration. After modifying `%PARAM%`
comments, rerun CMake manually.

## Parameter axis guidance

Split only parameters that change template instantiations (typically input value types).
Splitting integral parameters such as `BLOCK_THREADS` compiles redundant code into
separate executables and inflates build time.

## Cross-reference

`cccl-test` — how to build and run the generated CTest variants.
