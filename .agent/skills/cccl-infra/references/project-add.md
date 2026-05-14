# Adding or removing a CCCL project

## Adding a project

### 1. Register in `ci/project_files_and_dependencies.yaml`

Add one or more project keys. Common pattern: a `_public` key for the public API files
and an `_internal` key for tests/infra that sets `matrix_project` (the name used in
`ci/matrix.yaml` workflow rows).

```yaml
myproject_public:
  name: "MyProject Public API"
  lite_dependencies: [libcudacxx_public]
  full_dependencies: []
  include_regexes: ["myproject/include/"]

myproject_internal:
  name: "MyProject Tests/Infra"
  matrix_project: "myproject"
  lite_dependencies: []
  full_dependencies: [myproject_public]
  include_regexes: ["myproject/"]
  exclude_project_files: [myproject_public]
```

Projects without `matrix_project` are internal-only; they affect dependency tracking but
do not appear in `FULL_BUILD` / `LITE_BUILD` outputs. Projects that downstream projects
depend on should have their public key added to those projects' `lite_dependencies` or
`full_dependencies`.

For the `core` project: any dirty files not matched by any project trigger a full rebuild.
Infra files (CMake, ci/, AGENTS.md, etc.) fall here by default.

### 2. Add build and test scripts

Create `ci/build_<matrix_project>.sh` and `ci/test_<matrix_project>.sh` following the
existing patterns (e.g., `ci/build_cub.sh`, `ci/test_cub.sh`). Windows variants go under
`ci/windows/build_<matrix_project>.ps1` / `ci/windows/test_<matrix_project>.ps1` if needed.

### 3. Add a CMake preset

Add a preset in `CMakePresets.json` for the new project. Inherit from an appropriate base
preset. See `cccl-cmake` for preset conventions.

### 4. Add workflow rows in `ci/matrix.yaml`

Add rows to `pull_request`, `nightly`, and `weekly` sections referencing the `matrix_project`
name. Mirror the structure of a similar existing project.

Example:
```yaml
- {jobs: ['build'], project: 'myproject', std: 'minmax', cxx: ['gcc', 'clang', 'msvc']}
- {jobs: ['test'],  project: 'myproject', std: 'max',    cxx: ['gcc', 'clang'], gpu: 'rtx2080'}
```

### 5. Add to `tidy` dependencies (optional)

If the project has C++ headers that should be checked by clang-tidy, add both public and
internal keys to `tidy.full_dependencies` in `ci/project_files_and_dependencies.yaml`.

### 6. Update `ignore_regexes` if needed

If any files in the new project directory should not trigger CI (e.g., pure scripts,
benchmarks), add matching regexes to `ignore_regexes` at the bottom of
`ci/project_files_and_dependencies.yaml`.

## Removing a project

1. Remove all workflow rows referencing the project from `ci/matrix.yaml`.
2. Remove the project's keys from `ci/project_files_and_dependencies.yaml`.
3. Remove the project from `tidy.full_dependencies` if present.
4. Delete `ci/build_<project>.sh`, `ci/test_<project>.sh`, and Windows variants.
5. Remove the CMake preset from `CMakePresets.json`.
6. Remove the project directory and any top-level CMakeLists references.
7. Update `CONTRIBUTING.md` and any docs that list the project.

## Files touched summary

| File                            | Change                              |
|---------------------------------|-------------------------------------|
| `ci/project_files_and_dependencies.yaml` | New project keys, dependency chains |
| `ci/matrix.yaml`                | Workflow rows for new project       |
| `CMakePresets.json`             | New preset                          |
| `ci/build_<project>.sh`         | New build script                    |
| `ci/test_<project>.sh`          | New test script                     |
| `ci/windows/build_<project>.ps1` | New Windows build script (if needed) |
| `ci/windows/test_<project>.ps1` | New Windows test script (if needed)  |
| `CONTRIBUTING.md`               | Updated project list                |
