# Tool index — cccl-libcudacxx

## Used (canonical reference lives in another skill)

| Tool | Purpose | Reference |
|------|---------|-----------|
| `ci/build_libcudacxx.sh` | Full-matrix libcudacxx build: host/std/arch sweep. | `cccl-build` → `references/tools.md` |
| `ci/test_libcudacxx.sh` | Full-matrix libcudacxx test via lit + ctest; requires GPU. | `cccl-test` → `references/tools.md` |
| `ci/util/build_and_test_targets.sh` | Targeted build+test; `--lit-precompile-tests` / `--lit-tests` flags drive libcudacxx lit runs. | `cccl-build` → `references/build_and_test_targets_usage.md` |
