# Matrix expansion internals

## build-workflow.py pipeline

`ci/matrix.yaml` is the single source of truth. `build-workflow.py` reads it
and produces the dispatch JSON used by the parent workflow.

### Tag processing order

For each matrix entry:

1. **Validate** — required tags checked; unknown tags error.
2. **Set defaults** — missing optional tags filled from `matrix.yaml["tags"][tag]["default"]`.
3. **Explode** — any tag whose value is a list is exploded into N separate
   matrix jobs. `jobs` and `environment` are left unexploded (they are passed
   as lists to optimize scheduling).
4. **Canonicalize** — CTK aliases resolved (`"latest"` → `"13.X"`), host
   compiler version aliases resolved (`"gcc"` → `"gcc15"`).
5. **Set derived tags** — `std: "all"/"min"/"max"/"minmax"` resolved by
   intersecting CTK, host compiler, device compiler, and project std sets.
   `sm: "gpu"` replaced with the GPU's actual sm string.
6. **Re-explode** — derived tags that produce lists are exploded again.
7. **Apply exclusions** — `workflows.exclude` entries matched against each job;
   partial matches trim the list values rather than dropping the whole job.

### Two-stage grouping

Jobs with a `needs:` dependency in `matrix.yaml["jobs"]` become two-stage
(producer/consumer). A `build` job is the producer; `test*` jobs are consumers
that declare `needs: build`.

`finalize_workflow_dispatch_groups`:
- Merges consumers when multiple entries produce the same producer.
- Removes standalone duplicates of jobs that also appear as producers.
- Deduplicates standalone jobs.
- Assigns short GUIDs (base64 of incrementing 16-bit int) to every job and
  two-stage group for compact GHA naming.

### Output files (in `workflow/`)

| File                  | Contents                                                                |
|-----------------------|-------------------------------------------------------------------------|
| `workflow.json`       | Dispatch groups: `{group_name: {standalone: [...], two_stage: [...]}}`  |
| `dispatch.json`       | Shaped for GHA matrix: `{linux_two_stage: {keys, jobs}, ...}`           |
| `job_ids.json`        | `{id: "group_name job_name"}` — used by result aggregation              |
| `job_list.txt`        | Human-readable job list with IDs                                        |
| `runner_summary.json` | Runner counts table                                                     |
| `override.json`       | Present only when override workflow is active                           |
| `changes.md`          | inspect_changes summary (PR mode only)                                  |

### Override path

`--allow-override` is passed only in PR mode. If
`matrix_yaml["workflows"]["override"]` is non-empty, `build-workflow.py`
substitutes the override list for the requested workflow and writes
`workflow/override.json`. The `workflow-results` action checks for this file
and fails the workflow, blocking merging.

### Job command construction

`generate_dispatch_job_command` builds the shell command from the matrix job:

```
"./ci/build_cub.sh" -std "17" -arch "80-real"
"./ci/test_thrust.sh" -std "20" -arch "gpu"
```

Script path: `./ci/<job_prefix>_<project_id>.sh` (Linux) or
`./ci/windows/<job_prefix>_<project_id>.ps1`.

### force_producer_ctk

Some consumer job types declare `force_producer_ctk` in the jobs catalogue.
When present, the producer is built at a different CTK than the consumer runs
at. All consumers in a two-stage group must agree on the forced CTK.

### Lite workflow variant

When `--lite-build-projects` is populated and a `<workflow_name>_lite` workflow
exists in `matrix.yaml`, lite projects are pulled from that variant instead of
the full workflow. This reduces the matrix for projects whose transitive
dependencies changed but whose own files did not.
