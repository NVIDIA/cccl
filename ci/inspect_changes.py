#!/usr/bin/env python3
"""Identify dirty CCCL subprojects between two commits or explicit path lists."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "ci" / "project_files_and_dependencies.yaml"
CORE_PROJECT_KEY = "core"


class SummaryWriter:
    """Utility for duplicating output to stdout and an optional summary file."""

    def __init__(self, path: Optional[Path]):
        self._handle = path.open("a", encoding="utf-8") if path else None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle:
            self._handle.close()

    def log(self, line: str = "") -> None:
        print(line)
        if self._handle:
            self._handle.write(line + "\n")


@dataclass(frozen=True)
class ProjectConfig:
    """Per-project configuration derived from the YAML file."""

    key: str
    name: str
    matrix_project: Optional[str]
    include_regexes: Tuple[str, ...]
    exclude_regexes: Tuple[str, ...]
    exclude_project_files: Tuple[str, ...]
    lite_dependencies: Tuple[str, ...]
    full_dependencies: Tuple[str, ...]
    transitive_lite_dependencies: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Config:
    """Aggregated configuration for change detection."""

    projects: Dict[str, ProjectConfig]
    project_keys: Tuple[str, ...]
    ignore_regexes: Tuple[str, ...]

    def project(self, key: str) -> ProjectConfig:
        return self.projects[key]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for determining dirty files."""
    parser = argparse.ArgumentParser(
        description="Identify which CCCL projects require rebuilds between two commits."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--refs",
        nargs=2,
        metavar=("BASE", "HEAD"),
        help="Compare two refs using 'git diff --name-only' to determine dirty files",
    )
    group.add_argument(
        "--file",
        metavar="PATH",
        help="Read dirty file paths (one per line) from PATH",
    )
    group.add_argument(
        "--stdin",
        action="store_true",
        help="Read dirty file paths (one per line) from stdin",
    )
    parser.add_argument(
        "--summary",
        metavar="PATH",
        default=None,
        help="Optional path to write a markdown summary table",
    )
    return parser.parse_args(argv)


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    projects_raw = raw.get("projects")
    if not isinstance(projects_raw, dict) or not projects_raw:
        raise SystemExit(f"No projects defined in {path}")

    def to_tuple(value: Optional[Sequence[str] | str]) -> Tuple[str, ...]:
        if value is None:
            return tuple()
        if isinstance(value, (list, tuple)):
            return tuple(value)
        return (value,)

    project_keys: List[str] = list(projects_raw.keys())
    projects: Dict[str, ProjectConfig] = {}
    for key in project_keys:
        entry = projects_raw.get(key) or {}
        name = entry.get("name", key)
        matrix_project = entry.get("matrix_project")
        include_regexes = to_tuple(entry.get("include_regexes", []))
        exclude_regexes = to_tuple(entry.get("exclude_regexes", []))
        exclude_project_files = to_tuple(entry.get("exclude_project_files"))
        lite_dependencies = to_tuple(entry.get("lite_dependencies"))
        full_dependencies = to_tuple(entry.get("full_dependencies"))

        if key != CORE_PROJECT_KEY and not include_regexes:
            raise SystemExit(
                f"Project '{key}' must define at least one include_regex in {path}"
            )

        projects[key] = ProjectConfig(
            key=key,
            name=name,
            matrix_project=matrix_project,
            include_regexes=include_regexes,
            exclude_regexes=exclude_regexes,
            exclude_project_files=exclude_project_files,
            lite_dependencies=lite_dependencies,
            full_dependencies=full_dependencies,
            transitive_lite_dependencies=tuple(),
        )

    if CORE_PROJECT_KEY not in projects:
        raise SystemExit(f"Project configuration must define '{CORE_PROJECT_KEY}'")

    dependency_graph = build_dependency_graph_raw(projects)
    transitive_map = compute_transitive_dependencies(
        project_keys, projects, dependency_graph
    )

    for key, transitive in transitive_map.items():
        projects[key] = replace(projects[key], transitive_lite_dependencies=transitive)

    ignore_regexes = tuple(raw.get("ignore_regexes", ()))

    return Config(
        projects=projects,
        project_keys=tuple(project_keys),
        ignore_regexes=ignore_regexes,
    )


def run_git(args: Sequence[str], *, capture_output: bool = True) -> str:
    """Run a git command in the repo and optionally capture stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
    )
    if capture_output:
        return result.stdout.strip()
    return ""


def try_rev_parse(ref: str) -> str:
    try:
        return run_git(["rev-parse", ref])
    except subprocess.CalledProcessError:
        return ref


def ensure_fetched(ref: str) -> None:
    """Fetch a ref from origin ensuring availability for merge-base."""
    run_git(["fetch", "origin", ref, "-q"], capture_output=False)


def repo_is_shallow() -> bool:
    """Return True when the repository has a shallow history."""
    output = run_git(["rev-parse", "--is-shallow-repository"])
    return output.strip().lower() == "true"


def indent(text: str, prefix: str) -> str:
    """Indent multi-line text for readable logging blocks."""
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())


def anchor_regex(pattern: str) -> re.Pattern[str]:
    """Anchor a path regex to the repository root."""
    anchored = pattern if pattern.startswith("^") else f"^{pattern}"
    return re.compile(anchored)


def compile_patterns(patterns: Sequence[str]) -> Tuple[re.Pattern[str], ...]:
    """Compile a list of regex strings into anchored patterns."""
    return tuple(anchor_regex(pattern) for pattern in patterns)


def matches_any(patterns: Sequence[re.Pattern[str]], path: str) -> bool:
    """Return True when any compiled regex matches the given path."""
    return any(pattern.search(path) for pattern in patterns)


def build_dependency_graph_raw(
    projects: Dict[str, ProjectConfig],
) -> Dict[str, List[Tuple[str, str]]]:
    """Return mapping of project -> [(dependency, type)]."""
    graph: Dict[str, List[Tuple[str, str]]] = {}
    for key, project in projects.items():
        edges: List[Tuple[str, str]] = []
        edges.extend((dep, "full") for dep in project.full_dependencies)
        edges.extend((dep, "lite") for dep in project.lite_dependencies)
        graph[key] = edges
    return graph


def compute_transitive_dependencies(
    project_keys: Sequence[str],
    projects: Dict[str, ProjectConfig],
    graph: Dict[str, List[Tuple[str, str]]],
) -> Dict[str, Tuple[str, ...]]:
    """Return a dictionary of transitive dependencies for each project."""
    result: Dict[str, Tuple[str, ...]] = {}

    for key in project_keys:
        visited: set[str] = set()
        queue: deque[str] = deque(dep for dep, _ in graph.get(key, []))

        while queue:
            dep = queue.popleft()
            if dep == key or dep in visited:
                continue
            visited.add(dep)
            queue.extend(child for child, _ in graph.get(dep, []))

        project_cfg = projects[key]
        direct_full = set(project_cfg.full_dependencies)
        direct_lite = set(project_cfg.lite_dependencies)
        transitive = [
            dep
            for dep in project_keys
            if dep in visited and dep not in direct_full and dep not in direct_lite
        ]

        result[key] = tuple(transitive)

    return result


def project_dirty_files(
    project: ProjectConfig, dirty_files: Sequence[str]
) -> List[str]:
    """Collect dirty files that belong to the given project. Does not apply project exclusions yet."""
    include_patterns = compile_patterns(project.include_regexes)
    exclude_patterns = compile_patterns(project.exclude_regexes)

    if not include_patterns:
        return []

    included = [path for path in dirty_files if matches_any(include_patterns, path)]

    if exclude_patterns:
        included = [
            path for path in included if not matches_any(exclude_patterns, path)
        ]

    return included


def build_project_dirty_map(
    config: Config, dirty_files: Sequence[str]
) -> Dict[str, List[str]]:
    """Compute per-project dirty file lists, including a residual `core` list."""
    project_files: Dict[str, List[str]] = {}

    # First gather matches for every non-core project. Files may belong to multiple projects.
    for key in config.project_keys:
        if key == CORE_PROJECT_KEY:
            continue
        files = project_dirty_files(config.project(key), dirty_files)
        project_files[key] = files

    # Remove files that are owned by other projects when requested.
    for key in config.project_keys:
        if key == CORE_PROJECT_KEY:
            continue
        project = config.project(key)
        if not project.exclude_project_files:
            continue
        excluded_sets = [
            set(project_files.get(other, [])) for other in project.exclude_project_files
        ]
        if not excluded_sets:
            continue
        exclusions = set().union(*excluded_sets)
        project_files[key] = [
            path for path in project_files[key] if path not in exclusions
        ]

    matched_paths: set[str] = set()
    for key in config.project_keys:
        if key == CORE_PROJECT_KEY:
            continue
        matched_paths.update(project_files.get(key, []))

    core_files = [path for path in dirty_files if path not in matched_paths]
    project_files[CORE_PROJECT_KEY] = core_files
    return project_files


def write_output(key: str, value: str) -> None:
    """Emit a GitHub Actions output key/value pair."""
    line = f"{key}={value}"
    print(line)
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")


def collect_dirty_files(
    args: argparse.Namespace, config: Config
) -> Tuple[List[str], List[str]]:
    """Return normalized dirty file paths and the subset filtered as ignored."""
    if args.refs:
        if repo_is_shallow():
            run_git(["fetch", "origin", "--unshallow", "-q"], capture_output=False)

        base_ref = try_rev_parse(args.refs[0])
        head_ref = try_rev_parse(args.refs[1])

        ensure_fetched(base_ref)
        ensure_fetched(head_ref)

        base_sha = run_git(["merge-base", base_ref, head_ref])
        head_sha = run_git(["rev-parse", head_ref])

        print(f"Base SHA: {base_sha}")
        base_log = run_git(["log", "--oneline", "-1", base_sha])
        if base_log:
            print(indent(base_log, "  "))
        print(f"HEAD SHA: {head_sha}")
        head_log = run_git(["log", "--oneline", "-1", head_sha])
        if head_log:
            print(indent(head_log, "  "))

        dirty_files = run_git(["diff", "--name-only", base_sha, head_sha]).splitlines()
    elif args.file:
        print(f"Dirty files provided via file: {args.file}")
        with open(args.file, "r", encoding="utf-8") as handle:
            dirty_files = [line.strip() for line in handle if line.strip()]
    else:  # args.stdin
        print("Dirty files provided via stdin")
        dirty_files = [line.strip() for line in sys.stdin if line.strip()]

    print()

    dirty_files = [path for path in dirty_files if path]

    ignore_patterns = compile_patterns(config.ignore_regexes)
    if ignore_patterns:
        kept: List[str] = []
        ignored: List[str] = []
        for path in dirty_files:
            if matches_any(ignore_patterns, path):
                ignored.append(path)
            else:
                kept.append(path)
        return kept, ignored

    return dirty_files, []


def format_bullet_list(lines: Sequence[str], indent: str = "  ") -> List[str]:
    """Convert each string into a markdown bullet line with the given indent."""
    return [f"{indent}- {line}" for line in lines]


def build_reverse_dependency_graph(config: Config) -> Dict[str, List[Tuple[str, bool]]]:
    """Return mapping of dependency -> (dependent, requires_full_rebuild)."""
    reverse: Dict[str, List[Tuple[str, bool]]] = {}

    for project in config.projects.values():
        for dep in project.full_dependencies:
            reverse.setdefault(dep, []).append((project.key, True))
        for dep in project.lite_dependencies:
            reverse.setdefault(dep, []).append((project.key, False))
        for dep in project.transitive_lite_dependencies:
            reverse.setdefault(dep, []).append((project.key, False))

    return reverse


def propagate_dirty_projects(
    config: Config,
    initial_full: Sequence[str],  # Projects with dirty files
    reverse_graph: Dict[
        str, List[Tuple[str, bool]]
    ],  # dependency -> [(dependent, requires_full_rebuild)]
) -> Tuple[set[str], set[str]]:  # (full_set, lite_set)
    """Propagate rebuild requirements through the reverse dependency graph."""
    full_set: set[str] = set(initial_full)
    lite_set: set[str] = set()
    # BFS queue of (project_key, depth)
    queue: deque[Tuple[str, int]] = deque((key, 0) for key in initial_full)
    seen: set[Tuple[str, int]] = set(queue)

    while queue:
        current, depth = queue.popleft()
        for dependent, edge_full in reverse_graph.get(current, []):
            propagate_full = edge_full and depth == 0

            if propagate_full:
                if dependent in full_set:
                    continue
                if dependent in lite_set:
                    lite_set.remove(dependent)
                full_set.add(dependent)
                print(
                    "- Upstream dependency change detected (full rebuild): "
                    f"'{dependent}' ({config.project(dependent).name}) depends on dirty project '{current}'"
                )
            else:
                if dependent in full_set or dependent in lite_set:
                    continue
                lite_set.add(dependent)
                print(
                    "- Upstream dependency change detected (lite rebuild): "
                    f"'{dependent}' ({config.project(dependent).name}) depends on dirty project '{current}'"
                )

            key = (dependent, depth + 1)
            if key not in seen:
                seen.add(key)
                queue.append(key)

    return full_set, lite_set


def log_dependency_overview(config: Config) -> None:
    """Pretty-print the dependency data for each project."""
    print("Project Dependency Overview:")
    for key in config.project_keys:
        project = config.project(key)
        header = f"- {project.name} (key={key}"
        if project.matrix_project:
            header += f" matrix_project={project.matrix_project}"
        header += ")"
        print(header)

        if project.full_dependencies:
            print("  - direct full deps:")
            for line in format_bullet_list(project.full_dependencies, indent="      "):
                print(line)
        if project.lite_dependencies:
            print("  - direct lite deps:")
            for line in format_bullet_list(project.lite_dependencies, indent="      "):
                print(line)
        if project.transitive_lite_dependencies:
            print("  - transitive deps:")
            for line in format_bullet_list(
                project.transitive_lite_dependencies, indent="      "
            ):
                print(line)
    print()


def build_dirty_sections(
    config: Config,
    project_dirty_map: Dict[str, List[str]],
    ignored_files: Sequence[str],
) -> List[Tuple[str, List[str]]]:
    """Create (heading, files) tuples for non-empty dirty buckets."""
    sections: List[Tuple[str, List[str]]] = []
    for key in config.project_keys:
        if key == CORE_PROJECT_KEY:
            continue
        files = project_dirty_map.get(key, [])
        if files:
            sections.append((f"{config.project(key).name} ({key})", list(files)))

    core_files = project_dirty_map.get(CORE_PROJECT_KEY, [])
    if core_files:
        sections.append(
            (
                f"{config.project(CORE_PROJECT_KEY).name} ({CORE_PROJECT_KEY})",
                list(core_files),
            )
        )

    if ignored_files:
        sections.append(("Ignored Files", list(ignored_files)))

    return sections


def log_dirty_files(
    combined_dirty: Sequence[str],
    sections: Sequence[Tuple[str, Sequence[str]]],
) -> None:
    """Emit dirty-file information in markdown list form."""
    if sections:
        print("Dirty files by project:")
        for idx, (heading, files) in enumerate(sections):
            print(f"{heading}:")
            for line in format_bullet_list(files):
                print(line)
            if idx != len(sections) - 1:
                print()
        print()

    print("All dirty files:")
    if combined_dirty:
        for line in format_bullet_list(combined_dirty):
            print(line)
    else:
        print("  - (none)")
    print()


def determine_rebuild_sets(
    config: Config,
    project_dirty_map: Dict[str, List[str]],  # key -> dirty files
    reverse_graph: Dict[
        str, List[Tuple[str, bool]]
    ],  # dependency -> [(dependent, requires_full_rebuild)]
) -> Tuple[
    Dict[str, str], set[str], set[str]
]:  # (project_statuses, full_set, lite_set)
    """Return project statuses plus the full/lite rebuild sets."""
    core_dirty_files = project_dirty_map[CORE_PROJECT_KEY]

    if core_dirty_files:
        project_statuses: Dict[str, str] = {key: "Dirty" for key in config.project_keys}
        return project_statuses, set(config.project_keys), set()

    initially_dirty = [
        key
        for key in config.project_keys
        if key != CORE_PROJECT_KEY and project_dirty_map.get(key)
    ]
    for key in initially_dirty:
        print(f"- Changes detected in subproject '{key}' ({config.project(key).name})")

    full_set, lite_set = propagate_dirty_projects(
        config,
        initially_dirty,
        reverse_graph,
    )

    project_statuses: Dict[str, str] = {}
    for key in config.project_keys:
        if key in full_set:
            project_statuses[key] = "Dirty"
        elif key in lite_set:
            project_statuses[key] = "Dirty Deps"
        else:
            project_statuses[key] = "Clean"

    return project_statuses, full_set, lite_set


def compute_outputs(
    config: Config,
    full_set: set[str],
    lite_set: set[str],
) -> Tuple[List[str], List[str]]:  # (FULL_BUILD, LITE_BUILD) matrix_project lists
    """Convert rebuild sets into ordered matrix project lists."""
    full_output = [
        config.project(key).matrix_project
        for key in config.project_keys
        if key in full_set and config.project(key).matrix_project
    ]
    lite_output = [
        config.project(key).matrix_project
        for key in config.project_keys
        if key in lite_set and config.project(key).matrix_project
    ]
    return full_output, lite_output


def emit_outputs(full_output: Sequence[str], lite_output: Sequence[str]) -> None:
    """Write FULL_BUILD/LITE_BUILD strings to stdout and GitHub outputs."""
    print("Github Action Outputs:")
    write_output("FULL_BUILD", " ".join(full_output))
    write_output("LITE_BUILD", " ".join(lite_output))
    print()


def write_project_summary(
    config: Config,
    project_statuses: Dict[str, str],
    writer: SummaryWriter,
) -> None:
    """Render the status table inside the summary section."""
    name_width = max(len(config.project(key).name) for key in config.project_keys)
    writer.log(f"| {'Project':<{name_width}} | Status     |")
    writer.log(f"|{'-' * (name_width + 2)}|------------|")
    for key in config.project_keys:
        project = config.project(key)
        status = project_statuses.get(key, "Unknown?")
        writer.log(f"| {project.name:<{name_width}} | {status:<10} |")
    writer.log()


def write_summary_dirty_sections(
    combined_dirty: Sequence[str],
    sections: Sequence[Tuple[str, Sequence[str]]],
    writer: SummaryWriter,
) -> None:
    """Render dirty-file details inside the summary section."""
    writer.log("<details><summary><h4>👉 Dirty Files</h4></summary>")
    if sections:
        for idx, (heading, files) in enumerate(sections):
            writer.log()
            writer.log(f"{heading}:")
            for line in format_bullet_list(files):
                writer.log(line)

    writer.log()
    if combined_dirty:
        writer.log("All dirty files:")
        for line in format_bullet_list(combined_dirty):
            writer.log(line)

    writer.log()
    writer.log("</details>")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entrypoint used by the GitHub Action wrapper."""
    args = parse_args(argv)
    config = load_config(CONFIG_PATH)

    summary_path = Path(args.summary) if args.summary else None

    dirty_files, ignored_files = collect_dirty_files(args, config)

    project_dirty_map = build_project_dirty_map(config, dirty_files)
    reverse_graph = build_reverse_dependency_graph(config)

    print(f"Subprojects: {' '.join(config.project_keys)}")
    print()

    log_dependency_overview(config)

    combined_dirty = dirty_files + ignored_files
    sections = build_dirty_sections(config, project_dirty_map, ignored_files)
    log_dirty_files(combined_dirty, sections)

    print("Checking for changes...")

    project_statuses, full_set, lite_set = determine_rebuild_sets(
        config,
        project_dirty_map,
        reverse_graph,
    )

    full_output, lite_output = compute_outputs(config, full_set, lite_set)

    print()
    emit_outputs(full_output, lite_output)

    with SummaryWriter(summary_path) as summary_writer:
        print("::group::Project Change Summary")
        summary_writer.log(
            "<details><summary><h3>👃 Inspect Project Changes</h3></summary>"
        )
        summary_writer.log()
        write_project_summary(config, project_statuses, summary_writer)
        write_summary_dirty_sections(combined_dirty, sections, summary_writer)
        summary_writer.log("</details>")
        print("::endgroup::")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
