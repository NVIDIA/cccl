import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_MAX_COMMENT_BYTES = 60_000
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return payload


def require_config(value: Any, index: int) -> tuple[str, str]:
    if not isinstance(value, dict):
        raise SystemExit(f"matrix include entry {index} must be an object")
    config_id = value.get("id")
    config_name = value.get("name")
    if not isinstance(config_id, str) or not ID_RE.fullmatch(config_id):
        raise SystemExit(f"matrix include entry {index} has an invalid id")
    if not isinstance(config_name, str) or not config_name:
        raise SystemExit(f"matrix include entry {index} has an invalid name")
    return config_id, html.escape(config_name)


def summary_line(fragment: str, config_name: str) -> str:
    for line in fragment.splitlines():
        if line.startswith("<summary>"):
            return line
    return (
        f"<summary><strong>⚠️ {config_name}</strong> — reporting unavailable</summary>"
    )


def missing_fragment_section(
    config_name: str,
    artifacts_url: str,
    *,
    artifact_download_failed: bool,
) -> str:
    reason = (
        "The compile-time comment artifacts could not be downloaded."
        if artifact_download_failed
        else "The compile-time benchmark did not produce a comment fragment."
    )
    return "\n".join(
        [
            "<details>",
            (
                f"<summary><strong>⚠️ {config_name}</strong> — "
                "reporting unavailable</summary>"
            ),
            "",
            reason,
            "",
            f"**Artifacts:** [reports and traces]({artifacts_url})",
            "",
            "</details>",
        ]
    )


def compact_section(
    summary: str,
    config_id: str,
    *,
    has_fragment: bool,
    artifact_download_failed: bool,
) -> str:
    if has_fragment:
        message = (
            "Detailed tables are available in the "
            f"<code>compile-time-{config_id}-comment</code> artifact."
        )
    elif artifact_download_failed:
        message = "The compile-time comment artifacts could not be downloaded."
    else:
        message = "The compile-time benchmark did not produce a comment fragment."
    return f"<details>\n{summary}\n\n{message}\n\n</details>"


def preamble(*, artifacts_url: str, compact: bool, download_failed: bool) -> str:
    lines = [
        "<!-- cccl-compile-time-bench -->",
        "## ⏱️ CCCL compile-time benchmark comparisons",
        "",
    ]
    if compact:
        lines.extend(
            [
                "The detailed tables exceed the combined comment size limit. Full",
                "reports remain available in the per-configuration comment artifacts.",
                "",
                f"**Artifacts:** [reports and traces]({artifacts_url})",
                "",
            ]
        )
    else:
        lines.extend(["Each configuration is reported independently below.", ""])
    if download_failed:
        lines.extend(
            [
                "> [!WARNING]",
                "> Compile-time comment artifacts could not be downloaded; missing",
                "> configuration sections report that aggregation failure explicitly.",
                "",
            ]
        )
    return "\n".join(lines)


def render_combined_comment(
    matrix: dict[str, Any],
    fragments_dir: Path,
    *,
    artifacts_url: str,
    artifact_download_failed: bool = False,
    max_comment_bytes: int = DEFAULT_MAX_COMMENT_BYTES,
) -> str | None:
    configs = matrix.get("include", [])
    if not isinstance(configs, list):
        raise SystemExit("matrix include field must be an array")
    if not configs:
        return None
    if max_comment_bytes <= 0:
        raise SystemExit("max comment bytes must be positive")

    detailed_sections: list[str] = []
    compact_sections: list[str] = []
    for index, config in enumerate(configs):
        config_id, config_name = require_config(config, index)
        fragment_path = (
            fragments_dir / f"compile-time-{config_id}-comment" / "comment.md"
        )
        fragment = ""
        if fragment_path.is_file():
            fragment = fragment_path.read_text(encoding="utf-8").strip()

        if fragment:
            detailed_sections.append(fragment)
            config_summary = summary_line(fragment, config_name)
        else:
            detailed_sections.append(
                missing_fragment_section(
                    config_name,
                    artifacts_url,
                    artifact_download_failed=artifact_download_failed,
                )
            )
            config_summary = summary_line("", config_name)
        compact_sections.append(
            compact_section(
                config_summary,
                config_id,
                has_fragment=bool(fragment),
                artifact_download_failed=artifact_download_failed,
            )
        )

    detailed = (
        preamble(
            artifacts_url=artifacts_url,
            compact=False,
            download_failed=artifact_download_failed,
        ).rstrip()
        + "\n\n"
        + "\n\n".join(detailed_sections)
        + "\n"
    )
    if len(detailed.encode("utf-8")) <= max_comment_bytes:
        return detailed

    compact = (
        preamble(
            artifacts_url=artifacts_url,
            compact=True,
            download_failed=artifact_download_failed,
        ).rstrip()
        + "\n\n"
        + "\n\n".join(compact_sections)
        + "\n"
    )
    if len(compact.encode("utf-8")) > max_comment_bytes:
        raise SystemExit("compact comment exceeds the configured size limit")
    return compact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine compile-time configuration fragments into one PR comment."
    )
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--fragments-dir", type=Path, required=True)
    parser.add_argument("--artifacts-url", required=True)
    parser.add_argument("--artifact-download-outcome", default="success")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_COMMENT_BYTES)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    comment = render_combined_comment(
        load_json(args.matrix),
        args.fragments_dir,
        artifacts_url=args.artifacts_url,
        artifact_download_failed=args.artifact_download_outcome != "success",
        max_comment_bytes=args.max_bytes,
    )
    if comment is None:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(comment, encoding="utf-8")


if __name__ == "__main__":
    main()
