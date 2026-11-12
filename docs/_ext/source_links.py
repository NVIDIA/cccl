# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Link documentation pages to plain text and their versioned GitHub source.

Local previews of unpublished commits can target a published revision with
``-D cccl_source_repository=owner/repo -D cccl_source_ref=branch-or-tag``.
The ref must exist in the local checkout; links use its resolved commit ID.
``docs/gen_docs.bash`` enables ``cccl_text_output`` when building both formats.
"""

import os
import subprocess
from pathlib import Path
from urllib.parse import quote

from sphinx.util import logging

logger = logging.getLogger(__name__)


def collect_source_links(app):
    """Index committed docs once; generated source stubs have no GitHub link."""
    app._cccl_source_links = {}
    if app.builder.format != "html":
        return

    repo_root = Path(app.confdir).parent
    try:
        revision = subprocess.check_output(
            [
                "git",
                "rev-parse",
                "--verify",
                f"{app.config.cccl_source_ref}^{{commit}}",
            ],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        paths = subprocess.check_output(
            ["git", "ls-tree", "-rz", "--name-only", revision, "--", "docs"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        # Source archives can still offer the plain-text documentation.
        logger.info(
            "Raw GitHub source links are unavailable for %r: %s; "
            "omitting raw-source links.",
            app.config.cccl_source_ref,
            exc,
        )
        return

    repository = quote(app.config.cccl_source_repository.strip("/"), safe="/")
    for path in paths.split("\0"):
        if path:
            app._cccl_source_links[(repo_root / path).resolve()] = (
                f"https://raw.githubusercontent.com/{repository}/{revision}/"
                f"{quote(path, safe='/')}"
            )


def add_source_link(app, pagename, templatename, context, doctree):
    """Supply a source link only for pages backed by a document."""
    if doctree is None:
        return

    if app.config.cccl_text_output:
        context["cccl_text_url"] = context["pathto"](
            "_text/" + quote(pagename, safe="/") + ".txt", True
        )

    source = Path(app.env.doc2path(pagename)).resolve()
    url = app._cccl_source_links.get(source)
    if url:
        context["cccl_raw_source_url"] = url


def setup(app):
    # Fork builds use their own repository. Local builds default to upstream;
    # either value can be overridden with sphinx-build -D.
    app.add_config_value(
        "cccl_source_repository",
        os.environ.get("GITHUB_REPOSITORY", "NVIDIA/cccl"),
        "html",
    )
    app.add_config_value("cccl_source_ref", "HEAD", "html")
    app.add_config_value("cccl_text_output", False, "html")
    app.connect("builder-inited", collect_source_links)
    app.connect("html-page-context", add_source_link)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
