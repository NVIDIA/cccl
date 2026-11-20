# Sphinx configuration for the top-level site, which holds the landing page and
# the contributor, maintainer and infrastructure guides. These pages describe the
# repository rather than one release, so they have no version and are published
# once, at the site root.

import pathlib
import sys
from datetime import datetime

# Add docs/ to the path for registry.py
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import registry  # noqa: E402

_site = registry.load()["site"]

project = _site["title"]
author = "NVIDIA Corporation"
copyright = f"{datetime.now().year}, NVIDIA Corporation"
version = release = ""

extensions = ["myst_parser", "sphinx.ext.intersphinx"]

# The guides link to C++ pages, as in :doc:`cpp:cub/benchmarking`.
intersphinx_mapping = registry.intersphinx_mapping(None, registry.site_url())

# The landing page links to cpp/ and python/, which are built separately. Treat
# Markdown links as external so MyST does not report them as missing documents.
myst_all_links_external = True

html_theme = "nvidia_sphinx_theme"
html_baseurl = f"{registry.site_url()}/"
html_theme_options = {
    "switcher": None,
    "show_version_warning_banner": False,
    "navbar_align": "left",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}


def write_landing_page(app, config):
    """Write index.md with a link to each product in registry.yml."""
    lines = [
        f"# {_site['title']}",
        "",
        _site["summary"].strip(),
        "",
        "## Documentation",
        "",
    ]
    for entry in registry.products():
        lines += [
            f"### [{entry['name']}]({entry['key']}/)",
            "",
            entry["summary"].strip(),
            "",
        ]
    lines += [
        "```{toctree}",
        ":hidden:",
        ":maxdepth: 2",
        "",
        "contributors/index",
        "maintainers/index",
        "infrastructure/index",
        "```",
        "",
    ]
    (pathlib.Path(app.srcdir) / "index.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def setup(app):
    # config-inited runs before Sphinx reads the sources, so index.md exists by then.
    app.connect("config-inited", write_landing_page)
