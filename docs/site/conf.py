"""The root documentation project: unversioned, and about the repository.

It indexes the products and holds what describes the repository rather than one
release of one library: the contributor, maintainer and infrastructure docs. A
contributor guide has no 3.5.0 and no 1.2.0, so this project has no version and
no switcher, and is published once at the site root.

The index page is generated from registry.yml, so a new product appears in it
by being added there.
"""

import os
import pathlib
import sys
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import registry

_registry = registry.load()
_site = _registry["site"]

project = _site["title"]
author = "NVIDIA Corporation"
copyright = f"{datetime.now().year}, NVIDIA Corporation"

# No version, deliberately. The switcher and the version stamp belong to the
# product builds; this project has neither.
version = release = ""

extensions = ["myst_parser", "sphinx.ext.intersphinx"]

# The contributor and maintainer docs link into the C++ pages, a separate Sphinx
# project. The root is built last and maps every product.
_site_root = os.environ.get("CCCL_DOCS_SITE_URL", _site["base_url"]).rstrip("/")
intersphinx_mapping = registry.intersphinx_mapping(None, _site_root)

# The index links to product directories that exist only in the assembled site;
# without this MyST reports them as missing documents, which -W makes fatal.
myst_all_links_external = True

html_theme = "nvidia_sphinx_theme"
html_baseurl = f"{_site_root}/"

html_theme_options = {
    "switcher": None,
    "show_version_warning_banner": False,
    "navbar_align": "left",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}


def _write_index(app, config):
    """Generate index.md from the registry, so the two cannot disagree.

    Sphinx runs this on ``config-inited``, before it reads any source file, so
    the page exists by the time the build looks for it. The same hook point the
    C++ build already uses to generate its API stubs.
    """
    products = registry.products()
    lines = [
        f"# {_site['title']}",
        "",
        _site["summary"].strip(),
        "",
        "## Documentation",
        "",
    ]
    for entry in products:
        # Each product's landing page redirects to unstable/, so the index never
        # names a version and never goes stale when one ships.
        lines += [
            f"### [{entry['name']}]({entry['key']}/)",
            "",
            entry["summary"].strip(),
            "",
        ]

    # The repository-level docs this project owns; not registry-driven, since
    # they belong to no product.
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

    path = pathlib.Path(app.srcdir) / "index.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  generated index.md from registry.yml: {len(products)} products")


def setup(app):
    app.connect("config-inited", _write_index)
