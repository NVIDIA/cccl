#!/usr/bin/env python3
"""Read registry.yml, and answer the questions that depend on the product list.

The landing index, each product's cross-reference mapping, the tag grammar that
routes a release, the directories a build must skip and the script that
assembles the site all need the product list, and all of them read it here.
"""

import os
import pathlib
import re

import yaml

HERE = pathlib.Path(__file__).resolve().parent
REGISTRY = HERE / "registry.yml"

# Where build_site.py assembles the tree it is about to publish. The conf.py
# files run in subprocesses of that script rather than importing it, so the
# directory arrives through the environment -- the same route the site URL
# already takes.
SITE = pathlib.Path(os.environ.get("CCCL_DOCS_SITE_DIR", HERE / "_site"))

# Where the product build scripts leave their output, before the assembly places
# it. A product built on its own -- its script run directly, not through
# build_site.py -- finds its siblings' inventories here.
ARTIFACTS = HERE / "_build" / "artifacts" / "docs"

# A published version is the development line or an exact release. No rolling
# MAJOR.MINOR directory, whose content would change under a reader's link, and
# no pre-release taking a released version's URL.
DEVELOPMENT = "unstable"
VERSION = re.compile(r"^(unstable|[0-9]+\.[0-9]+\.[0-9]+)\Z")


def load(path=REGISTRY):
    return yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))


def products(path=REGISTRY):
    """Every product, in build order -- also the order the landing index uses."""
    return load(path)["products"]


def product(key, path=REGISTRY):
    for entry in products(path):
        if entry["key"] == key:
            return entry
    raise SystemExit(
        f"error: no product {key!r} in registry.yml.\n"
        f"       It lists: {', '.join(p['key'] for p in products(path))}"
    )


def resolve_tag(tag, path=REGISTRY):
    """``"v3.5.0"`` -> ``("cpp", "3.5.0")``.

    The prefix selects the product and the remainder is the version, so one tag
    carries both and nothing else supplies either. A prefix that matches no
    product, or a remainder that is not an exact release, stops the build rather
    than publishing under a name nobody chose.

    Longest prefix first, so a product whose prefix extends another's -- a
    hypothetical ``python-compute-`` beside ``python-`` -- cannot be shadowed by
    the shorter one.
    """
    text = (tag or "").strip()
    for entry in sorted(products(path), key=lambda p: -len(p["tag_prefix"])):
        prefix = entry["tag_prefix"]
        if text.startswith(prefix):
            version = text[len(prefix) :]
            if VERSION.match(version) and version != DEVELOPMENT:
                return entry["key"], version
            break

    expected = " or ".join(
        f"{p['tag_prefix']}MAJOR.MINOR.PATCH" for p in products(path)
    )
    raise SystemExit(
        f"error: {text!r} is not a release tag for any product.\n"
        f"       Expected {expected}.\n"
        "       Pre-release tags are not published to the stable switcher."
    )


def intersphinx_mapping(key, base_url, target=DEVELOPMENT, path=REGISTRY, site=None):
    """Where this product resolves cross-references to its siblings.

    Sphinx resolves a cross-reference inside one project, so a page in one
    project cannot reach a page in another by name unless it is handed the
    other's inventory. That is what intersphinx does: the other project
    publishes ``objects.inv``, this one reads it, and the reference resolves by
    symbol rather than by a URL somebody typed. A symbol that moves fails the
    build; a URL that moves fails silently, for a reader, later.

    ``key`` is the calling product, and it maps only the products listed above
    it: build_site.py builds in registry order, so a product listed below has
    no inventory yet -- in the run that adds it, not even a published one. The
    root project passes ``None`` and, built last, maps them all.

    ``target`` is the siblings' version. CCCL's products release independently,
    so there is no matching version to point at -- C++ 3.5.0 has no
    corresponding Python release -- and siblings are targeted at the development
    line, which always exists. Changing that policy is this one argument.

    The tuple's two halves are deliberately different. The URL is where the link
    *points*, so it must be the published address a reader will follow. The
    second element is where the inventory is *read from*, and that should be the
    tree this run is about to publish: the published site is a publish behind by
    construction, so a page added and referenced in the same commit would not
    resolve against it.
    """
    base = base_url.rstrip("/")
    root = pathlib.Path(SITE if site is None else site)
    mapping = {}
    if key is not None:
        product(key, path)  # an unknown key is an error, not "map everything"
    for entry in products(path):
        sibling = entry["key"]
        if sibling == key:
            break  # listed later means built later: no inventory yet
        # The assembled site first, because it is what is about to be published.
        # Then this run's raw build output, for a product built on its own rather
        # than through build_site.py. The published site is the last resort.
        inventory, where = None, "published site over HTTP"
        for candidate, source in (
            (root / sibling / target / "objects.inv", "assembled site"),
            (ARTIFACTS / sibling / target / "objects.inv", "this run's build"),
        ):
            if candidate.is_file():
                inventory, where = str(candidate), source
                break
        mapping[sibling] = (f"{base}/{sibling}/{target}/", inventory)
        # Whether the check ran against fresh data is not otherwise answerable
        # from a build log.
        print(f"  intersphinx {sibling}: {where}")
    return mapping


def exclude_patterns(key, path=REGISTRY):
    """Source roots this product's build must not sweep in.

    Only matters for a product whose source root contains another's. The C++
    sources sit at docs/, so docs/python/ and docs/site/ are inside its tree,
    and Sphinx would otherwise publish both under a C++ version they never
    shipped under. Derived from the registry so a third product is excluded by
    being added rather than by being remembered.
    """
    config = load(path)
    mine = (HERE / product(key, path)["source_root"]).resolve()

    roots = [entry["source_root"] for entry in config["products"]]
    roots.append(config["root"]["source_root"])

    patterns = []
    for root in roots:
        other = (HERE / root).resolve()
        if other == mine:
            continue
        try:
            patterns.append(str(other.relative_to(mine)))
        except ValueError:
            continue  # not inside this source root; nothing to exclude
    return sorted(patterns)
