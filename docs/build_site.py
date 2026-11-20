#!/usr/bin/env python3
"""Build the docs for every product and combine them into one site.

    python3 docs/build_site.py                          all products, as unstable
    python3 docs/build_site.py --product python         one product
    python3 docs/build_site.py --product cpp --label 3.5.0
    python3 docs/build_site.py --skip-build             reuse the last build

The result goes to docs/_site/ and is what gets published:

    index.html, contributors/, ...   the top-level site (docs/site)
    cpp/index.html                   redirect to unstable/
    cpp/nv-versions.json             the versions in the version switcher
    cpp/unstable/, cpp/3.5.0/, ...   one folder per version
    python/...                       the same layout
"""

import argparse
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import registry  # noqa: E402

VENV = HERE / "env"


def run(command, cwd=HERE):
    """Run a command and stop if it fails. A string runs through the shell."""
    shown = command if isinstance(command, str) else shlex.join(map(str, command))
    print(f"  $ {shown}")
    if subprocess.run(command, shell=isinstance(command, str), cwd=cwd).returncode:
        raise SystemExit(f"error: command failed: {shown}")


def python_bin():
    python = VENV / "bin" / "python"
    return str(python) if python.exists() else sys.executable


def ensure_venv():
    if (VENV / "bin" / "python").exists():
        return
    print("creating docs/env")
    run([sys.executable, "-m", "venv", VENV])
    pip = [VENV / "bin" / "python", "-m", "pip", "install", "--quiet"]
    run([*pip, "-r", HERE / "requirements.txt"])


def copy_published_site(branch, site_dir):
    """Copy the site that is already published on `branch` into site_dir."""
    repo = HERE.parent
    found = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{branch}^{{commit}}"],
        cwd=repo,
        capture_output=True,
    )
    if found.returncode:
        print(f"  {branch} does not exist yet, so there is nothing to copy")
        return

    # The site is in the branch's docs/ folder. Run git archive from the
    # repository root, because it resolves "<branch>:docs" relative to the
    # current directory.
    archive = subprocess.run(
        ["git", "archive", f"{branch}:docs"], cwd=repo, capture_output=True
    )
    if archive.returncode:
        raise SystemExit(f"error: cannot read docs/ from {branch}")

    tar = site_dir / "published.tar"
    tar.write_bytes(archive.stdout)
    shutil.unpack_archive(tar, site_dir, format="tar")
    tar.unlink()
    print(f"  copied the published site from {branch}")


def build_product(entry, label):
    print(f"building {entry['key']} at {label}")
    run(entry["build"].format(label=label))


def copy_product(entry, label, site_dir):
    """Copy a product's build output into the site."""
    source = registry.ARTIFACTS / entry["key"]
    if not source.is_dir():
        raise SystemExit(f"error: no build output for {entry['key']} in {source}")

    target = site_dir / entry["key"]
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        # Replace the version folder as a whole, so pages deleted from the
        # sources also disappear from the site.
        destination = target / item.name
        if destination.is_dir():
            shutil.rmtree(destination)
        elif destination.exists():
            destination.unlink()
        if item.is_dir():
            shutil.copytree(item, destination, symlinks=True)
        else:
            shutil.copy2(item, destination)

    version_dir = target / label
    pages = len(list(version_dir.rglob("*.html"))) if version_dir.is_dir() else 0
    print(f"  copied {entry['key']}/{label}: {pages} pages")


def version_order(versions):
    """unstable first, then releases from newest to oldest."""
    releases = sorted(
        (v for v in versions if v != registry.DEVELOPMENT),
        key=lambda v: tuple(int(part) for part in v.split(".")),
        reverse=True,
    )
    first = [registry.DEVELOPMENT] if registry.DEVELOPMENT in versions else []
    return first + releases


def write_version_list(product_dir):
    """Write nv-versions.json, which the version switcher reads.

    It lists the version folders in product_dir, which are the ones copied
    from the published site plus the one just built.
    """
    names = [p.name for p in product_dir.iterdir() if p.is_dir()]
    versions = version_order([n for n in names if registry.VERSION.match(n)])
    base = f"{registry.site_url()}/{product_dir.name}"
    entries = [{"version": v, "url": f"{base}/{v}/"} for v in versions]
    path = product_dir / "nv-versions.json"
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"  {product_dir.name}/nv-versions.json: {', '.join(versions)}")


def build_top_level_site(site_dir):
    """Build docs/site, the pages at the site root."""
    # -d keeps Sphinx's cache out of the published site.
    run(
        [python_bin(), "-m", "sphinx.cmd.build", "-b", "html", "-q", "-W"]
        + ["--keep-going", "-d", HERE / "_build" / "site-doctrees"]
        + [HERE / "site", site_dir]
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the docs site into docs/_site.")
    parser.add_argument(
        "--label",
        default=registry.DEVELOPMENT,
        help="version to build: unstable or MAJOR.MINOR.PATCH",
    )
    parser.add_argument(
        "--product",
        action="append",
        dest="products",
        help="build only this product (repeatable)",
    )
    parser.add_argument("--site", default=HERE / "_site", help="output directory")
    parser.add_argument(
        "--skip-build", action="store_true", help="reuse the existing build output"
    )
    parser.add_argument(
        "--published-branch",
        default="",
        help="start from the site published on this branch",
    )
    args = parser.parse_args(argv)

    if not registry.VERSION.match(args.label):
        raise SystemExit(
            f"error: --label must be unstable or MAJOR.MINOR.PATCH, got {args.label!r}"
        )

    keys = args.products or [entry["key"] for entry in registry.products()]
    entries = [registry.product(key) for key in keys]
    site_dir = pathlib.Path(args.site).resolve()
    site_dir.mkdir(parents=True, exist_ok=True)

    # The product builds read this to find each other's objects.inv files.
    os.environ["CCCL_DOCS_SITE_DIR"] = str(site_dir)

    ensure_venv()
    if args.published_branch:
        copy_published_site(args.published_branch, site_dir)

    for entry in entries:
        if not args.skip_build:
            build_product(entry, args.label)
        copy_product(entry, args.label, site_dir)

    for entry in registry.products():
        if (site_dir / entry["key"]).is_dir():
            write_version_list(site_dir / entry["key"])

    # Rebuild the top-level site only on unstable runs, which build main. A
    # release run would otherwise replace the guides with the older copy in its
    # release branch. Build it last, so its links into the products resolve
    # against the pages that were just built.
    if args.label == registry.DEVELOPMENT:
        build_top_level_site(site_dir)
    else:
        print("keeping the published top-level site")

    # Without this file, GitHub Pages runs Jekyll, which drops folders whose
    # names start with "_", such as Sphinx's _static.
    (site_dir / ".nojekyll").touch()

    print(f"\nbuilt {site_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
