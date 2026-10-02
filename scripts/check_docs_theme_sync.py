"""Check that the NeuroTrade and knowledge-base docs sites share the same theme.

Both sites must look and behave the same. This script compares this repo with its
sibling checkout and exits 1 on any drift:

  * the shared files below must be byte-identical (this script is one of them);
  * the `theme:` block of mkdocs.yml must match, except the `logo:` icon;
  * the zensical version pin and the deployment-time stamp command must match.

Usage:  python scripts/check_docs_theme_sync.py [path-to-other-repo]
The other repo defaults to ../knowledge-base or ../neuro-trade next to this one.
Fix drift by making the same change in both repos, then commit both.
"""

import difflib
import re
import sys
from pathlib import Path

SHARED_FILES = [
    "docs/stylesheets/extra.css",
    "docs/javascripts/extra.js",
    "docs/javascripts/build-info.js",
    "scripts/check_docs_theme_sync.py",
]
REPOS = {
    "vishalhulawale/neuro-trade": ("requirements-docs.txt", ".github/workflows/docs.yml"),
    "vishalhulawale/knowledge-base": ("requirements.txt", ".github/workflows/deploy.yml"),
}


def read(path):
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def repo_name(root):
    match = re.search(r"^repo_name:\s*(\S+)", read(root / "mkdocs.yml"), re.M)
    if not match or match.group(1) not in REPOS:
        sys.exit(f"{root}: mkdocs.yml repo_name is not one of {', '.join(REPOS)}")
    return match.group(1)


def theme_block(root):
    """The mkdocs.yml `theme:` block, without the site-specific logo icon."""
    lines = read(root / "mkdocs.yml").split("\n")
    start = lines.index("theme:")
    end = next(i for i in range(start + 1, len(lines)) if lines[i] and not lines[i][0].isspace())
    return "\n".join(l for l in lines[start:end] if not l.strip().startswith("logo:")).strip() + "\n"


def matching_line(root, rel, pattern):
    found = [l.strip() for l in read(root / rel).split("\n") if re.search(pattern, l)]
    return "\n".join(found or [f"(no line matching {pattern!r} in {rel})"]) + "\n"


def main():
    here = Path(__file__).resolve().parent.parent
    name = repo_name(here)
    if len(sys.argv) > 1:
        there = Path(sys.argv[1]).resolve()
    else:
        other = next(n for n in REPOS if n != name)
        there = here.parent / other.split("/")[1]
    if not (there / "mkdocs.yml").exists():
        sys.exit(f"Other repo not found at {there}. Pass its path as the first argument.")
    other_name = repo_name(there)

    checks = [(rel, lambda r, rel=rel: read(r / rel) if (r / rel).exists() else "(missing)\n")
              for rel in SHARED_FILES]
    checks.append(("mkdocs.yml theme: block (logo excluded)", theme_block))
    checks.append(("zensical version pin", lambda r: matching_line(
        r, REPOS[repo_name(r)][0], r"^zensical")))
    checks.append(("deployment-time stamp step", lambda r: matching_line(
        r, REPOS[repo_name(r)][1], r"DOCS_BUILD_TIME")))

    drift = 0
    for label, get in checks:
        a, b = get(here), get(there)
        if a != b:
            drift += 1
            print(f"DRIFT: {label}")
            sys.stdout.writelines(difflib.unified_diff(
                a.splitlines(True), b.splitlines(True), name, other_name))
            print()
    if drift:
        print(f"{drift} difference(s) between {name} and {other_name}. Make the same change in both.")
        return 1
    print(f"Docs theme in sync: {name} and {other_name}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
