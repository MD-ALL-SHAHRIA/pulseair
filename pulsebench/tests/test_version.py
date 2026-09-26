"""The version is declared in four files; they must agree.

``pulsebench.__version__`` sat at 0.1.0 while the repository released v1.0.0, because
nothing checked. Four files carry a version -- the package, the packaging metadata and
two citation files -- and a reader who cites the wrong one cites something that does
not exist. Comparing them to each other rather than to a constant means a bump needs
no edit here.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pulsebench

REPO_ROOT = Path(__file__).resolve().parents[2]


def _cff_version(path: Path) -> str:
    # A one-key regex rather than a YAML dependency: the citation files are validated
    # as YAML by their own CI job, so this only has to read one line.
    m = re.search(r"^version:\s*(\S+)\s*$", path.read_text(), re.M)
    assert m, f"{path} declares no version"
    return m.group(1).strip("\"'")


def test_package_and_packaging_metadata_agree():
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        declared = tomllib.load(fh)["project"]["version"]
    assert pulsebench.__version__ == declared, (
        f"pulsebench.__version__ is {pulsebench.__version__} but pyproject.toml says "
        f"{declared}. pip would report the second and `import pulsebench` the first.")


def test_both_citation_files_agree_with_the_package():
    for rel in ("CITATION.cff", "pulsebench/CITATION.cff"):
        assert _cff_version(REPO_ROOT / rel) == pulsebench.__version__, (
            f"{rel} cites version {_cff_version(REPO_ROOT / rel)}, but the package is "
            f"{pulsebench.__version__}")


def test_the_version_is_a_release_number_not_a_placeholder():
    assert re.fullmatch(r"\d+\.\d+\.\d+", pulsebench.__version__), (
        f"{pulsebench.__version__!r} is not an x.y.z release number")
