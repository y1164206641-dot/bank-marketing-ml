"""Bootstrap a local, workspace-contained Python dependency directory without pip.

Why this exists
---------------
The DSH file sandbox blocks pip's temporary extraction directories on this
machine (``Permission denied`` on ``*.whl.metadata``), but plain HTTP downloads
into the workspace work.  This script resolves an explicit set of wheels from
PyPI's JSON API into a local cache and then unzips them into ``.pydeps``;
the project runs with::

    set PYTHONPATH=<repo>\\.pydeps

Stages are split (``--only download`` / ``--only extract``) because writing a
large wheel and immediately re-opening it in the same process proved flaky on
this filesystem; two short processes are reliable.

It is dependency-free (stdlib only) and records resolved versions in
``.pydeps/VERSIONS.txt``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

PY_TAG = "cp312-cp312-win_amd64"
PURE_TAGS = {"py3-none-any", "py2.py3-none-any"}

PACKAGES = [
    "scikit-learn",
    "scipy",
    "joblib",
    "cloudpickle",
    "narwhals",
    "threadpoolctl",
    "matplotlib",
    "contourpy",
    "kiwisolver",
    "fonttools",
    "cycler",
    "packaging",
    "pyparsing",
    "python-dateutil",
]

PYPI_JSON = "https://pypi.org/pypi/{name}/json"
PURE = {"joblib", "cloudpickle", "narwhals", "threadpoolctl", "cycler",
        "packaging", "pyparsing", "fonttools", "python-dateutil"}


def version_key(v: str):
    parts = re.findall(r"\d+", v.split("+")[0])
    return tuple(int(p) for p in parts[:4]) + (0,) * (4 - len(parts[:4]))


def installed_version(pkg: str) -> str | None:
    try:
        return __import__(pkg).__version__
    except Exception:
        return None


def parse_numpy_constraints(requires_dist: list[str] | None) -> list[str]:
    out = []
    for line in requires_dist or []:
        if "numpy" not in line.lower():
            continue
        m = re.search(r"numpy\s*([<>=!~]+[^;\s,]+)", line, re.I)
        if m:
            out.append(m.group(1).strip())
    return out


def satisfies(version: str, constraints: list[str]) -> bool:
    v = version_key(version)

    def check(token: str) -> bool:
        op, raw = re.match(r"(<=|>=|==|!=|<|>)\s*([\w.]+)", token).groups()
        t = version_key(raw)
        return {"<": v < t, "<=": v <= t, ">": v > t, ">=": v >= t,
                "==": v == t, "!=": v != t}[op]

    return all(check(c) for c in constraints)


def pick_file(files: list[dict], prefer_pure: bool) -> dict | None:
    order = (PURE_TAGS, {PY_TAG}) if prefer_pure else ({PY_TAG}, PURE_TAGS)
    for tagset in order:
        for f in files:
            if f.get("yanked"):
                continue
            name = f["filename"]
            if name.endswith(".whl") and any(name.endswith(f"-{t}.whl") for t in tagset):
                if "win_amd64" in name or "none-any" in name:
                    return f
    return None


def resolve(name: str, numpy_version: str) -> dict:
    with urllib.request.urlopen(PYPI_JSON.format(name=name), timeout=60) as r:
        meta = json.load(r)
    versions = sorted((v for v in meta["releases"] if version_key(v) > (0,)),
                      key=version_key, reverse=True)
    for v in versions:
        f = pick_file(meta["releases"][v], name in PURE)
        if not f:
            continue
        constraints = parse_numpy_constraints(meta["info"].get("requires_dist"))
        if constraints and not satisfies(numpy_version, constraints):
            continue
        return {"version": v, "url": f["url"], "filename": f["filename"],
                "size": f["size"]}
    raise SystemExit(f"could not resolve a wheel for {name}")


def download(url: str, dest: Path, attempts: int = 5) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            print(f"  downloading {dest.name} (attempt {i})")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as r, open(part, "wb") as fh:
                expected = int(r.headers.get("Content-Length") or 0)
                written = 0
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
                    written += len(chunk)
            if expected and written != expected:
                raise IOError(f"short read {written}/{expected}")
            with zipfile.ZipFile(part) as zf:
                bad = zf.testzip()
                if bad is not None:
                    raise IOError(f"corrupt member: {bad}")
            part.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            part.unlink(missing_ok=True)
            print(f"    failed: {exc}")
    raise SystemExit(f"could not download {dest.name}: {last}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=str(Path(__file__).resolve().parents[1] / ".pydeps"))
    ap.add_argument("--cache", default=str(Path(__file__).resolve().parents[1] / "wheels"))
    ap.add_argument("--only", choices=["download", "extract", "both"], default="both")
    args = ap.parse_args()

    target, cache = Path(args.target).resolve(), Path(args.cache).resolve()
    target.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    numpy_version = installed_version("numpy")
    if not numpy_version:
        raise SystemExit("numpy must be importable in the base interpreter")

    resolved = []
    for pkg in PACKAGES:
        info = resolve(pkg, numpy_version)
        info["package"] = pkg
        resolved.append(info)

    if args.only in ("download", "both"):
        print(f"base numpy: {numpy_version}")
        for info in resolved:
            dest = cache / info["filename"]
            if dest.exists() and zipfile.is_zipfile(dest):
                print(f"  cached {dest.name}")
            else:
                download(info["url"], dest)
        (cache / "resolved.json").write_text(json.dumps(resolved, indent=2))

    if args.only in ("extract", "both"):
        if not (cache / "resolved.json").exists():
            raise SystemExit("run with --only download first")
        for info in json.loads((cache / "resolved.json").read_text()):
            whl = cache / info["filename"]
            print(f"  extracting {whl.name}")
            with zipfile.ZipFile(whl) as zf:
                zf.extractall(target)
        lines = [f"base-python: {sys.version.split()[0]}", f"numpy: {numpy_version}"]
        lines += [f"{i['package']}: {i['version']}" for i in resolved]
        (target / "VERSIONS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("done ->", target)


if __name__ == "__main__":
    main()
