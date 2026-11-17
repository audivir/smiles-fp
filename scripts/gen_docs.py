"""Generates DOCS.md from the docstrings of the type stubs.

Usage: `python scripts/gen_docs.py` writes the file, `--check` fails if it is out of date.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "python" / "smiles_fp"
INIT = PACKAGE / "__init__.py"
SOURCES = (PACKAGE / "_smiles_fp.pyi", PACKAGE / "helpers.py", PACKAGE / "search.py")
OUTPUT = ROOT / "DOCS.md"
SECTIONS = ("Args", "Returns", "Raises")

# every public function belongs to exactly one block, which sets the order of the documentation.
GROUPS: dict[str, tuple[str, list[str]]] = {
    "Molecules and SMILES": (
        "Conversion between SMILES and RDKit molecules, one at a time or multi-threaded.",
        ["mol_from_smi", "get_mols", "mols_from_smiles", "mols_to_smiles"],
    ),
    "Cleaning": (
        "Standardization of SMILES and molecules in one multi-threaded pass.",
        ["clean"],
    ),
    "Scaffolds and tautomers": (
        "Murcko scaffolds and tautomers, computed multi-threaded.",
        [
            "murcko_scaffolds",
            "canonical_tautomers",
            "enumerate_tautomers",
            "score_tautomers",
            "pick_canonical_tautomers",
        ],
    ),
    "Fingerprints": (
        "Morgan fingerprints and their binary files.",
        [
            "get_morgan_fps",
            "save_morgan_fps_from_smiles",
            "save_fingerprints",
            "load_fingerprints",
            "load_fingerprints_as_ndarray",
        ],
    ),
    "Similarity in memory": (
        "Tanimoto similarities of fingerprints that fit into memory.",
        [
            "bulk_tanimoto_parallel",
            "bulk_tanimoto_parallel_topk",
            "internal_tanimoto_parallel",
            "to_matrix",
        ],
    ),
    "Similarity from files": (
        "Tanimoto similarities of fingerprint files, read with memory-mapping.",
        ["bulk_tanimoto_mmap", "bulk_tanimoto_mmap_topk", "internal_tanimoto_mmap"],
    ),
    "Search": (
        "Top-k search and chunked similarity matrices for large collections.",
        ["similarity_search", "windowed_bulk_tanimoto"],
    ),
}


def public_names() -> list[str]:
    """Reads `__all__` of the package, which sets the order of the documentation."""
    for node in ast.parse(INIT.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return [str(ast.literal_eval(e)) for e in getattr(node.value, "elts", [])]
    raise ValueError("No __all__ in the package")


def param(arg: ast.arg, default: ast.expr | None) -> str:
    """Formats a parameter as `name: type = default`."""
    text = arg.arg
    if arg.annotation is not None:
        text += f": {ast.unparse(arg.annotation)}"
    if default is not None:
        text += f" = {ast.unparse(default)}"
    return text.replace("'", '"')


def signature(node: ast.FunctionDef) -> str:
    """Formats the signature with one parameter per line."""
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    parts = [param(a, d) for a, d in zip(positional, defaults, strict=True)]
    if args.kwonlyargs:
        parts.append("*")
        parts += [param(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=True)]
    returns = f" -> {ast.unparse(node.returns)}".replace("'", '"') if node.returns else ""
    body = "".join(f"    {p},\n" for p in parts)
    return f"def {node.name}(\n{body}){returns}"


def split_sections(doc: str) -> tuple[str, dict[str, list[str]]]:
    """Splits a Google-style docstring into the summary text and its sections."""
    lines = inspect.cleandoc(doc).splitlines()
    text: list[str] = []
    sections: dict[str, list[str]] = {}
    current = text
    for line in lines:
        name = line.rstrip(":")
        if line.endswith(":") and name in SECTIONS:
            current = sections.setdefault(name, [])
        else:
            current.append(line)
    return "\n".join(text).strip(), sections


def items(lines: list[str]) -> list[tuple[str, str]]:
    """Groups the lines of Args or Raises into pairs of name and description."""
    result: list[tuple[str, str]] = []
    base = min((len(line) - len(line.lstrip()) for line in lines if line.strip()), default=0)
    for line in lines:
        indent = len(line) - len(line.lstrip())
        if not line.strip():
            continue
        if indent > base and result:
            result[-1] = (result[-1][0], f"{result[-1][1]} {line.strip()}")
        else:
            name, _, desc = line.partition(":")
            result.append((name.strip(), desc.strip()))
    return result


def decorators(node: ast.FunctionDef) -> list[str]:
    """Returns the names of the decorators, like `overload` or `deprecated`."""
    names = []
    for deco in node.decorator_list:
        target = deco.func if isinstance(deco, ast.Call) else deco
        names.append(ast.unparse(target))
    return names


def deprecation(nodes: list[ast.FunctionDef]) -> str:
    """Returns the message of a `deprecated` decorator, or an empty string."""
    for node in nodes:
        for deco in node.decorator_list:
            if isinstance(deco, ast.Call) and ast.unparse(deco.func) == "deprecated":
                return str(ast.literal_eval(deco.args[0]))
    return ""


def render_function(name: str, nodes: list[ast.FunctionDef]) -> str:
    """Renders the signatures and the docstring of a function.

    Overloads are shown instead of the implementation, which has the docstring if they have none.
    """
    overloads = [n for n in nodes if "overload" in decorators(n)]
    shown = overloads or nodes
    docs = [ast.get_docstring(n) for n in nodes]
    doc = next((d for d in docs if d), "")
    text, sections = split_sections(doc)
    out = [f"### `{name}`", "", "```python", "\n\n".join(signature(n) for n in shown), "```", ""]
    if message := deprecation(nodes):
        out += [f"**Deprecated.** {message}", ""]
    out += [text, ""] if text else []
    if "Args" in sections:
        out += ["**Arguments**", ""]
        out += [f"- `{n}`: {d}" for n, d in items(sections["Args"])] + [""]
    if "Returns" in sections:
        out += ["**Returns**", "", " ".join(s.strip() for s in sections["Returns"]).strip(), ""]
    if "Raises" in sections:
        out += ["**Raises**", ""]
        out += [f"- `{n}`: {d}" for n, d in items(sections["Raises"])] + [""]
    return "\n".join(out)


def collect() -> dict[str, list[ast.FunctionDef]]:
    """Collects the public functions of all sources, with their overloads in one list."""
    functions: dict[str, list[ast.FunctionDef]] = {}
    for source in SOURCES:
        for node in ast.parse(source.read_text()).body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                functions.setdefault(node.name, []).append(node)
    return functions


def check_groups(names: list[str]) -> None:
    """Fails if the blocks and the public names do not match."""
    grouped = [n for _, members in GROUPS.values() for n in members]
    missing = sorted(set(names) - set(grouped))
    unknown = sorted(set(grouped) - set(names))
    duplicate = sorted({n for n in grouped if grouped.count(n) > 1})
    if missing or unknown or duplicate:
        raise ValueError(
            f"Fix GROUPS in {Path(__file__).name}: no block for {missing}, "
            f"not public {unknown}, in several blocks {duplicate}"
        )


def anchor(title: str) -> str:
    """Builds the GitHub anchor of a heading."""
    return title.lower().replace(" ", "-")


def render() -> str:
    """Builds the whole documentation."""
    functions = collect()
    names = public_names()
    check_groups(names)
    out = [
        "# smiles-fp API",
        "",
        "<!-- generated by scripts/gen_docs.py from the docstrings, do not edit -->",
        "",
    ]
    for title, (summary, members) in GROUPS.items():
        out += [f"- [{title}](#{anchor(title)}): {summary}"]
        out += [f"  - [`{n}`](#{n})" for n in members]
    out += [""]
    for title, (summary, members) in GROUPS.items():
        out += [f"## {title}", "", summary, ""]
        out += [render_function(n, functions[n]) for n in members]
    return "\n".join(out).rstrip("\n") + "\n"


def main() -> int:
    """Writes the documentation, or checks it with `--check`."""
    text = render()
    if "--check" in sys.argv:
        if not OUTPUT.exists() or OUTPUT.read_text() != text:
            sys.stderr.write("DOCS.md is out of date, run `python scripts/gen_docs.py`\n")
            return 1
        return 0
    OUTPUT.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
