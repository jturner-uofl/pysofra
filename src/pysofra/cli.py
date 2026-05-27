"""Command-line interface for PySofra.

A thin one-shot CLI for the most common workflow: build a Table 1
from a tabular file and write it out in one of the supported
backends. Aimed at three audiences:

1. Clinical researchers who run R interactively and need a Python
   table without leaving the shell.
2. Quarto / RMarkdown users who want to embed a PySofra table via
   ``{{< include ... >}}`` without spinning up a Python kernel.
3. Shell-script automation pipelines (``make``, ``snakemake``,
   ``nextflow``) where a published table is the build artefact.

Usage
-----
::

    pysofra table mydata.csv \\
        --by arm \\
        --vars age,sex,bmi,outcome \\
        --add-p --add-smd \\
        --out table1.docx

    pysofra version
    pysofra check mydata.csv --by arm --vars age,sex,bmi

Supported input formats: ``.csv``, ``.tsv``, ``.parquet``, ``.xlsx``,
``.xls``, ``.json`` (line-delimited).

Supported output extensions: ``.html`` ``.md`` ``.tex`` ``.docx``
``.pptx`` ``.xlsx`` ``.png`` ``.typ`` ``.qmd`` (Quarto block).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__


def _read_table(path: Path) -> pandas.DataFrame:  # noqa: F821
    """Read a tabular file by extension; raise on unknown."""
    import pandas as pd
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path, lines=True)
    raise SystemExit(
        f"unrecognised input extension {suffix!r} (path: {path}). "
        f"Supported: .csv .tsv .parquet .xlsx .xls .json"
    )


def _split_vars(arg: str | None) -> list[str] | None:
    if arg is None:
        return None
    return [v.strip() for v in arg.split(",") if v.strip()]


def _write_table(table, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".html":
        path.write_text(table.to_html())
    elif suffix in (".md", ".markdown"):
        path.write_text(table.to_markdown())
    elif suffix == ".tex":
        path.write_text(table.to_latex())
    elif suffix == ".typ":
        table.to_typst_file(path)
    elif suffix == ".qmd":
        # default to HTML pass-through; users wanting LaTeX should pipe
        path.write_text(table.to_quarto(format="html"))
    elif suffix == ".docx":
        table.to_docx(path)
    elif suffix == ".pptx":
        table.to_pptx(path)
    elif suffix == ".xlsx":
        table.to_xlsx(path)
    elif suffix == ".png":
        table.to_image(path)
    else:
        raise SystemExit(
            f"unrecognised output extension {suffix!r} (path: {path}). "
            f"Supported: .html .md .tex .typ .qmd .docx .pptx .xlsx .png"
        )


def _cmd_table(args: argparse.Namespace) -> int:
    import pandas as pd  # noqa: F401  imported indirectly via _read_table

    from . import tbl_one
    df = _read_table(Path(args.input))
    variables = _split_vars(args.vars)
    labels = None
    if args.labels:
        labels = {}
        for kv in args.labels.split(","):
            if "=" not in kv:
                raise SystemExit(
                    f"--labels expects 'col=label,col=label,...'; "
                    f"got {kv!r}"
                )
            k, v = kv.split("=", 1)
            labels[k.strip()] = v.strip()
    table = tbl_one(
        df, by=args.by, variables=variables, labels=labels,
        missing=args.missing,
    )
    if args.add_p:
        table = table.add_p()
    if args.add_smd:
        table = table.add_smd()
    if args.add_n:
        table = table.add_n()
    if args.add_overall:
        table = table.add_overall()
    if args.add_q:
        table = table.add_q(method=args.q_method)
    if args.check_safety:
        table = table.with_safety_warnings()

    out_path = Path(args.out) if args.out else None
    if out_path is None:
        # No --out: print Markdown to stdout
        sys.stdout.write(table.to_markdown())
        return 0
    _write_table(table, out_path)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    """Build the table and *only* print safety warnings."""
    from . import tbl_one
    df = _read_table(Path(args.input))
    table = tbl_one(
        df, by=args.by, variables=_split_vars(args.vars),
        missing=args.missing,
    )
    if args.add_p:
        table = table.add_p()
    if args.add_smd:
        table = table.add_smd()
    warns = table.check_safety()
    if not warns:
        print("OK — no publication-safety flags.")
        return 0
    print(f"FAIL — {len(warns)} publication-safety flag(s):", file=sys.stderr)
    for w in warns:
        print(f"  [{w.code}] {w.row_label}: {w.message}", file=sys.stderr)
    return 2


def _cmd_version(_args: argparse.Namespace) -> int:
    print(__version__)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pysofra",
        description=(
            "PySofra command-line interface — build publication-quality "
            "summary tables in one shot."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"pysofra {__version__}",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND",
                                 required=True)

    # ---- table -----------------------------------------------------
    p_t = sub.add_parser(
        "table",
        help="Build a Table 1 / tbl_one and write to a file.",
    )
    p_t.add_argument("input", help="Input file (.csv .tsv .parquet "
                                    ".xlsx .xls .json)")
    p_t.add_argument("--by", default=None,
                     help="Stratification column (e.g. treatment arm).")
    p_t.add_argument("--vars", default=None,
                     help="Comma-separated list of variables to include. "
                          "Default: all columns except --by.")
    p_t.add_argument("--labels", default=None,
                     help="Comma-separated 'col=label' pairs.")
    p_t.add_argument("--missing", default="ifany",
                     choices=("never", "ifany", "always"),
                     help="Missing-row policy (default ifany).")
    p_t.add_argument("--add-p", action="store_true",
                     help="Append a p-value column.")
    p_t.add_argument("--add-smd", action="store_true",
                     help="Append a standardized mean-difference column.")
    p_t.add_argument("--add-n", action="store_true",
                     help="Append a per-variable N column.")
    p_t.add_argument("--add-overall", action="store_true",
                     help="Prepend an unstratified 'Overall' column.")
    p_t.add_argument("--add-q", action="store_true",
                     help="Append a multiplicity-adjusted q-value column.")
    p_t.add_argument("--q-method", default="fdr_bh",
                     help="--add-q adjustment method (default fdr_bh).")
    p_t.add_argument("--check-safety", action="store_true",
                     help="Run check_safety() and attach warnings as "
                          "footnotes.")
    p_t.add_argument("--out", "-o", default=None,
                     help="Output file. Extension picks the backend "
                          "(.html .md .tex .typ .qmd .docx .pptx .xlsx "
                          ".png). Omit to print Markdown to stdout.")
    p_t.set_defaults(func=_cmd_table)

    # ---- check -----------------------------------------------------
    p_c = sub.add_parser(
        "check",
        help="Build a table and print publication-safety warnings; "
             "exit 0 if clean, 2 if any flag fires.",
    )
    p_c.add_argument("input", help="Input file.")
    p_c.add_argument("--by", default=None)
    p_c.add_argument("--vars", default=None)
    p_c.add_argument("--missing", default="ifany",
                     choices=("never", "ifany", "always"))
    p_c.add_argument("--add-p", action="store_true")
    p_c.add_argument("--add-smd", action="store_true")
    p_c.set_defaults(func=_cmd_check)

    # ---- version ---------------------------------------------------
    p_v = sub.add_parser("version", help="Print PySofra version.")
    p_v.set_defaults(func=_cmd_version)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
