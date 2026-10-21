"""Command line interface: ``coco-archive convert`` and ``check``.

Usage::

    coco-archive convert SOURCE TARGET [--from FORMAT] [--to FORMAT] [--algorithm NAME]
    coco-archive check SOURCE...

``convert`` reads a COCO archive (a directory, ``.tgz``, ``.tar.gz`` or
``.zip``) or a Parquet file written by `Archive.write_parquet`, and writes it
as a Parquet file with `Archive.write_parquet`, as a cleaned-up COCO archive in
a ZIP file with `Archive.write_archive`, or into a new or empty directory with
`Archive.write_directory`. The formats are inferred from the file names unless
given with ``--from`` and ``--to``. ``check`` reads archives or Parquet files
and lists the issues found while reading; it exits with status 1 if there are
any. Sources can be local or ``http://`` or ``https://`` URLs.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.parse
import zipfile
from pathlib import Path

import polars as pl

from . import __version__
from .archive import Archive

_INPUT_FORMATS = ("archive", "parquet")
_OUTPUT_FORMATS = ("parquet", "zip", "directory")


def _is_url(path: str) -> bool:
    return path.startswith(("http://", "https://"))


def _input_format(path: str) -> str:
    """Infer the format of `path`: a Parquet file by its extension, else an archive."""
    if _is_url(path):
        return "parquet" if urllib.parse.urlsplit(path).path.endswith(".parquet") else "archive"
    return "parquet" if path.endswith(".parquet") and Path(path).is_file() else "archive"


def _output_format(path: str) -> str:
    """Infer the format to write to `path` from its extension, or a directory if it has none."""
    if path.endswith(".parquet"):
        return "parquet"
    if path.endswith(".zip"):
        return "zip"
    if path.endswith((".tgz", ".tar.gz")):
        raise ValueError(f"{path}: writing tar files is not supported, use a .zip file or a directory")
    if Path(path).is_dir() or path.endswith(("/", "\\")) or not Path(path).suffix:
        return "directory"
    raise ValueError(f"{path}: unknown output format, use --to {{{','.join(_OUTPUT_FORMATS)}}}")


def _read(path: str, format: str | None = None, algorithm: str | None = None) -> Archive:
    """Read a Parquet file written by `Archive.write_parquet` or a COCO archive, from a path or URL."""
    if (format or _input_format(path)) == "parquet":
        archive = Archive.read_parquet(path)
        if algorithm is not None:
            archive.algorithm = algorithm
        return archive
    return Archive.read_archive(path, algorithm=algorithm)


def _convert(args: argparse.Namespace) -> int:
    format = args.to or _output_format(args.target)
    archive = _read(args.source, args.from_, args.algorithm)
    if format == "parquet":
        archive.write_parquet(args.target)
    elif format == "zip":
        archive.write_archive(args.target)
    else:
        archive.write_directory(args.target)
    print(f"{args.target}: {archive!r}")
    if archive.issues:
        print(f"{len(archive.issues)} issues, run 'coco-archive check {args.source}' for details", file=sys.stderr)
    return 0


def _check(args: argparse.Namespace) -> int:
    status = 0
    for path in args.sources:
        archive = _read(path)
        print(f"{path}: {archive!r}")
        for issue in archive.issues:
            print(f"  {issue}")
        if archive.issues:
            status = 1
    return status


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface with `argv` (default: ``sys.argv[1:]``) and return the exit status."""
    parser = argparse.ArgumentParser(prog="coco-archive", description="Convert and check COCO result archives.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    convert = commands.add_parser(
        "convert",
        help="convert between COCO archives and Parquet files",
        description="Read SOURCE and write it to TARGET. Formats are inferred from the names: a SOURCE ending in "
        ".parquet is a Parquet file, anything else a COCO archive (directory, .tgz, .tar.gz or .zip). A TARGET "
        "ending in .parquet is written as a Parquet file, one ending in .zip as a cleaned-up COCO archive, and an "
        "existing directory or a name without extension as a cleaned-up COCO archive directory.",
    )
    convert.add_argument("source", help="archive directory, .tgz, .tar.gz or .zip file, or Parquet file, or URL")
    convert.add_argument("target", help="file to write, overwritten if it exists, or directory, must be empty")
    convert.add_argument("--from", dest="from_", choices=_INPUT_FORMATS, help="format of SOURCE")
    convert.add_argument("--to", choices=_OUTPUT_FORMATS, help="format of TARGET")
    convert.add_argument("--algorithm", help="algorithm name (default: kept, or name of SOURCE without extension)")
    convert.set_defaults(run=_convert)

    check = commands.add_parser("check", help="list the issues found while reading archives")
    check.add_argument("sources", nargs="+", help="archive directories, archive files or Parquet files, or their URLs")
    check.set_defaults(run=_check)

    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except (OSError, ValueError, KeyError, tarfile.TarError, zipfile.BadZipFile, pl.exceptions.PolarsError) as e:
        parser.exit(2, f"coco-archive: error: {e}\n")


if __name__ == "__main__":
    sys.exit(main())
