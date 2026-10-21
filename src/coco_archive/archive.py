"""Implementation of `Archive` and `Run`, see the package documentation for an overview."""

from __future__ import annotations

import io
import json
import posixpath
import re
import tarfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import IO, Any

import polars as pl

REASON = pl.Enum(["target", "evals", "recommendation", "final"])
"""Data type of the ``reason`` column of `Run.data`.

``target``
    Row from the ``.dat`` file, the best value reached a new target.
``evals``
    Row from the ``.tdat`` file, the evaluation count reached a new budget.
``recommendation``
    Row from the ``.mdat`` file, the algorithm recommended a solution.
``final``
    Last row of the ``.tdat`` file, the last evaluation of the run.

The categories are ordered, so sorting by ``reason`` puts ``target`` before
``evals`` before ``recommendation`` before ``final``.
"""

_F64, _I64 = pl.Float64, pl.Int64

# Column layouts: (layout, column-header prefix, base columns). Order matters, first match wins.
_LAYOUTS: list[tuple[str, str, list[tuple[str, Any]]]] = [
    (
        "new2",
        "f evaluations | g evaluations | best noise-free fitness - Fopt (",
        [("evals", _I64), ("g_evals", _I64), ("best_f", _F64), ("f_measured", _F64), ("best_f_measured", _F64)],
    ),
    (
        "new",
        "f evaluations | g evaluations | best noise-free fitness - Fopt |",
        [
            ("evals", _I64),
            ("g_evals", _I64),
            ("best_f", _F64),
            ("f", _F64),
            ("f_measured", _F64),
            ("best_f_measured", _F64),
        ],
    ),
    (
        "old",
        "function evaluation | noise-free fitness - Fopt",
        [("evals", _I64), ("f", _F64), ("best_f", _F64), ("f_measured", _F64), ("best_f_measured", _F64)],
    ),
    (
        "biobj",
        "function evaluation | indicator value | target hit",
        [("evals", _I64), ("indicator", _F64), ("target_hit", _F64)],
    ),
    ("biobj", "function evaluation | indicator value", [("evals", _I64), ("indicator", _F64)]),
]

_HEADER_KV = re.compile(r"""\s*([A-Za-z0-9_]+)\s*=\s*('[^']*'|"[^"]*"|[^,]*?)\s*(?:,|$)""")
_PREAMBLE_KV = re.compile(r"\s*([^,=]+?)\s*=\s*([^,]+?)\s*(?:,|$)")
_RUN = re.compile(r"(\d+)(?::(\d+)\|(\S+))?")
_FOPT = re.compile(r"\(([^)]*)\)")


@dataclass
class Run:
    """One run (trial) of the algorithm on one problem instance.

    Runs are created by `Archive.read_archive` and `Archive.read_parquet`;
    there is usually no need to create them directly.

    Attributes
    ----------
    suite : str
        Name of the benchmark suite, e.g. ``"bbob"`` or ``"bbob-biobj"``. Old
        archives don't name the suite. It is then ``"bbob-biobj"`` for
        bi-objective data, ``"bbob-noisy"`` for function numbers above 100 and
        ``"bbob"`` otherwise. The old name ``"sbox-cost"`` is reported as
        ``"bbob-boxed"``.
    function : int
        Function number within the suite.
    dimension : int
        Dimension of the search space.
    instance : int
        Instance number of the problem. It need not be unique within an
        archive.
    finished : bool
        False if the logger wasn't finalized properly ("ill-finalized run"):
        the index file lists the run without its final evaluation count and
        best value. Its data is complete up to the last logged row.
    layout : str
        Column layout of the data files: ``"old"``, ``"new"``, ``"new2"`` or
        ``"biobj"``. It determines the columns of `data`, see the package
        documentation.
    fopt : float or None
        Optimal f-value of the problem instance as given in the data files,
        None for bi-objective data. The ``best_f`` and ``f`` columns are
        relative to it, ``f_measured`` and ``best_f_measured`` are not.
    reference_value : float or None
        Reference hypervolume of the instance (bi-objective data only), to
        which the ``indicator`` column is relative.
    info_file : str
        Path of the index file within the archive that lists the run.
    info_line : int
        Line number (1-based) of the run's entry in `info_file`.
    position : int
        Position (0-based) of the run on that line. Together with
        `info_file` and `info_line` it identifies where the run came from,
        e.g. to find duplicates.
    metadata : dict
        Key-value pairs of the header line of the index file entry, e.g.
        ``{"funcId": 17, "DIM": 40, "Precision": 1e-08, "algId": "BFGS"}``.
        Numbers are converted to int or float, quotes are removed from
        strings, and unknown keys are kept.
    data : polars.DataFrame
        The rows of the run's ``.dat``, ``.tdat`` and ``.mdat`` files, merged
        and sorted by ``evals``, with a ``reason`` column of type `REASON`.
    comment : str
        Comment line of the index file entry without the leading ``%``,
        usually a description of the algorithm.

    Examples
    --------
    >>> url = "https://raw.githubusercontent.com/numbbo/data-archive/gh-pages/data-archive/bbob/2009/BFGS_ros_noiseless.tgz"
    >>> run = Archive.read_archive(url).runs[0]
    >>> run.suite, run.function, run.dimension, run.instance, run.finished
    ('bbob', 17, 40, 1, True)
    >>> run.data.filter(pl.col("reason") == "final").select("evals", "best_f")
    shape: (1, 2)
    ┌─────────┬──────────┐
    │ evals   ┆ best_f   │
    │ ---     ┆ ---      │
    │ i64     ┆ f64      │
    ╞═════════╪══════════╡
    │ 1301135 ┆ 6.814104 │
    └─────────┴──────────┘
    """

    suite: str
    function: int
    dimension: int
    instance: int
    finished: bool
    layout: str
    fopt: float | None
    reference_value: float | None
    info_file: str
    info_line: int
    position: int
    metadata: dict[str, Any]
    data: pl.DataFrame = field(repr=False)
    comment: str = ""


@dataclass
class Archive:
    """All runs of one algorithm, read from a COCO result archive or a Parquet file.

    Use `read_archive` to read a COCO archive and `read_parquet` to read
    a file written by `write_parquet`.

    Attributes
    ----------
    path : pathlib.Path, str or None
        The file the archive was originally read from, its URL as a str, or
        None if it was read from a file object without a name. `read_parquet`
        keeps the path of the COCO archive that the Parquet file was made
        from.
    algorithm : str
        Name of the algorithm. It defaults to the archive file name without
        extension. The ``algId`` or ``algorithm`` value in the index files is
        not used for this, because experimenters often leave it at a
        placeholder such as ``'ALG'``; it remains available in `Run.metadata`.
    runs : list of Run
        All runs in the order they were read: index files sorted by path, then
        the order within each file. Runs of one problem can be spread over
        several entries and index files (batches, threads, one directory per
        instance); they are all collected here. Duplicate runs are kept.
    issues : list of str
        Defects found while reading, e.g. missing data files, malformed index
        file lines, or index files listing a different number of runs than
        the data files contain. Each message names the file and line. The
        affected lines or runs are skipped, since their runs can't be matched
        to their data reliably.

    Examples
    --------
    >>> url = "https://raw.githubusercontent.com/numbbo/data-archive/gh-pages/data-archive/bbob/2009/BFGS_ros_noiseless.tgz"
    >>> archive = Archive.read_archive(url)
    >>> archive
    Archive('BFGS_ros_noiseless', 2160 runs, 0 issues)
    >>> archive.runs_frame().group_by("dimension").len().sort("dimension")
    shape: (6, 2)
    ┌───────────┬─────┐
    │ dimension ┆ len │
    │ ---       ┆ --- │
    │ i64       ┆ u32 │
    ╞═══════════╪═════╡
    │ 2         ┆ 360 │
    │ 3         ┆ 360 │
    │ 5         ┆ 360 │
    │ 10        ┆ 360 │
    │ 20        ┆ 360 │
    │ 40        ┆ 360 │
    └───────────┴─────┘
    """

    path: Path | str | None
    algorithm: str
    runs: list[Run]
    issues: list[str]

    def runs_frame(self) -> pl.DataFrame:
        """Return a table of the runs, with their metadata but without their data.

        Returns
        -------
        polars.DataFrame
            One row per run with the columns ``run`` (index into `runs`),
            ``suite``, ``function``, ``dimension``, ``instance``,
            ``finished``, ``layout``, ``fopt``, ``reference_value``,
            ``info_file``, ``info_line``, ``position`` (see `Run`) and
            ``rows`` (number of rows of `Run.data`). The column types are the
            same for every archive, even one without runs.

        Examples
        --------
        >>> url = "https://raw.githubusercontent.com/numbbo/data-archive/gh-pages/data-archive/bbob/2009/BFGS_ros_noiseless.tgz"
        >>> archive = Archive.read_archive(url)

        Number of runs per dimension and function:

        >>> runs = archive.runs_frame().sort("dimension", "function")
        >>> runs.pivot("dimension", index="function", values="run", aggregate_function="len")
        shape: (24, 7)
        ┌──────────┬─────┬─────┬─────┬─────┬─────┬─────┐
        │ function ┆ 2   ┆ 3   ┆ 5   ┆ 10  ┆ 20  ┆ 40  │
        │ ---      ┆ --- ┆ --- ┆ --- ┆ --- ┆ --- ┆ --- │
        │ i64      ┆ u32 ┆ u32 ┆ u32 ┆ u32 ┆ u32 ┆ u32 │
        ╞══════════╪═════╪═════╪═════╪═════╪═════╪═════╡
        │ 1        ┆ 15  ┆ 15  ┆ 15  ┆ 15  ┆ 15  ┆ 15  │
        │ 2        ┆ 15  ┆ 15  ┆ 15  ┆ 15  ┆ 15  ┆ 15  │
        │ 3        ┆ 15  ┆ 15  ┆ 15  ┆ 15  ┆ 15  ┆ 15  │
        │ 4        ┆ 15  ┆ 15  ┆ 15  ┆ 15  ┆ 15  ┆ 15  │
        │ 5        ┆ 15  ┆ 15  ┆ 15  ┆ 15  ┆ 15  ┆ 15  │
        │ …        ┆ …   ┆ …   ┆ …   ┆ …   ┆ …   ┆ …   │
        │ 20       ┆ 15  ┆ 15  ┆ 15  ┆ 15  ┆ 15  ┆ 15  │
        │ 21       ┆ 15  ┆ 15  ┆ 15  ┆ 15  ┆ 15  ┆ 15  │
        │ 22       ┆ 15  ┆ 15  ┆ 15  ┆ 15  ┆ 15  ┆ 15  │
        │ 23       ┆ 15  ┆ 15  ┆ 15  ┆ 15  ┆ 15  ┆ 15  │
        │ 24       ┆ 15  ┆ 15  ┆ 15  ┆ 15  ┆ 15  ┆ 15  │
        └──────────┴─────┴─────┴─────┴─────┴─────┴─────┘

        Number of unfinished runs:

        >>> archive.runs_frame().filter(~pl.col("finished")).height
        0
        """
        schema = {
            "suite": pl.String,
            "function": _I64,
            "dimension": _I64,
            "instance": _I64,
            "finished": pl.Boolean,
            "layout": pl.String,
            "fopt": _F64,
            "reference_value": _F64,
            "info_file": pl.String,
            "info_line": _I64,
            "position": _I64,
        }
        return pl.DataFrame(
            [{n: getattr(r, n) for n in schema} | {"rows": r.data.height} for r in self.runs],
            schema=schema | {"rows": _I64},
        ).with_row_index("run")

    @classmethod
    def read_archive(cls, path: str | Path | IO[bytes], algorithm: str | None = None) -> Archive:
        r"""Read a COCO result archive.

        Parameters
        ----------
        path : str, pathlib.Path or binary file object
            A ``.tgz``, ``.tar.gz`` or ``.zip`` archive as published in the
            COCO data archive or written by `write_archive`, or a directory
            with the extracted contents of such an archive, e.g. one written
            by `write_directory`. The archive file can be local, an
            ``http://`` or ``https://`` URL, or an open binary file. A URL is
            downloaded into memory first.
        algorithm : str, optional
            Name of the algorithm, stored as `algorithm`. Defaults to the file
            name of `path` without extension, or the directory name.

        Returns
        -------
        Archive

        Raises
        ------
        ValueError
            If `algorithm` is not given and `path` is a file object without a
            name.

        Notes
        -----
        The archive is read in memory, nothing is extracted to disk. Members
        of a ZIP file and files in a directory are read when needed. A tar file can only be read
        sequentially, so its ``.info``, ``.dat``, ``.tdat`` and ``.mdat``
        members are read in one pass.

        Every ``.info`` file in the archive is read. Its runs are matched by
        position to the run blocks of the data files it references, because
        instance numbers need not be unique. All variants that occur in
        published archives are handled:

        - UTF-8 or Latin-1 text with LF or CRLF line endings;
        - data file paths with ``\`` as separator, and paths relative to a
          ``folder`` given in the header (bi-objective data);
        - the column layouts of all logger versions, with or without ``x``;
        - bi-objective entries with one line per (function, dimension);
        - runs listed without final evaluation count and best value
          (`Run.finished` is False);
        - data files with or without a ``.mdat`` file of recommendations.
          Most archives have none; their runs have no ``recommendation``
          rows.

        Problems that make a line or run impossible to read reliably are
        recorded in `issues`, and the affected runs are skipped. Reading a
        large archive takes several seconds; use `write_parquet` to convert
        it once.

        Examples
        --------
        >>> url = "https://raw.githubusercontent.com/numbbo/data-archive/gh-pages/data-archive/bbob/2009/BFGS_ros_noiseless.tgz"
        >>> archive = Archive.read_archive(url)
        >>> archive
        Archive('BFGS_ros_noiseless', 2160 runs, 0 issues)

        Name the algorithm after its path in the data archive:

        >>> archive = Archive.read_archive(url, algorithm="bbob/2009/BFGS_ros_noiseless")

        Read a local copy or an open file:

        >>> from urllib.request import urlretrieve
        >>> _ = urlretrieve(url, "BFGS_ros_noiseless.tgz")
        >>> archive = Archive.read_archive("BFGS_ros_noiseless.tgz")
        >>> with open("BFGS_ros_noiseless.tgz", "rb") as f:
        ...     archive = Archive.read_archive(f)
        """
        source: Path | IO[bytes]
        if isinstance(path, str) and path.startswith(("http://", "https://")):
            with urllib.request.urlopen(path) as response:
                source = io.BytesIO(response.read())
            name = posixpath.basename(urllib.parse.urlsplit(path).path)
        elif hasattr(path, "read"):
            source = path
            name = getattr(path, "name", None)
            path = Path(name) if isinstance(name, str) else None
            name = path.name if path else None
        else:
            source = path = Path(path)
            name = path.resolve().name
        if algorithm is None:
            if not name:
                raise ValueError("algorithm must be given for a file object without a name")
            algorithm = re.sub(r"\.(tgz|tar\.gz|zip)$", "", name)
        issues: list[str] = []
        runs: list[Run] = []
        with _Members(source) as members:
            for name in sorted(members.names):
                if posixpath.basename(name).endswith(".info"):
                    runs.extend(_read_info(members, name, issues))
        return cls(path, algorithm, runs, issues)

    def write_parquet(self, path: str | Path) -> None:
        """Write the archive to a single zstd-compressed Parquet file.

        Parameters
        ----------
        path : str or pathlib.Path
            The file to write. An existing file is overwritten.

        See Also
        --------
        read_parquet : Read the file back into an `Archive`.

        Notes
        -----
        The file holds one table with the rows of all runs. Its columns are

        - ``algorithm``, the same in every row;
        - the data columns of all runs (see the package documentation). A
          column a run doesn't have is null in its rows;
        - ``run``, the index of the run in `runs`;
        - ``suite``, ``function``, ``dimension``, ``instance``, ``finished``,
          ``layout``, ``fopt`` and ``reference_value`` of the run.

        The rows are sorted by function, dimension and run, so filtering on
        these columns reads only part of the file. Because of the
        ``algorithm`` column, the files of several archives can be scanned as
        one table. ``x`` is stored as float32, which is exact for the 5
        significant digits COCO logs.

        Everything else needed to restore the `Archive` exactly is stored once
        in the file's key-value metadata: `path`, `algorithm`, `issues`, the
        remaining run attributes, each run's set of columns, and runs without
        rows.

        If pyarrow is installed, it is used to write floats with
        ``BYTE_STREAM_SPLIT`` and evaluation counts with
        ``DELTA_BINARY_PACKED`` encoding, which polars can't write. This makes
        the file 2-15 % smaller. Without pyarrow the file is written by
        polars. Both kinds of file have the same columns and are read the same
        way.

        Examples
        --------
        >>> url = "https://raw.githubusercontent.com/numbbo/data-archive/gh-pages/data-archive/bbob/2009/BFGS_ros_noiseless.tgz"
        >>> archive = Archive.read_archive(url)
        >>> archive.write_parquet("BFGS_ros_noiseless.parquet")

        The file can be queried with polars directly, e.g. the final best
        values on function 1:

        >>> pl.scan_parquet("BFGS_ros_noiseless.parquet").filter(
        ...     pl.col("function") == 1, pl.col("reason") == "final"
        ... ).select("dimension", "instance", "evals", "best_f").collect()
        shape: (90, 4)
        ┌───────────┬──────────┬───────┬────────────┐
        │ dimension ┆ instance ┆ evals ┆ best_f     │
        │ ---       ┆ ---      ┆ ---   ┆ ---        │
        │ i64       ┆ i64      ┆ i64   ┆ f64        │
        ╞═══════════╪══════════╪═══════╪════════════╡
        │ 2         ┆ 1        ┆ 9     ┆ 4.4054e-13 │
        │ 2         ┆ 2        ┆ 9     ┆ 6.8212e-13 │
        │ 2         ┆ 3        ┆ 9     ┆ 2.0464e-12 │
        │ 2         ┆ 4        ┆ 9     ┆ 1.9327e-12 │
        │ 2         ┆ 5        ┆ 9     ┆ 6.7502e-14 │
        │ …         ┆ …        ┆ …     ┆ …          │
        │ 40        ┆ 4        ┆ 123   ┆ 6.8620e-9  │
        │ 40        ┆ 5        ┆ 123   ┆ 3.6859e-10 │
        │ 40        ┆ 1        ┆ 123   ┆ 6.2940e-9  │
        │ 40        ┆ 2        ┆ 164   ┆ 1.3927e-11 │
        │ 40        ┆ 3        ┆ 123   ┆ 6.5210e-9  │
        └───────────┴──────────┴───────┴────────────┘
        """
        runs = self.runs_frame().drop("rows", "info_file", "info_line", "position")  # provenance stays in the metadata
        if self.runs:
            table = pl.concat(
                [r.data.with_columns(pl.lit(i, pl.UInt32).alias("run")) for i, r in enumerate(self.runs)],
                how="diagonal_relaxed",
            )
            table = table.join(runs, on="run", how="left").sort("function", "dimension", "run", maintain_order=True)
            if "x" in table.columns:
                table = table.with_columns(pl.col("x").cast(pl.List(pl.Float32)))
        else:
            table = pl.DataFrame(schema={"run": pl.UInt32})
        table = table.select(pl.lit(self.algorithm, pl.String).alias("algorithm"), pl.all())
        # run fields column-wise; fields with few distinct values as indices into "values"
        fields = {n: [getattr(r, n) for r in self.runs] for n in _RUN_FIELDS}
        fields["columns"] = [r.data.columns for r in self.runs]
        values = {}
        for n in _SHARED_FIELDS:
            index: dict[str, int] = {}
            fields[n] = [index.setdefault(json.dumps(v, sort_keys=True), len(index)) for v in fields[n]]
            values[n] = [json.loads(v) for v in index]
        meta = {
            "version": _PARQUET_VERSION,
            "path": None if self.path is None else str(self.path),
            "algorithm": self.algorithm,
            "issues": self.issues,
            "values": values,
            "runs": fields,
        }
        metadata = {_PARQUET_KEY: json.dumps(meta)}
        try:
            import pyarrow.parquet as pq
        except ImportError:
            table.write_parquet(path, compression="zstd", row_group_size=100_000, metadata=metadata)
            return
        floats = [n for n, t in table.schema.items() if t in (_F64, pl.Float32)]
        if "x" in table.columns:
            floats.append("x.list.element")
        deltas = [n for n in ("evals", "g_evals") if n in table.columns]
        arrow = table.to_arrow()
        pq.write_table(
            arrow.replace_schema_metadata(metadata),
            path,
            compression="zstd",
            compression_level=9,
            row_group_size=100_000,
            use_dictionary=[n for n in table.columns if n not in floats + deltas + ["x"]],
            use_byte_stream_split=floats,
            column_encoding=dict.fromkeys(deltas, "DELTA_BINARY_PACKED"),
        )

    @classmethod
    def read_parquet(cls, path: str | Path | IO[bytes]) -> Archive:
        """Read an archive from a Parquet file written by `write_parquet`.

        Parameters
        ----------
        path : str, pathlib.Path or binary file object
            A local file, an ``http://`` or ``https://`` URL, or an open binary
            file. A URL is downloaded into memory first, since all rows are
            needed anyway.

        Returns
        -------
        Archive
            The archive as it was written. ``x`` is converted back to float64.

        Raises
        ------
        KeyError
            If the file wasn't written by `write_parquet`.
        ValueError
            If the file was written by an incompatible version of this package.

        See Also
        --------
        write_parquet : Write the file.

        Examples
        --------
        >>> url = "https://raw.githubusercontent.com/numbbo/data-archive/gh-pages/data-archive/bbob/2009/BFGS_ros_noiseless.tgz"
        >>> Archive.read_archive(url).write_parquet("BFGS_ros_noiseless.parquet")
        >>> archive = Archive.read_parquet("BFGS_ros_noiseless.parquet")
        >>> archive = Archive.read_parquet("https://example.org/BFGS_ros_noiseless.parquet")  # doctest: +SKIP
        >>> with open("BFGS_ros_noiseless.parquet", "rb") as f:
        ...     archive = Archive.read_parquet(f)
        """
        if isinstance(path, str) and path.startswith(("http://", "https://")):
            with urllib.request.urlopen(path) as response:
                path = io.BytesIO(response.read())
        meta = json.loads(pl.read_parquet_metadata(path)[_PARQUET_KEY])
        if meta["version"] not in _PARQUET_READABLE:
            raise ValueError(f"unsupported coco_archive Parquet version {meta['version']!r}")
        if hasattr(path, "seek"):
            path.seek(0)
        table = pl.read_parquet(path)
        parts = table.partition_by("run", as_dict=True, maintain_order=True)
        fields = meta["runs"]
        for n, values in meta["values"].items():
            fields[n] = [values[v] for v in fields[n]]
        runs = []
        for i, columns in enumerate(fields.pop("columns")):
            info = {n: v[i] for n, v in fields.items()}
            part = parts.get((i,))
            if part is None:
                part = pl.DataFrame(schema={c: table.schema.get(c, _EMPTY_DTYPES.get(c, _F64)) for c in columns})
            data = part.select(columns).with_columns(pl.col("reason").cast(REASON))
            if "x" in columns:
                data = data.with_columns(pl.col("x").cast(pl.List(_F64)))
            runs.append(Run(**info, data=data))
        path = meta["path"]
        if path is not None and not path.startswith(("http://", "https://")):
            path = Path(path)
        return cls(path, meta["algorithm"], runs, meta["issues"])

    def write_archive(self, path: str | Path) -> None:
        """Write the archive as a cleaned-up COCO result archive in a ZIP file.

        The result can be read by cocopp and by `read_archive`, and gives
        the same data as the archive it was made from.

        Parameters
        ----------
        path : str or pathlib.Path
            The ZIP file to write. An existing file is overwritten. Its name
            without extension is used as the name of the index file.

        See Also
        --------
        read_archive : Read the archive back.

        Notes
        -----
        The ZIP file contains

        - a single index file ``<name>.info``. It has one entry per (suite,
          function, dimension) for single-objective data, and one entry per
          suite with one line per (function, dimension) for bi-objective
          data. All runs of a problem are in one entry, even if they were
          spread over several entries or files before;
        - one pair of data files ``<suite>/f<FFF>_d<DDD>.dat`` and ``.tdat``
          per line of the index file, e.g. ``bbob/f001_d020.dat``, and a
          ``.mdat`` file if any of their runs has ``recommendation`` rows.

        Compared to published archives, the output is normalized:

        - every entry has a ``suite`` key. Archives without it store the final
          best value in the index file shifted by the target precision; here
          it is always the final Δf;
        - the final evaluation count and best value in the index file are
          taken from the ``final`` row of each run, the authoritative source;
        - `algorithm` is written as ``algId`` (``algorithm`` for bi-objective
          data);
        - the other header keys and the comment line of an entry are taken
          from its first run;
        - all files are UTF-8 with LF line endings, and paths use ``/``.

        Numbers are written in the shortest notation that reads back as the
        same value, ``x`` as float32. Runs skipped while reading (see
        `issues`) are not written, nor are ``.rdat`` and ``.cdat`` files.

        Examples
        --------
        >>> url = "https://raw.githubusercontent.com/numbbo/data-archive/gh-pages/data-archive/bbob/2009/BFGS_ros_noiseless.tgz"
        >>> archive = Archive.read_archive(url)
        >>> archive.write_archive("BFGS_ros_noiseless.zip")

        Write only some of the runs, here those up to dimension 5. The runs of
        several archives of the same algorithm can be combined the same way,
        e.g. with ``a.runs + b.runs``:

        >>> runs = [r for r in archive.runs if r.dimension <= 5]
        >>> Archive(archive.path, archive.algorithm, runs, archive.issues).write_archive("BFGS_ros_noiseless_d5.zip")
        >>> Archive.read_archive("BFGS_ros_noiseless_d5.zip")
        Archive('BFGS_ros_noiseless_d5', 1080 runs, 0 issues)
        """
        path = Path(path)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for name, text in self._archive_files(path.stem).items():
                z.writestr(name, text)

    def write_directory(self, path: str | Path) -> None:
        """Write the archive as a cleaned-up COCO result archive into a directory.

        The files are the same as those `write_archive` puts into the ZIP
        file.

        Parameters
        ----------
        path : str or pathlib.Path
            The directory to write. It is created if it doesn't exist, and
            must be empty if it does. Its name is used as the name of the
            index file.

        Raises
        ------
        FileExistsError
            If `path` exists and is not an empty directory.

        See Also
        --------
        write_archive : Write the files into a ZIP file instead.
        read_archive : Read the directory back.

        Examples
        --------
        >>> url = "https://raw.githubusercontent.com/numbbo/data-archive/gh-pages/data-archive/bbob/2009/BFGS_ros_noiseless.tgz"
        >>> Archive.read_archive(url).write_parquet("BFGS_ros_noiseless.parquet")
        >>> archive = Archive.read_parquet("BFGS_ros_noiseless.parquet")
        >>> archive.write_directory("BFGS_ros_noiseless")
        >>> Archive.read_archive("BFGS_ros_noiseless")
        Archive('BFGS_ros_noiseless', 2160 runs, 0 issues)
        """
        path = Path(path)
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(f"{path} exists and is not an empty directory")
        files = self._archive_files(path.resolve().name)
        path.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (path / name).parent.mkdir(parents=True, exist_ok=True)
            (path / name).write_text(text, encoding="utf-8", newline="\n")

    def _archive_files(self, name: str) -> dict[str, str]:
        """Contents of the files of the archive written by `write_archive`, keyed by path; the index file is `name`."""
        entries: dict[tuple, dict[str, list[int]]] = {}  # entry key -> file stem -> run indices
        for i, run in enumerate(self.runs):
            biobj = run.layout == "biobj"
            key = (run.suite, biobj) + (() if biobj else (run.function, run.dimension, run.layout))
            stem = f"{run.suite}/f{run.function:03d}_d{run.dimension:03d}"
            entries.setdefault(key, {}).setdefault(stem, []).append(i)
        stems = [stem for files in entries.values() for stem in files]
        lines, finals = _data_lines(self.runs)
        info, files = [], {}
        for key in sorted(entries, key=lambda k: (k[0], k[1], *k[2:])):
            entry = entries[key]
            first = self.runs[next(iter(entry.values()))[0]]
            info.append(_info_header(first, self.algorithm))
            info.append(f"% {first.comment}")
            for stem in sorted(entry):
                indices = entry[stem]
                if stems.count(stem) > 1:  # several single-objective layouts for one problem
                    stem = f"{stem}_{first.layout}"
                items = ", ".join(
                    f"{self.runs[i].instance}:{finals[i][0]}|{finals[i][1]:.1e}"
                    if self.runs[i].finished and i in finals
                    else str(self.runs[i].instance)
                    for i in indices
                )
                problem = self.runs[indices[0]]
                prefix = f"function = {problem.function}, dim = {problem.dimension}, " if key[1] else ""
                info.append(f"{prefix}{stem}.dat, {items}")
                kinds = ["dat", "tdat"] + ["mdat"] * any((i, "mdat") in lines for i in indices)
                for kind in kinds:
                    files[f"{stem}.{kind}"] = "".join(
                        _block_header(self.runs[i], kind, k) + lines.get((i, kind), "") for k, i in enumerate(indices)
                    )
        return {f"{name}.info": "\n".join(info) + "\n"} | files

    def __repr__(self) -> str:
        return f"Archive({self.algorithm!r}, {len(self.runs)} runs, {len(self.issues)} issues)"


_RUN_FIELDS = [f.name for f in fields(Run) if f.name != "data"]
_PARQUET_KEY = "coco_archive"
_PARQUET_VERSION = 4
_PARQUET_READABLE = (3, 4)  # version 3 is version 4 without recommendation rows
_SHARED_FIELDS = ("comment", "metadata", "columns", "info_file")
# dtypes of data columns that are not of type Float64, for runs whose columns don't occur in any row
_EMPTY_DTYPES = {"evals": _I64, "g_evals": _I64, "reason": REASON, "g_digits": pl.String, "x": pl.List(_F64)}


# data file of the rows of each reason
_KINDS = {"target": "dat", "evals": "tdat", "recommendation": "mdat", "final": "tdat"}

# Column-header lines written by `Archive.write_archive`, ``{fopt}`` is replaced by fopt as ``%.12e``
_COLUMN_HEADERS = {
    "old": "function evaluation | noise-free fitness - Fopt ({fopt}) | best noise-free fitness - Fopt | "
    "measured fitness | best measured fitness | x1 | x2...",
    "new": "f evaluations | g evaluations | best noise-free fitness - Fopt | noise-free fitness - Fopt ({fopt}) | "
    "measured fitness | best measured fitness | x1 | x2...",
    "new2": "f evaluations | g evaluations | best noise-free fitness - Fopt ({fopt}) + sum g_i+ | measured fitness | "
    "best measured fitness or single-digit g-values | x1 | x2...",
}
_DATA_FORMATS = {"new": "bbob-new", "new2": "bbob-new2"}
# header keys set by `_info_header` or meaningless after rewriting
_STRUCTURAL_KEYS = {"suite", "funcId", "DIM", "Precision", "algId", "algorithm", "indicator", "folder", "data_format"}


def _info_header(run: Run, algorithm: str) -> str:
    """Header line of the index file entry written for `run` and the runs grouped with it."""
    if run.layout == "biobj":
        header = {"suite": run.suite, "algorithm": algorithm, "indicator": run.metadata.get("indicator", "hyp")}
    else:
        precision = float(run.metadata.get("Precision", 1e-8))
        header = {"suite": run.suite, "funcId": run.function, "DIM": run.dimension, "Precision": f"{precision:.3e}"}
        header["algId"] = algorithm
    header |= {k: v for k, v in run.metadata.items() if k not in _STRUCTURAL_KEYS}
    if run.layout in _DATA_FORMATS:
        header["data_format"] = _DATA_FORMATS[run.layout]

    def value(key: str, v: Any) -> str:
        if not isinstance(v, str) or key == "Precision":
            return str(v)
        return f'"{v}"' if "'" in v and '"' not in v else "'{}'".format(v.replace("'", ""))

    return ", ".join(f"{k} = {value(k, v)}" for k, v in header.items())


def _block_header(run: Run, kind: str, index: int) -> str:
    """Preamble and column-header line of a run block in a ``.dat`` or ``.tdat`` file."""
    if run.layout != "biobj":
        return (
            "% " + _COLUMN_HEADERS[run.layout].format(fopt=f"{run.fopt:.12e}" if run.fopt is not None else "nan") + "\n"
        )
    name = f"{run.suite}_f{run.function:03d}_i{run.instance:02d}_d{run.dimension:02d}"
    instance = f"% instance = {run.instance}"
    if run.reference_value is not None:
        instance += f", reference value = {run.reference_value!r}"
    columns = "function evaluation | indicator value"
    if kind == "dat" and "target_hit" in run.data.columns:
        columns += " | target hit"
    return f"%\n% index = {index}, name = {name}\n{instance}\n% {columns}\n"


def _data_lines(runs: list[Run]) -> tuple[dict[tuple[int, str], str], dict[int, tuple[int, float]]]:
    """Format the data rows of all runs as text.

    Returns the lines keyed by (run index, ``dat``, ``tdat`` or ``mdat``), each ending in LF, and the evaluation
    count and best value (``best_f`` or ``indicator``) of the ``final`` row of each run.
    """
    groups: dict[tuple[str, ...], list[pl.DataFrame]] = {}
    for i, run in enumerate(runs):
        if run.data.height:
            columns = tuple(run.data.columns)
            groups.setdefault(columns, []).append(run.data.with_columns(pl.lit(i, pl.UInt32).alias("run")))
    lines: dict[tuple[int, str], str] = {}
    finals: dict[int, tuple[int, float]] = {}
    for columns, frames in groups.items():
        table = pl.concat(frames)
        value = "best_f" if "best_f" in columns else "indicator"
        final = table.filter(pl.col("reason") == "final").select("run", "evals", value)
        finals |= {run: (evals, best) for run, evals, best in final.iter_rows()}
        parts = [pl.col(c).cast(pl.String) for c in columns if c not in ("reason", "x")]
        if "x" in columns:
            parts.append(pl.col("x").list.eval(pl.element().cast(pl.Float32).cast(pl.String)).list.join(" "))
        text = (
            table.with_row_index("order")
            .select(
                "run",
                pl.col("reason").cast(pl.String).replace_strict(_KINDS, return_dtype=pl.String).alias("kind"),
                (pl.col("reason") == "final").alias("final"),  # the final row ends the .tdat block
                "order",
                (pl.concat_str(parts, separator=" ", ignore_nulls=True) + "\n").alias("line"),
            )
            .sort("run", "kind", "final", "order")
            .group_by("run", "kind", maintain_order=True)
            .agg(pl.col("line").str.join(""))
        )
        lines |= {(run, kind): line for run, kind, line in text.iter_rows()}
    return lines, finals


@dataclass
class _Block:
    header: str
    preamble: list[str]
    rows: list[list[str]]


class _Members:
    """Text access to archive members, keyed by normalised path. ZIP members and files are read on demand."""

    _EXTENSIONS = (".info", ".dat", ".tdat", ".mdat")

    def __init__(self, path: Path | IO[bytes]):
        self._zip = None
        self._tar_content: dict[str, bytes] = {}
        self._zip_names: dict[str, str] = {}
        self._dir: Path | None = None
        if isinstance(path, Path) and path.is_dir():
            self._dir = path
            self.names = {f.relative_to(path).as_posix() for f in path.rglob("*") if f.is_file()}
            return
        start = None if isinstance(path, Path) else path.tell()
        if zipfile.is_zipfile(path):
            self._zip = zipfile.ZipFile(path)
            for info in self._zip.infolist():
                if not info.is_dir():
                    self._zip_names[posixpath.normpath(info.filename)] = info.filename
            self.names = set(self._zip_names)
        else:
            if start is not None:
                path.seek(start)  # is_zipfile moved it
            # gzip streams have no random access, so read the relevant members in one pass
            file_object = start is not None
            with tarfile.open(None if file_object else path, "r:*", fileobj=path if file_object else None) as tar:
                for info in tar:
                    if info.isfile() and info.name.endswith(self._EXTENSIONS):
                        self._tar_content[posixpath.normpath(info.name)] = tar.extractfile(info).read()
            self.names = set(self._tar_content)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        if self._zip is not None:
            self._zip.close()

    def text(self, name: str) -> str | None:
        """Return the member as text (UTF-8 or Latin-1, LF line endings), or None if it doesn't exist."""
        if self._dir is not None:
            if name not in self.names:
                return None
            raw = (self._dir / name).read_bytes()
        elif self._zip is not None:
            if name not in self._zip_names:
                return None
            raw = self._zip.read(self._zip_names[name])
        else:
            raw = self._tar_content.get(name)
            if raw is None:
                return None
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        return text.replace("\r\n", "\n")


def _parse_value(value: str) -> Any:
    if value[:1] in "'\"":
        return value[1:-1]
    for convert in (int, float):
        try:
            return convert(value)
        except ValueError:
            pass
    return value


def _suite(header: dict[str, Any]) -> str:
    """Suite name from the header, inferred for old archives that don't name it."""
    suite = header.get("suite")
    if suite:
        return "bbob-boxed" if suite == "sbox-cost" else suite
    if "indicator" in header:
        return "bbob-biobj"
    return "bbob-noisy" if int(header.get("funcId", 0)) > 100 else "bbob"


def _read_info(members: _Members, info_name: str, issues: list[str]) -> list[Run]:
    """Parse one index file and the data files it references."""
    runs: list[Run] = []
    header: dict[str, Any] | None = None
    comment = ""
    for lineno, line in enumerate(members.text(info_name).split("\n"), start=1):
        stripped = line.strip()
        if stripped.startswith("%"):
            comment = stripped[1:].strip()
        if not stripped or stripped.startswith("%"):
            continue
        where = f"{info_name}:{lineno}"
        fields = [t.strip() for t in stripped.split(",")]
        if not any(t.endswith("dat") for t in fields):
            pairs = _HEADER_KV.findall(stripped)
            if not pairs:
                issues.append(f"{where}: malformed line skipped")
                continue
            header = {k: _parse_value(v) for k, v in pairs}
            comment = ""
            continue
        if header is None:
            issues.append(f"{where}: data line without header skipped")
            continue
        runs.extend(_read_data_line(members, info_name, where, lineno, header, comment, fields, issues))
    return runs


def _read_data_line(
    members: _Members,
    info_name: str,
    where: str,
    lineno: int,
    header: dict[str, Any],
    comment: str,
    fields: list[str],
    issues: list[str],
) -> list[Run]:
    """Parse one data line of an index file and pair its runs with the run blocks of its data files."""
    refs, prefix, items = [], {}, []
    for token in fields:
        if token.endswith("dat"):
            refs.append(token)
        elif "=" in token:
            key, value = token.split("=", 1)
            prefix[key.strip()] = _parse_value(value.strip())
        elif m := _RUN.fullmatch(token):
            items.append((int(m[1]), m[2] is not None))
        else:
            issues.append(f"{where}: malformed run {token!r}, line skipped")
            return []
    if not items:
        return []

    blocks: dict[str, list[_Block | None]] = {".dat": [], ".tdat": [], ".mdat": []}  # None: no .mdat file
    folder = str(header.get("folder", ""))
    mdat_files = 0
    for ref in refs:
        stem = posixpath.splitext(
            posixpath.normpath(posixpath.join(posixpath.dirname(info_name), folder, ref.replace("\\", "/")))
        )[0]
        for ext, ext_blocks in blocks.items():
            text = members.text(stem + ext)
            if text is None and ext == ".mdat":  # optional, most loggers don't write it
                continue
            if text is None:
                issues.append(f"{where}: missing {stem + ext}, line skipped")
                return []
            mdat_files += ext == ".mdat"
            ext_blocks.extend(_parse_blocks(text))
    if not mdat_files:
        blocks[".mdat"] = [None] * len(items)
    elif mdat_files != len(refs):
        issues.append(f"{where}: .mdat files for only {mdat_files} of {len(refs)} data files, line skipped")
        return []
    for ext, ext_blocks in blocks.items():
        if len(ext_blocks) != len(items):
            issues.append(f"{where}: {len(ext_blocks)} run blocks in {ext} for {len(items)} runs, line skipped")
            return []

    suite = _suite(header)
    biobj = "indicator" in header
    function = int(prefix["function"] if biobj else header["funcId"])
    dimension = int(prefix["dim"] if biobj else header["DIM"])
    runs = []
    for position, ((instance, finished), dat, tdat, mdat) in enumerate(
        zip(items, blocks[".dat"], blocks[".tdat"], blocks[".mdat"])
    ):
        layout = _layout(tdat.header)
        if layout is None or _layout(dat.header) is None or (mdat is not None and _layout(mdat.header) is None):
            issues.append(f"{where}: unknown column header {tdat.header!r}, run {position} skipped")
            continue
        preamble = {k: v for line in tdat.preamble for k, v in _PREAMBLE_KV.findall(line)}
        if "instance" in preamble and int(preamble["instance"]) != instance:
            issues.append(f"{where}: run {position} has instance {instance} but data file says {preamble['instance']}")
        fopt = _FOPT.search(tdat.header)
        try:
            frames = [_frame(dat, suite, dimension, "target"), _frame(tdat, suite, dimension, "evals")]
            if mdat is not None:
                frames.append(_frame(mdat, suite, dimension, "recommendation"))
            data = pl.concat(frames, how="diagonal_relaxed")
        except ValueError as e:
            issues.append(f"{where}: run {position} skipped: {e}")
            continue
        runs.append(
            Run(
                suite=suite,
                function=function,
                dimension=dimension,
                instance=instance,
                finished=finished,
                layout=layout[0],
                fopt=float(fopt[1]) if fopt and not biobj else None,
                reference_value=float(preamble["reference value"]) if "reference value" in preamble else None,
                info_file=info_name,
                info_line=lineno,
                position=position,
                metadata=dict(header),
                data=data.sort(["evals", "reason"], maintain_order=True),
                comment=comment,
            )
        )
    return runs


def _parse_blocks(text: str) -> list[_Block]:
    """Split a data file into run blocks at column-header lines."""
    blocks: list[_Block] = []
    pending: list[str] = []  # comment lines since the last data row
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("%"):
            body = stripped[1:].strip()
            if body.startswith(("function evaluation", "f evaluations")):
                blocks.append(_Block(body, pending, []))
                pending = []
            else:
                pending.append(body)
        elif blocks:
            pending = []  # comments between data rows belong to the current run
            blocks[-1].rows.append(stripped.split())
    return blocks


def _layout(header: str) -> tuple[str, list[tuple[str, Any]]] | None:
    for name, prefix, columns in _LAYOUTS:
        if header.startswith(prefix):
            return name, columns
    return None


def _frame(block: _Block, suite: str, dimension: int, reason: str) -> pl.DataFrame:
    """Convert a run block into a DataFrame with base columns, ``x`` and ``reason``."""
    name, columns = _layout(block.header)
    if name == "new2" and suite == "bbob-constrained":
        columns = columns[:-1] + [("g_digits", pl.String)]  # digit string with significant leading zeros
    width = len(columns)
    has_x = False
    for row in block.rows:
        if len(row) == width + dimension:
            has_x = True
        elif len(row) != width:
            raise ValueError(f"row with {len(row)} fields, expected {width} or {width + dimension}")
    cells = list(zip(*block.rows)) if block.rows else [()] * width
    data: dict[str, pl.Series] = {}
    for (col, dtype), values in zip(columns, cells):
        convert = int if dtype == _I64 else float if dtype == _F64 else str
        data[col] = pl.Series(col, [convert(v) for v in values], dtype=dtype)
    frame = pl.DataFrame(data)
    if has_x and all(len(row) == width + dimension for row in block.rows):
        x = [pl.Series(f"x{i}", [float(v) for v in values], dtype=_F64) for i, values in enumerate(cells[width:])]
        frame = frame.with_columns(pl.concat_list(x).alias("x"))
    elif has_x:  # some rows without x
        frame = frame.with_columns(
            pl.Series("x", [[float(v) for v in row[width:]] or None for row in block.rows], dtype=pl.List(_F64))
        )
    reasons = [reason] * frame.height
    if reason == "evals" and reasons:
        reasons[-1] = "final"
    return frame.select(pl.col("evals"), pl.Series("reason", reasons, dtype=REASON), pl.exclude("evals"))
