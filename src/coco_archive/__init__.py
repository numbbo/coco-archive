"""Read, convert and write COCO benchmarking result archives with polars.

`COCO <https://github.com/numbbo/coco>`_ (COmparing Continuous Optimisers)
benchmarks optimisation algorithms on suites of test problems such as
``bbob``, ``bbob-noisy``, ``bbob-biobj``, ``bbob-largescale``,
``bbob-mixint``, ``bbob-constrained`` and ``bbob-boxed``. The COCO logger
writes the results of an experiment as text files, and the published results
of several hundred algorithms are collected as ``.tgz``, ``.tar.gz`` or
``.zip`` archives in the `numbbo/data-archive
<https://github.com/numbbo/data-archive>`_ repository.

This package reads such an archive into an `Archive` object, independent of
cocopp. The data of each run is a `polars.DataFrame`, and polars is the only
required dependency. An archive can be stored as a single Parquet file, which
loads much faster than the text files. It can also be written back as a
cleaned-up COCO archive.

Data model
----------
An archive holds the results of one algorithm. It consists of **runs**
(trials), one per execution of the algorithm on a problem instance. A run is
described by its ``suite``, ``function`` number, ``dimension`` and
``instance`` number. Instance numbers can repeat within an archive, e.g. the
2009 experiments ran instances 1-5 three times each, so a run is identified
by its position in `Archive.runs`, not by its instance.

The text archive contains index files (``.info``) and data files. An index
file lists the runs and the data files that hold their rows. Three kinds of
data file are read:

``.dat`` (target-triggered)
    One row each time the best value reaches the next target of a
    logarithmic target grid.
``.tdat`` (budget-triggered)
    One row each time the number of evaluations reaches the next value of a
    logarithmic grid, plus one row at the last evaluation of the run.
``.mdat`` (recommendations, optional)
    One row each time the algorithm recommends a solution with
    ``coco_recommend_solution``. Recent loggers write it, most archives don't
    have it.

The other data files (``.rdat``, ``.cdat``) are ignored.

For each run, the rows of these files are merged into `Run.data` and sorted by
``evals``. The column ``reason`` (an enum, see `REASON`) says why a row was
logged:

==================  =====================  ==================================
reason              source                 meaning
==================  =====================  ==================================
``target``          ``.dat``               the best value reached a new
                                           target
``evals``           ``.tdat``              the evaluation count reached a new
                                           budget
``recommendation``  ``.mdat``              the algorithm recommended a
                                           solution
``final``           last row of ``.tdat``  the last evaluation of the run
==================  =====================  ==================================

The same evaluation can therefore occur in several rows with different
reasons. Every run has exactly one ``final`` row. Its ``evals`` is the total
number of evaluations, and its ``best_f`` (or ``indicator``) is the final best
value.

A recommended solution is evaluated without counting as an evaluation. In a
``recommendation`` row, ``evals`` is the number of evaluations so far,
``f_measured`` and ``x`` are the f-value and the recommended solution, and
``best_f`` and ``best_f_measured`` are the best values of the evaluated
solutions so far, not of the recommendations.

The other columns depend on the column layout of the data files
(`Run.layout`):

===================  ====================  ===================================
column               layouts               meaning
===================  ====================  ===================================
``evals``            all                   number of f-evaluations so far
``g_evals``          new, new2             number of constraint evaluations so
                                           far (0 except in bbob-constrained)
``best_f``           old, new, new2        best noise-free f-value so far
                                           minus fopt (Δf). In
                                           bbob-constrained, the sum of the
                                           constraint violations is added.
``f``                old, new              noise-free f-value of the current
                                           evaluation minus fopt
``f_measured``       old, new, new2        measured (possibly noisy) f-value
                                           of the current evaluation
``best_f_measured``  old, new, new2        best measured f-value so far
``g_digits``         new2 (constrained)    one digit per constraint encoding
                                           the magnitude of the constraint
                                           values, kept as a string
``indicator``        biobj                 hypervolume indicator relative to
                                           `Run.reference_value` (smaller is
                                           better)
``target_hit``       biobj (``.dat``)      smallest target reached, null in
                                           ``.tdat`` rows
``x``                old, new, new2        current solution as a list of
                                           floats, if the logger wrote it
===================  ====================  ===================================

The layouts are ``old`` (logger output up to 2018), ``new`` (``data_format =
bbob-new``, 2018 only), ``new2`` (``data_format = bbob-new2``, 2019 and later)
and ``biobj`` (bi-objective suites). Loggers write ``x`` only up to a
logger-dependent dimension. ``x`` is logged with 5 significant digits.

Examples
--------
Read an archive and look at its runs:

>>> from coco_archive import Archive
>>> url = "https://raw.githubusercontent.com/numbbo/data-archive/gh-pages/data-archive/bbob/2009/BFGS_ros_noiseless.tgz"
>>> archive = Archive.read_archive(url)
>>> archive
Archive('BFGS_ros_noiseless', 2160 runs, 0 issues)
>>> run = archive.runs[0]
>>> run.suite, run.function, run.dimension, run.instance
('bbob', 17, 40, 1)
>>> run.data.columns
['evals', 'reason', 'f', 'best_f', 'f_measured', 'best_f_measured', 'x']

Final evaluation count and best value of every run:

>>> import polars as pl
>>> final = pl.concat(
...     [r.data.filter(pl.col("reason") == "final").select("evals", "best_f").with_columns(run=i)
...      for i, r in enumerate(archive.runs)]
... )

Convert to Parquet once, then load the file in well under a second:

>>> archive.write_parquet("BFGS_ros_noiseless.parquet")
>>> archive = Archive.read_parquet("BFGS_ros_noiseless.parquet")

A Parquet file can also be queried directly with polars, without
`Archive`:

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

Write a cleaned-up COCO archive that cocopp can read:

>>> archive.write_archive("BFGS_ros_noiseless.zip")
"""

from ._version import __version__  # noqa: F401
from .archive import REASON, Archive, Run

__all__ = ["REASON", "Archive", "Run"]
