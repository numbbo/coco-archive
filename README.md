<h1 align="center">
<img src="https://raw.githubusercontent.com/numbbo/coco-archive/main/cocoarc.svg" width="400">
</h1><br>

[![PyPI Downloads](https://img.shields.io/pypi/dm/coco-archive.svg?label=PyPI%20downloads)](https://pypi.org/project/coco-archive/)
[![GitHub Issues](https://img.shields.io/github/issues/numbbo/coco-archive)](https://github.com/numbbo/coco-archive/issues)
![GitHub Repo stars](https://img.shields.io/github/stars/numbbo/coco-archive?style=flat)

`coco-archive` reads, converts and writes COCO result archives (`.tgz`, `.tar.gz`, `.zip`), independent of `cocopp`, with [polars](https://pola.rs) as the only dependency. 

- **Website:** https://coco-platform.org (was: https://numbbo.github.io)
- **Source code:** https://github.com/numbbo/coco-archive/
- **Contributing:** https://github.com/numbbo/coco-archive/pulls
- **Bug reports:** https://github.com/numbbo/coco-archive/issues

## Reading an archive

```python
from coco_archive import Archive

url = "https://raw.githubusercontent.com/numbbo/data-archive/gh-pages/data-archive/bbob/2009/BFGS_ros_noiseless.tgz"
archive = Archive.read_archive(url)  # or a local .tgz/.tar.gz/.zip file, directory or open file
archive.runs_frame()          # one row per run: suite, function, dimension, instance, ...
run = archive.runs[0]
run.data                      # polars DataFrame of this run
archive.issues                # defects found while reading (affected lines are skipped)
```

Archives are read into memory and extracted from there.
A URL is downloaded into memory first.

## Converting to Parquet

Parsing the text files can take quite a bit of time, so for repeated use, you can convert an archive into a single Parquet file, which loads in under a second:

```python
from coco_archive import Archive

archive.write_parquet("BFGS_ros_noiseless.parquet")
archive = Archive.read_parquet("BFGS_ros_noiseless.parquet")
```

The file contains one table with the rows of all runs.
The metadata of the runs is stored in extra columns alongside the results, and all other information as a JSON string in the file's key-value metadata under the key `coco_archive`.

## Command line

```
coco-archive convert https://raw.githubusercontent.com/numbbo/data-archive/gh-pages/data-archive/bbob/2009/BFGS_ros_noiseless.tgz \
    BFGS_ros_noiseless.parquet
coco-archive convert BFGS_ros_noiseless.parquet BFGS_ros_noiseless-clean/
coco-archive convert messy.tgz clean.zip
coco-archive check BFGS_ros_noiseless.parquet other.tgz
```

`convert` reads a directory with COCO results, a COCO archive (`.tgz`, `.tar.gz` or `.zip`) or a Parquet file and writes it as a Parquet file, a sanitized COCO archive in a ZIP file, or a sanitized COCO archive directory (see below). 
Using `--from {archive,parquet}` and `--to {parquet,zip,directory}` you can choose the source and destination formats explicitly.
If you need to adjust the algorithm name, which be default is taken from the file or directory name, pass it in via `--algorithm NAME`. 
`check` lists the issues found while reading each archive or Parquet file and exits with status 1 if there are any. 

## Writing a sanitized COCO archive

```python
archive.write_archive("BFGS_ros_noiseless.zip")
```

 writes a ZIP file (or `archive.write_directory("BFGS_ros_noiseless")` a directory) in the COCO format that cocopp (and `Archive.read_archive`) can read:

* a single index file `BFGS_ros_noiseless.info` with one entry per (suite, function, dimension), or one entry per suite with one data line per (function, dimension) for bi-objective data
* data files `<suite>/f<FFF>_d<DDD>.dat` and `.tdat`, and `.mdat` if any of their runs has recommendations
* No `.rdat` or `.cdat` files are written

## Run data

`run.data` contains the rows of the run's `.dat`, `.tdat` and (if present) `.mdat` files merged and sorted by `evals`. 
The column `reason` gives then reason why a row was logged:

| reason | source | meaning |
|---|---|---|
| `target` | `.dat` | the best value reached a new target |
| `evals` | `.tdat` | the number of evaluations reached a new budget |
| `recommendation` | `.mdat` | the algorithm recommended a solution |
| `final` | last row of `.tdat` | the last evaluation of the run |

The same evaluation can therefore appear in several rows for different reasons.
The remaining columns depend on the layout of the data (`run.layout`), e.g. `g_evals`, `best_f`, `f_measured`, ... for single-objective data and `indicator`, `target_hit` for bi-objective data. 
Decision variables, if logged, are in the list column `x`.
