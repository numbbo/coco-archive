"""Round trip of a small synthetic archive through the command line interface."""

import functools
import http.server
import io
import shutil
import tarfile
import threading

import polars as pl
import pytest

from coco_archive import REASON, Archive, Run
from coco_archive.cli import main


def _run(function: int, instance: int, position: int, recommendations: bool = False) -> Run:
    data = pl.DataFrame(
        {
            "evals": [1, 5, 1, 8, 10],
            "reason": ["target", "target", "evals", "evals", "final"],
            "g_evals": [0, 0, 0, 0, 0],
            "best_f": [10.0, 1e-3, 10.0, 1e-3, 1e-3],
            "f_measured": [89.5, 79.501, 89.5, 80.25, 80.5],
            "best_f_measured": [89.5, 79.501, 89.5, 79.501, 79.501],
            "x": [[0.5, -1.25], [0.0, 0.125], [0.5, -1.25], [1.5, 2.0], [3.0, -4.5]],
        },
        schema_overrides={"reason": REASON},
    )
    if recommendations:
        recommended = pl.DataFrame(
            {
                "evals": [5, 10],
                "reason": ["recommendation", "recommendation"],
                "g_evals": [0, 0],
                "best_f": [1e-3, 1e-3],
                "f_measured": [79.75, 79.625],
                "best_f_measured": [79.501, 79.501],
                "x": [[0.25, 0.5], [0.0, 0.0625]],
            },
            schema_overrides={"reason": REASON},
        )
        data = pl.concat([data, recommended])
    data = data.sort("evals", "reason", maintain_order=True)
    return Run(
        suite="bbob",
        function=function,
        dimension=2,
        instance=instance,
        finished=True,
        layout="new2",
        fopt=79.5,
        reference_value=None,
        info_file="test.info",
        info_line=3,
        position=position,
        metadata={"Precision": 1e-8, "algId": "test"},
        data=data,
        comment="synthetic test data",
    )


@pytest.fixture
def source(tmp_path):
    # recommendations on function 1 except instance 2, which gets an empty .mdat block; none on function 2
    runs = [_run(f, i, p, recommendations=f == 1 and i != 2) for f in (1, 2) for p, i in enumerate((1, 2, 3))]
    path = tmp_path / "test"
    Archive(path, "test", runs, []).write_directory(path)
    return path


def _key(archive: Archive) -> pl.DataFrame:
    return archive.runs_frame().drop("run", "info_file", "info_line", "position")


def _same(a: Archive, b: Archive) -> bool:
    return _key(a).equals(_key(b)) and all(x.data.equals(y.data) for x, y in zip(a.runs, b.runs))


def test_convert(source, tmp_path):
    original = Archive.read_archive(source)
    assert len(original.runs) == 6
    assert [r.data.height for r in original.runs] == [7, 5, 7, 5, 5, 5]
    assert original.issues == []

    assert main(["convert", str(source), str(tmp_path / "test.parquet")]) == 0
    packed = Archive.read_parquet(tmp_path / "test.parquet")
    assert packed.algorithm == "test"
    assert _same(packed, original)

    assert main(["convert", str(tmp_path / "test.parquet"), str(tmp_path / "out")]) == 0
    assert _same(Archive.read_archive(tmp_path / "out"), original)

    assert main(["convert", str(tmp_path / "test.parquet"), str(tmp_path / "out.zip"), "--algorithm", "renamed"]) == 0
    archive = Archive.read_archive(tmp_path / "out.zip")
    assert archive.algorithm == "out"
    assert archive.runs[0].metadata["algId"] == "renamed"
    assert _same(archive, original)

    assert main(["convert", str(source), str(tmp_path / "clean") + "/"]) == 0
    assert _same(Archive.read_archive(tmp_path / "clean"), original)


def test_convert_explicit_formats(source, tmp_path):
    assert main(["convert", str(source), str(tmp_path / "data.bin"), "--to", "parquet"]) == 0
    args = ["--from", "parquet", "--to", "directory"]
    assert main(["convert", str(tmp_path / "data.bin"), str(tmp_path / "dir.zip"), *args]) == 0
    assert (tmp_path / "dir.zip").is_dir()
    assert _same(Archive.read_archive(tmp_path / "dir.zip"), Archive.read_archive(source))


@pytest.mark.parametrize("target", ["out.tgz", "out.tar.gz", "out.txt"])
def test_convert_rejects_unknown_target(source, tmp_path, target):
    with pytest.raises(SystemExit) as exit:
        main(["convert", str(source), str(tmp_path / target)])
    assert exit.value.code == 2
    assert not (tmp_path / target).exists()


def test_convert_refuses_non_empty_directory(source, tmp_path):
    main(["convert", str(source), str(tmp_path / "test.parquet")])
    with pytest.raises(SystemExit) as exit:
        main(["convert", str(tmp_path / "test.parquet"), str(source)])
    assert exit.value.code == 2


def test_convert_reports_unreadable_source(source, tmp_path):
    with pytest.raises(SystemExit) as exit:
        main(["convert", str(source), str(tmp_path / "out"), "--from", "parquet"])
    assert exit.value.code == 2


def _recommendations(archive: Archive) -> list[int]:
    return [r.data.filter(pl.col("reason") == "recommendation").height for r in archive.runs]


def test_recommendations(source, tmp_path):
    assert (source / "bbob" / "f001_d002.mdat").exists()
    assert not (source / "bbob" / "f002_d002.mdat").exists()
    original = Archive.read_archive(source)
    assert _recommendations(original) == [2, 0, 2, 0, 0, 0]
    rows = original.runs[0].data.filter(pl.col("reason") == "recommendation")
    assert rows["evals"].to_list() == [5, 10]
    assert rows["x"].to_list() == [[0.25, 0.5], [0.0, 0.0625]]
    # the final row stays last
    assert original.runs[0].data["reason"].to_list()[-2:] == ["recommendation", "final"]

    for target in ("out.zip", "out.parquet", "out"):
        assert main(["convert", str(source), str(tmp_path / target)]) == 0
    assert _same(Archive.read_archive(tmp_path / "out.zip"), original)
    assert _same(Archive.read_parquet(tmp_path / "out.parquet"), original)
    assert _same(Archive.read_archive(tmp_path / "out"), original)


def test_recommendations_mismatch(source, capsys):
    mdat = source / "bbob" / "f001_d002.mdat"
    text = mdat.read_text()
    mdat.write_text(text[: text.rindex("% f evaluations")])  # drop the block of the last run
    assert main(["check", str(source)]) == 1
    assert "2 run blocks in .mdat for 3 runs, line skipped" in capsys.readouterr().out


def test_check(source, tmp_path, capsys):
    assert main(["check", str(source)]) == 0
    broken = tmp_path / "broken"
    shutil.copytree(source, broken)
    (broken / "bbob" / "f001_d002.tdat").unlink()
    assert main(["check", str(broken)]) == 1
    assert "missing bbob/f001_d002.tdat" in capsys.readouterr().out


@pytest.fixture
def archives(source, tmp_path):
    """The source archive as .tgz and .zip files."""
    tgz = tmp_path / "test.tgz"
    with tarfile.open(tgz, "w:gz") as tar:
        tar.add(source, arcname="test")
    zip = tmp_path / "test.zip"
    Archive.read_archive(source).write_archive(zip)
    return [tgz, zip]


@pytest.fixture
def server(tmp_path):
    """Base URL of an HTTP server for the files in `tmp_path`."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=tmp_path)
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
        httpd.shutdown()


def test_read_archive_file_object(source, archives):
    original = Archive.read_archive(source)
    for path in archives:
        with open(path, "rb") as f:
            archive = Archive.read_archive(f)
        assert archive.path == path
        assert archive.algorithm == "test"
        assert _key(archive).equals(_key(original))
        archive = Archive.read_archive(io.BytesIO(path.read_bytes()), algorithm="test")
        assert archive.path is None
        assert _key(archive).equals(_key(original))
    with pytest.raises(ValueError):
        Archive.read_archive(io.BytesIO(archives[0].read_bytes()))


def test_read_archive_url(source, archives, server, tmp_path):
    original = Archive.read_archive(source)
    for path in archives:
        url = f"{server}/{path.name}"
        archive = Archive.read_archive(url)
        assert archive.path == url
        assert archive.algorithm == "test"
        assert _key(archive).equals(_key(original))
    archive.write_parquet(tmp_path / "url.parquet")
    assert Archive.read_parquet(tmp_path / "url.parquet").path == url
    assert main(["convert", url, str(tmp_path / "packed.parquet")]) == 0
    assert main(["check", url, f"{server}/packed.parquet"]) == 0
