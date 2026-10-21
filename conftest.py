"""Setup for the doctests in ``src``: run each in an empty directory, with the example archive downloaded once."""

import email.message
import shutil
import urllib.request
import urllib.response

import pytest
from _pytest.doctest import DoctestItem

EXAMPLE_URL = (
    "https://raw.githubusercontent.com/numbbo/data-archive/gh-pages/data-archive/bbob/2009/BFGS_ros_noiseless.tgz"
)


@pytest.fixture(scope="session")
def example_archive(request):
    """Local copy of the example archive, kept in the pytest cache between sessions."""
    path = request.config.cache.mkdir("coco-archive") / "BFGS_ros_noiseless.tgz"
    if not path.exists():
        try:
            with urllib.request.urlopen(EXAMPLE_URL) as response, open(f"{path}.part", "wb") as f:
                shutil.copyfileobj(response, f)
        except OSError as e:
            pytest.skip(f"can't download the example archive: {e}")
        path.with_name(f"{path.name}.part").rename(path)
    return path


@pytest.fixture(autouse=True)
def _doctest_setup(request, tmp_path, monkeypatch):
    if not isinstance(request.node, DoctestItem):
        return
    path = request.getfixturevalue("example_archive")
    urlopen = urllib.request.urlopen

    def cached_urlopen(url, *args, **kwargs):
        if url != EXAMPLE_URL:
            return urlopen(url, *args, **kwargs)
        return urllib.response.addinfourl(open(path, "rb"), email.message.Message(), url, 200)

    monkeypatch.setattr(urllib.request, "urlopen", cached_urlopen)
    monkeypatch.chdir(tmp_path)
