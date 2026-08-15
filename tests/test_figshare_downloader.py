# tests/test_figshare_downloader.py

"""Regression tests for the dataset downloader.

These cover the defect reported against ``scdiffeq.datasets.larry``: the direct
Figshare host is behind an AWS WAF that answers with an empty HTTP 202, and the
API fallback used to be skipped entirely unless an API token was configured -- so
anonymous users could not download anything, and a challenge body could be written
to disk and only fail much later inside ``anndata.read_h5ad``.

Everything here is offline; ``requests.get`` is replaced with a stub.
"""

# -- import packages: ---------------------------------------------------------
import pathlib

import pytest
import requests

# -- import local dependencies: -----------------------------------------------
from scdiffeq.datasets import _figshare_downloader as fd

# -- constants: ---------------------------------------------------------------
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


# -- test doubles: ------------------------------------------------------------
class FakeResponse:
    """Minimal stand-in for a streamed ``requests`` response."""

    def __init__(self, status_code=200, headers=None, body=b"", chunks=None):
        self.status_code = status_code
        self.headers = headers if headers is not None else {}
        self._body = body
        self._chunks = chunks

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=None):
        if self._chunks is not None:
            for chunk in self._chunks:
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk
        else:
            yield self._body


@pytest.fixture
def recorder(monkeypatch):
    """Patch ``requests.get`` and record every URL requested."""

    calls = []
    responses = {}

    def fake_get(url, **kwargs):
        calls.append({"url": url, "kwargs": kwargs})
        for pattern, response in responses.items():
            if pattern in url:
                if isinstance(response, Exception):
                    raise response
                return response
        raise AssertionError(f"unexpected URL requested: {url}")

    monkeypatch.setattr(fd.requests, "get", fake_get)
    return {"calls": calls, "responses": responses}


def _urls(recorder):
    return [call["url"] for call in recorder["calls"]]


# -- tests: -------------------------------------------------------------------
def test_waf_challenge_writes_no_file(recorder, tmp_path):
    """A 202 + x-amzn-waf-action challenge must not be reported as success."""

    waf = FakeResponse(
        status_code=202,
        headers={"Content-Length": "0", "x-amzn-waf-action": "challenge"},
        body=b"",
    )
    recorder["responses"]["api.figshare.com"] = waf
    recorder["responses"]["figshare.com/ndownloader"] = waf

    target = tmp_path / "larry.h5ad"

    with pytest.raises(fd.FigshareDownloadError):
        fd.figshare_downloader(figshare_id="55415231", write_path=target)

    assert not target.exists()
    assert list(tmp_path.iterdir()) == [], "no partial files may be left behind"


def test_challenge_page_body_fails_validation(recorder, tmp_path):
    """An HTML challenge page served with HTTP 200 must not become an .h5ad."""

    html = FakeResponse(
        status_code=200,
        headers={"Content-Length": "48"},
        body=b"<!DOCTYPE html><html><body>Just a moment</body></html>",
    )
    recorder["responses"]["api.figshare.com"] = html
    recorder["responses"]["figshare.com/ndownloader"] = html

    target = tmp_path / "larry.h5ad"

    with pytest.raises(fd.FigshareDownloadError) as excinfo:
        fd.figshare_downloader(figshare_id="55415231", write_path=target)

    assert "HTML" in str(excinfo.value)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_truncated_h5ad_fails_validation(recorder, tmp_path):
    """A short/garbage body must be rejected rather than cached as a dataset."""

    garbage = FakeResponse(status_code=200, body=b"not an hdf5 file at all")
    recorder["responses"]["api.figshare.com"] = garbage
    recorder["responses"]["figshare.com/ndownloader"] = garbage

    target = tmp_path / "larry.h5ad"

    with pytest.raises(fd.FigshareDownloadError) as excinfo:
        fd.figshare_downloader(figshare_id="55415231", write_path=target)

    assert "signature" in str(excinfo.value)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_api_is_attempted_without_a_token(recorder, tmp_path, monkeypatch):
    """The API endpoint serves these files anonymously - never gate it on a token.

    This is the defect that made the original WAF fix ineffective for end users.
    """

    monkeypatch.delenv("FIGSHARE_API_TOKEN", raising=False)

    recorder["responses"]["api.figshare.com"] = FakeResponse(
        status_code=200,
        headers={"Content-Length": str(len(HDF5_MAGIC))},
        body=HDF5_MAGIC,
    )

    target = tmp_path / "larry.h5ad"
    fd.figshare_downloader(figshare_id="55415231", write_path=target)

    assert target.exists()
    assert target.read_bytes() == HDF5_MAGIC

    requested = _urls(recorder)
    assert any("api.figshare.com/v2/file/download" in url for url in requested)
    # No Authorization header should be sent when no token is configured.
    assert recorder["calls"][0]["kwargs"].get("headers") == {}


def test_missing_content_length_is_not_treated_as_empty(recorder, tmp_path):
    """Chunked/gzipped responses omit Content-Length; that is not an empty body."""

    recorder["responses"]["api.figshare.com"] = FakeResponse(
        status_code=200,
        headers={},  # deliberately no Content-Length
        body=HDF5_MAGIC + b"payload",
    )

    target = tmp_path / "larry.h5ad"
    fd.figshare_downloader(figshare_id="55415231", write_path=target)

    assert target.exists()
    assert target.read_bytes() == HDF5_MAGIC + b"payload"


def test_api_is_tried_before_the_waf_blocked_direct_host(recorder, tmp_path):
    """Order must be Zenodo -> API v2 -> direct, so the blocked host is last."""

    recorder["responses"]["api.figshare.com"] = FakeResponse(
        status_code=200, body=HDF5_MAGIC
    )
    recorder["responses"]["figshare.com/ndownloader"] = FakeResponse(
        status_code=202,
        headers={"Content-Length": "0", "x-amzn-waf-action": "challenge"},
    )

    target = tmp_path / "larry.h5ad"
    fd.figshare_downloader(figshare_id="55415231", write_path=target)

    requested = _urls(recorder)
    assert len(requested) == 1, "direct host should not be contacted after API success"
    assert "api.figshare.com" in requested[0]


def test_direct_host_is_used_when_the_api_fails(recorder, tmp_path):
    """The direct host remains a genuine fallback, not dead code."""

    recorder["responses"]["api.figshare.com"] = requests.exceptions.ConnectionError(
        "boom"
    )
    recorder["responses"]["figshare.com/ndownloader"] = FakeResponse(
        status_code=200, body=HDF5_MAGIC
    )

    target = tmp_path / "larry.h5ad"
    fd.figshare_downloader(figshare_id="55415231", write_path=target)

    assert target.exists()
    requested = _urls(recorder)
    assert "api.figshare.com" in requested[0]
    assert "figshare.com/ndownloader" in requested[1]


def test_partial_file_is_removed_after_a_mid_stream_failure(recorder, tmp_path):
    """A connection dropped mid-transfer must not leave a truncated cache file."""

    recorder["responses"]["api.figshare.com"] = FakeResponse(
        status_code=200,
        chunks=[HDF5_MAGIC, requests.exceptions.ConnectionError("dropped")],
    )
    recorder["responses"]["figshare.com/ndownloader"] = FakeResponse(
        status_code=202,
        headers={"Content-Length": "0", "x-amzn-waf-action": "challenge"},
    )

    target = tmp_path / "larry.h5ad"

    with pytest.raises(fd.FigshareDownloadError):
        fd.figshare_downloader(figshare_id="55415231", write_path=target)

    assert not target.exists()
    assert list(tmp_path.iterdir()) == [], "the .part file must be cleaned up"


def test_non_h5ad_targets_are_validated_too(recorder, tmp_path):
    """CSV targets still reject challenge pages, but need no binary signature."""

    recorder["responses"]["api.figshare.com"] = FakeResponse(
        status_code=200, body=b"index,ct_pseudotime\n0,0.5\n"
    )

    target = tmp_path / "larry.ct_obs_df.csv"
    fd.figshare_downloader(figshare_id="54312011", write_path=target)

    assert target.exists()
    assert target.read_text().startswith("index,")


def test_token_is_sent_when_configured(recorder, tmp_path, monkeypatch):
    """A configured token is still used - it just is not required."""

    monkeypatch.setenv("FIGSHARE_API_TOKEN", "sekrit")
    recorder["responses"]["api.figshare.com"] = FakeResponse(
        status_code=200, body=HDF5_MAGIC
    )

    fd.figshare_downloader(figshare_id="55415231", write_path=tmp_path / "larry.h5ad")

    headers = recorder["calls"][0]["kwargs"].get("headers")
    assert headers == {"Authorization": "token sekrit"}


def test_write_path_parent_is_created(recorder, tmp_path):
    """Downloading into a not-yet-existing directory should just work."""

    recorder["responses"]["api.figshare.com"] = FakeResponse(
        status_code=200, body=HDF5_MAGIC
    )

    target = pathlib.Path(tmp_path) / "nested" / "dir" / "larry.h5ad"
    fd.figshare_downloader(figshare_id="55415231", write_path=target)

    assert target.exists()
