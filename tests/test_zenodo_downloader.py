# tests/test_zenodo_downloader.py

"""Tests for the Zenodo download path.

Zenodo is the primary source (record 10.5281/zenodo.21947161) and, unlike
Figshare, publishes a per-file size and checksum. Verifying those is the only
thing standing between a silently corrupted transfer and a cached dataset, so
the failure modes are covered here explicitly.

Offline: ``requests.get`` is replaced with a stub.
"""

# -- import packages: ---------------------------------------------------------
import hashlib

import pytest
import requests

# -- import local dependencies: -----------------------------------------------
from scdiffeq.datasets import _zenodo_downloader as zd

PAYLOAD = b"index,ct_pseudotime\n0,0.5\n"
GOOD_MD5 = hashlib.md5(PAYLOAD).hexdigest()


class FakeResponse:
    def __init__(self, status_code=200, headers=None, body=b"", payload=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=None):
        yield self._body


def _record(size=len(PAYLOAD), checksum=f"md5:{GOOD_MD5}"):
    return {
        "files": [
            {
                "key": "larry.ct_var_df.csv",
                "size": size,
                "checksum": checksum,
                "links": {"self": "https://zenodo.org/api/records/1/files/x/content"},
            }
        ]
    }


@pytest.fixture
def zenodo(monkeypatch):
    """Serve a fake record, then a fake file body."""

    state = {"record": _record(), "body": PAYLOAD}

    def fake_get(url, **kwargs):
        if "/files/" in url:
            return FakeResponse(body=state["body"])
        return FakeResponse(payload=state["record"])

    monkeypatch.setattr(zd.requests, "get", fake_get)
    return state


def test_successful_download_verifies_checksum(zenodo, tmp_path):
    target = tmp_path / "out.csv"
    zd.zenodo_downloader(
        record_id="1", filename="larry.ct_var_df.csv", write_path=str(target)
    )

    assert target.read_bytes() == PAYLOAD


def test_checksum_mismatch_is_rejected(zenodo, tmp_path):
    """A corrupted body must raise rather than be accepted."""

    zenodo["body"] = b"index,ct_pseudotime\n0,9.9\n"  # same length, different bytes

    with pytest.raises(zd.ZenodoChecksumError, match="Checksum mismatch"):
        zd.zenodo_downloader(
            record_id="1",
            filename="larry.ct_var_df.csv",
            write_path=str(tmp_path / "out.csv"),
        )


def test_size_mismatch_is_rejected(zenodo, tmp_path):
    """A truncated transfer must raise even before the checksum is compared."""

    zenodo["body"] = PAYLOAD[:5]

    with pytest.raises(zd.ZenodoChecksumError, match="Size mismatch"):
        zd.zenodo_downloader(
            record_id="1",
            filename="larry.ct_var_df.csv",
            write_path=str(tmp_path / "out.csv"),
        )


def test_unsupported_checksum_algorithm_is_tolerated(zenodo, tmp_path):
    """An algorithm we cannot compute must not break the download."""

    zenodo["record"] = _record(checksum="sha3-512-nonsense:abc")

    target = tmp_path / "out.csv"
    zd.zenodo_downloader(
        record_id="1", filename="larry.ct_var_df.csv", write_path=str(target)
    )

    assert target.read_bytes() == PAYLOAD


def test_missing_file_lists_what_is_available(zenodo, tmp_path):
    with pytest.raises(FileNotFoundError, match="Available files"):
        zd.zenodo_downloader(
            record_id="1", filename="nope.h5ad", write_path=str(tmp_path / "x.h5ad")
        )


def test_corrupt_zenodo_payload_does_not_reach_the_cache(monkeypatch, tmp_path):
    """A checksum failure must leave no file behind at the destination.

    zenodo_file_downloader is what the loaders call, so this is the boundary that
    actually protects the on-disk cache.
    """

    from scdiffeq.datasets import _figshare_downloader as fd

    def boom(*args, **kwargs):
        raise zd.ZenodoChecksumError("Checksum mismatch for 'x'")

    monkeypatch.setattr(fd, "zenodo_downloader", boom)
    monkeypatch.setattr(fd, "ZENODO_RECORD_ID", "1")

    target = tmp_path / "larry_unprocessed.processed.h5ad"
    assert fd.zenodo_file_downloader(filename=target.name, write_path=target) is False
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []
