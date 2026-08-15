# tests/conftest.py

"""Shared fixtures.

The suite is offline by design: the datasets are 2-5 GB each, so every download
is stubbed. Two autouse guards below enforce that, because ZENODO_RECORD_ID now
points at a real published record and it is otherwise easy to turn a stubbed test
into a multi-GB transfer without noticing.
"""

# -- import packages: ---------------------------------------------------------
import pytest


@pytest.fixture(autouse=True)
def zenodo_disabled(request, monkeypatch):
    """Point the downloader at no Zenodo record for offline tests.

    Tests that exercise the Zenodo path patch a fake record id back on themselves.
    """
    if "slow" in request.keywords:
        return
    monkeypatch.setattr(
        "scdiffeq.datasets._figshare_downloader.ZENODO_RECORD_ID", None
    )


@pytest.fixture(autouse=True)
def no_network(request, monkeypatch):
    """Fail loudly instead of hanging if an offline test tries to use the network.

    Tests marked ``slow`` are exempt: they are opt-in and genuinely hit the
    network.
    """
    if "slow" in request.keywords:
        return

    import socket

    def _blocked(self, *args, **kwargs):
        raise RuntimeError(
            "network access attempted in an offline test; stub the request or "
            "mark the test @pytest.mark.slow"
        )

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
