# -- import packages: ---------------------------------------------------------
import logging
import os
import pathlib
import requests
import tqdm

# -- import local dependencies: -----------------------------------------------
from ._zenodo_downloader import zenodo_downloader

# -- set type hints: ----------------------------------------------------------
from typing import Optional, Tuple, Union

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)


# -- Zenodo configuration: ----------------------------------------------------
# Primary data source: a public Zenodo record, downloadable without
# authentication and checksum-verified on arrival.
# https://doi.org/10.5281/zenodo.21947161
ZENODO_RECORD_ID = "21947161"

# Mapping from Figshare IDs to Zenodo filenames
# Only files uploaded to Zenodo should be listed here
ZENODO_FILES = {
    # LARRY dataset (primary tutorial data)
    "55415231": "larry.h5ad",
    "52612805": "larry_unprocessed.h5ad",
    "54312011": "larry.ct_obs_df.csv",
    "54312008": "larry.ct_var_df.csv",
    "54635780": "Weinreb2020_growth-all_kegg.pt",
    # Human hematopoiesis dataset
    "54154235": "_hsc_all_combined_all_layers.h5ad",
    "54154238": "_dynamo_hematopoiesis_v1.h5ad",
    "54154232": "human_hematopoiesis.processed.h5ad",
    "54154226": "human_hematopoiesis.scaler.pkl",
    "54154223": "human_hematopoiesis.pca.pkl",
    "54154229": "human_hematopoiesis.umap.pkl",
    # Pancreatic endocrinogenesis dataset
    "54151331": "_downloaded.pancreas.h5ad",
    "54151208": "adata.pancreatic_endocrinogenesis.cytotrace.h5ad",
    "54151202": "pancreatic_endocrinogenesis.scaler.pkl",
    "54151205": "pancreatic_endocrinogenesis.pca.pkl",
    "54151199": "pancreatic_endocrinogenesis.umap.pkl",
}


# -- custom exception: --------------------------------------------------------
class FigshareDownloadError(Exception):
    """Raised when all download sources fail (e.g., WAF challenge, network error)."""

    pass


# -- payload validation: ------------------------------------------------------
# Leading bytes that identify a well-formed payload, keyed by target suffix.
# A WAF challenge page or a truncated transfer fails these, which is what keeps a
# corrupt file from being written to the cache and only exploding much later
# inside `anndata.read_h5ad`.
_MAGIC_BY_SUFFIX = {
    ".h5ad": (b"\x89HDF\r\n\x1a\n",),
    ".h5": (b"\x89HDF\r\n\x1a\n",),
    # torch.save writes a zip archive (>=1.6); older checkpoints are raw pickles.
    ".pt": (b"PK\x03\x04", b"\x80"),
    ".pkl": (b"\x80",),
}

# Bodies that are unmistakably an error/challenge page rather than data.
_HTML_PREFIXES = (b"<!DOCTYPE", b"<!doctype", b"<html", b"<HTML")


def _validate_payload(path: pathlib.Path, suffix: str) -> Tuple[bool, str]:
    """Check that a downloaded file looks like the data we asked for.

    Args:
        path: Location of the freshly downloaded (temporary) file.
        suffix: Suffix of the *final* destination, e.g. ``".h5ad"``. The temp file
            carries a ``.part`` suffix, so it cannot be used for this.

    Returns:
        ``(is_valid, reason)``.
    """
    if not path.exists():
        return False, "no file was written"

    size = path.stat().st_size
    if size == 0:
        return False, "file is empty"

    with open(path, "rb") as f:
        head = f.read(16)

    if head.startswith(_HTML_PREFIXES):
        return False, "response body is an HTML page, not data"

    expected = _MAGIC_BY_SUFFIX.get(suffix.lower())
    if expected and not head.startswith(expected):
        return False, (
            f"missing expected {suffix} file signature "
            f"(got {head[:8]!r}, {size} bytes)"
        )

    return True, "ok"


def _content_length_is_zero(response) -> bool:
    """True only when Content-Length is present *and* explicitly zero.

    A missing header is not the same as a zero-length body: chunked and gzipped
    responses legitimately omit it, and treating those as empty would reject
    perfectly good downloads.
    """
    value = response.headers.get("Content-Length")
    return value is not None and value.strip().isdigit() and int(value) == 0


def _unlink(path: pathlib.Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as e:  # pragma: no cover - defensive
        logger.debug(f"Could not remove {path}: {e}")


# -- operational class: -------------------------------------------------------
class FigshareDownloader:
    """Download dataset files, preferring sources that work without authentication.

    Sources are attempted in this order:

    1. Zenodo (public record, no authentication)
    2. Figshare API v2 (works anonymously; a token is used only if one is set)
    3. Figshare direct download (usually blocked by AWS WAF; last resort)

    The download is written to a temporary sibling file and validated before being
    moved into place, so a failed or challenged transfer never leaves a partial
    file that later looks like a populated cache.
    """

    def __init__(
        self,
        chunk_size: int = 8 * 1024 * 1024,
        api_token: Optional[str] = None,
    ):
        """
        Args:
            chunk_size: Download chunk size in bytes (default: 8 MB)
            api_token: Figshare API token. Optional - the API endpoint serves these
                files anonymously. Can also be set via the FIGSHARE_API_TOKEN
                environment variable.
        """
        self.chunk_size = chunk_size
        self.api_token = api_token or os.getenv("FIGSHARE_API_TOKEN")

    @property
    def figshare_url(self) -> str:
        return f"https://figshare.com/ndownloader/files/{self.figshare_id}"

    @property
    def figshare_api_url(self) -> str:
        return f"https://api.figshare.com/v2/file/download/{self.figshare_id}"

    def _download_with_progress(self, response, write_path: str) -> None:
        """Stream response content to disk with a progress bar."""
        value = response.headers.get("Content-Length")
        total_size = int(value) if value is not None and value.strip().isdigit() else 0

        with open(write_path, "wb") as f:
            with tqdm.tqdm(
                total=total_size,
                unit="iB",
                unit_scale=True,
                desc="Downloading",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ) as pbar:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    def _try_zenodo(self, write_path: str) -> bool:
        """Try downloading from Zenodo. Returns True if bytes were written."""
        if ZENODO_RECORD_ID is None:
            logger.debug("Zenodo record ID not configured, skipping Zenodo")
            return False

        zenodo_filename = ZENODO_FILES.get(str(self.figshare_id))
        if not zenodo_filename:
            logger.debug(f"No Zenodo mapping for figshare ID {self.figshare_id}")
            return False

        try:
            logger.info(f"Attempting download from Zenodo: {zenodo_filename}")
            zenodo_downloader(
                record_id=ZENODO_RECORD_ID,
                filename=zenodo_filename,
                write_path=write_path,
                chunk_size=self.chunk_size,
            )
            return True
        except Exception as e:
            logger.warning(f"Zenodo download failed: {e}")
            return False

    def _try_figshare_api(self, write_path: str) -> bool:
        """Try the Figshare API v2 endpoint. Returns True if bytes were written.

        This endpoint serves the files anonymously, so no token is required. When a
        token happens to be configured we send it, but its absence is not a reason
        to skip the attempt - that gate was why anonymous users could not download
        anything while the WAF was blocking the direct host.
        """
        try:
            logger.info("Attempting Figshare API v2 download...")
            headers = {}
            if self.api_token:
                headers["Authorization"] = f"token {self.api_token}"
                logger.debug("Using configured Figshare API token")

            response = requests.get(
                self.figshare_api_url,
                headers=headers,
                stream=True,
                timeout=30,
                allow_redirects=True,
            )
            response.raise_for_status()

            if _content_length_is_zero(response):
                logger.warning("Figshare API returned an empty response")
                return False

            self._download_with_progress(response, write_path)
            return True

        except requests.exceptions.RequestException as e:
            logger.warning(f"Figshare API download failed: {e}")
            return False

    def _try_figshare_direct(self, write_path: str) -> bool:
        """Try the direct Figshare host. Returns True if bytes were written.

        This host sits behind an AWS WAF that answers our requests with an HTTP 202
        challenge and no body, so this is a last-resort fallback.
        """
        try:
            logger.info("Attempting direct Figshare download...")
            response = requests.get(
                self.figshare_url,
                stream=True,
                timeout=30,
                allow_redirects=True,
            )

            waf_action = response.headers.get("x-amzn-waf-action", "").lower()
            if waf_action == "challenge":
                logger.warning("Figshare blocked by AWS WAF challenge")
                return False

            if response.status_code == 202 and _content_length_is_zero(response):
                logger.warning("Figshare returned 202 with no content (likely WAF)")
                return False

            response.raise_for_status()

            if _content_length_is_zero(response):
                logger.warning("Figshare returned an empty response")
                return False

            self._download_with_progress(response, write_path)
            return True

        except requests.exceptions.RequestException as e:
            logger.warning(f"Figshare direct download failed: {e}")
            return False

    def download(self, write_path: Union[str, pathlib.Path]) -> None:
        """Download the file, trying each source until one yields a valid payload.

        Order: Zenodo -> Figshare API v2 -> Figshare direct.
        """
        write_path = pathlib.Path(write_path)
        write_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = write_path.with_name(f".{write_path.name}.part")

        sources = (
            ("Zenodo", self._try_zenodo),
            ("Figshare API v2", self._try_figshare_api),
            ("Figshare direct", self._try_figshare_direct),
        )

        failures = []
        try:
            for label, attempt in sources:
                _unlink(tmp_path)

                try:
                    wrote_bytes = attempt(str(tmp_path))
                except Exception as e:  # never let one source abort the cascade
                    logger.warning(f"{label} download raised {type(e).__name__}: {e}")
                    failures.append(f"{label}: {type(e).__name__}: {e}")
                    continue

                if not wrote_bytes:
                    failures.append(f"{label}: unavailable")
                    continue

                is_valid, reason = _validate_payload(tmp_path, write_path.suffix)
                if not is_valid:
                    logger.warning(
                        f"{label} returned an unusable payload ({reason}); discarding."
                    )
                    failures.append(f"{label}: {reason}")
                    continue

                os.replace(tmp_path, write_path)
                logger.info(f"Downloaded via {label} -> {write_path}")
                return
        finally:
            _unlink(tmp_path)

        detail = "\n".join(f"    - {failure}" for failure in failures)
        raise FigshareDownloadError(
            f"All download sources failed for figshare ID {self.figshare_id}.\n"
            f"  Attempts:\n{detail}\n"
            f"  Troubleshooting:\n"
            f"    1. Retry - the Figshare API endpoint is usually reachable anonymously.\n"
            f"    2. Download manually from: {self.figshare_url}\n"
            f"       and save it to: {write_path}\n"
            f"    3. Set FIGSHARE_API_TOKEN for authenticated API access.\n"
            f"       Get a token from: https://figshare.com/account/applications\n"
        )

    def __call__(self, figshare_id: Union[int, str], write_path: Union[str, pathlib.Path]):
        self.figshare_id = str(figshare_id)
        return self.download(write_path=write_path)


# -- function: ----------------------------------------------------------------
def zenodo_file_downloader(
    filename: str,
    write_path: Union[str, pathlib.Path],
    chunk_size: int = 8 * 1024 * 1024,
) -> bool:
    """Download a file that exists only on Zenodo, with no Figshare counterpart.

    Used for artifacts we publish ourselves (e.g. prebuilt preprocessed objects)
    rather than mirror. Returns ``False`` when Zenodo is not configured or the
    file is unavailable, so callers can fall back to computing it locally.
    """
    if ZENODO_RECORD_ID is None:
        logger.debug("Zenodo record ID not configured; cannot fetch Zenodo-only file")
        return False

    write_path = pathlib.Path(write_path)
    write_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = write_path.with_name(f".{write_path.name}.part")

    try:
        zenodo_downloader(
            record_id=ZENODO_RECORD_ID,
            filename=filename,
            write_path=str(tmp_path),
            chunk_size=chunk_size,
        )
        is_valid, reason = _validate_payload(tmp_path, write_path.suffix)
        if not is_valid:
            logger.warning(f"Zenodo returned an unusable {filename} ({reason}).")
            return False

        os.replace(tmp_path, write_path)
        logger.info(f"Downloaded {filename} from Zenodo -> {write_path}")
        return True

    except Exception as e:
        logger.warning(f"Could not fetch {filename} from Zenodo: {e}")
        return False
    finally:
        _unlink(tmp_path)


def figshare_downloader(
    figshare_id: Union[int, str],
    write_path: Union[str, pathlib.Path],
    chunk_size: int = 8 * 1024 * 1024,
    api_token: Optional[str] = None,
):
    """
    Download a dataset file, preferring sources that work without authentication.

    Sources are attempted in this order:

    1. Zenodo (public record, no authentication required)
    2. Figshare API v2 (serves these files anonymously)
    3. Figshare direct download (usually blocked by AWS WAF)

    The file is validated before being moved into place, so a failed transfer does
    not leave a partial file behind.

    Args:
        figshare_id: Figshare file ID
        write_path: Local path to save the file
        chunk_size: Download chunk size in bytes (default: 8 MB)
        api_token: Optional Figshare API token (or set FIGSHARE_API_TOKEN)

    Raises:
        FigshareDownloadError: If every source fails

    Example:
        >>> figshare_downloader("55415231", "./larry.h5ad")
    """
    downloader = FigshareDownloader(chunk_size=chunk_size, api_token=api_token)
    return downloader(figshare_id=figshare_id, write_path=write_path)
