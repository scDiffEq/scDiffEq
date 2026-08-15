# -- import packages: ---------------------------------------------------------
import hashlib
import logging
import requests
import tqdm

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)


# -- custom exception: --------------------------------------------------------
class ZenodoChecksumError(Exception):
    """Raised when a downloaded file does not match the checksum Zenodo published."""

    pass


# -- operational class: -------------------------------------------------------
class ZenodoDownloader:
    """Download files from Zenodo public records (no authentication required)."""

    def __init__(
        self,
        record_id: str,
        chunk_size: int = 8 * 1024 * 1024,
    ):
        """
        Args:
            record_id: Zenodo record ID (e.g., "1234567")
            chunk_size: Download chunk size in bytes (default: 8 MB)
        """
        self.record_id = str(record_id)
        self.chunk_size = chunk_size
        self.api_url = f"https://zenodo.org/api/records/{self.record_id}"
        self._file_cache = None

    def _fetch_record_metadata(self):
        """Fetch record metadata from Zenodo API."""
        if self._file_cache is not None:
            return self._file_cache

        logger.debug(f"Fetching Zenodo record metadata: {self.api_url}")
        response = requests.get(self.api_url, timeout=30)
        response.raise_for_status()

        record = response.json()
        self._file_cache = {f["key"]: f for f in record.get("files", [])}
        return self._file_cache

    def get_file_info(self, filename: str) -> dict:
        """Get file metadata including download URL."""
        files = self._fetch_record_metadata()

        if filename not in files:
            available = list(files.keys())
            raise FileNotFoundError(
                f"File '{filename}' not found in Zenodo record {self.record_id}. "
                f"Available files: {available}"
            )

        return files[filename]

    def download(self, filename: str, write_path: str):
        """
        Download a file from the Zenodo record.

        Args:
            filename: Name of the file in the Zenodo record
            write_path: Local path to save the file
        """
        file_info = self.get_file_info(filename)
        url = file_info["links"]["self"]
        expected_size = file_info.get("size", 0)
        # Zenodo publishes a per-file checksum as "<algorithm>:<hexdigest>".
        expected_checksum = file_info.get("checksum") or ""

        logger.info(f"Downloading from Zenodo: {filename}")
        logger.debug(f"URL: {url}")

        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("Content-Length", expected_size))

        algorithm, _, expected_digest = expected_checksum.partition(":")
        hasher = None
        if expected_digest:
            try:
                hasher = hashlib.new(algorithm)
            except ValueError:
                logger.debug(f"Unsupported Zenodo checksum algorithm: {algorithm!r}")

        n_bytes = 0
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
                        if hasher is not None:
                            hasher.update(chunk)
                        n_bytes += len(chunk)
                        pbar.update(len(chunk))

        if expected_size and n_bytes != expected_size:
            raise ZenodoChecksumError(
                f"Size mismatch for '{filename}': expected {expected_size} bytes, "
                f"received {n_bytes}."
            )

        if hasher is not None:
            digest = hasher.hexdigest()
            if digest != expected_digest:
                raise ZenodoChecksumError(
                    f"Checksum mismatch for '{filename}': expected "
                    f"{algorithm}:{expected_digest}, computed {algorithm}:{digest}."
                )
            logger.debug(f"Verified {algorithm} checksum for {filename}")

        logger.info(f"Download complete: {write_path}")


# -- function: ----------------------------------------------------------------
def zenodo_downloader(
    record_id: str,
    filename: str,
    write_path: str,
    chunk_size: int = 8 * 1024 * 1024,
):
    """
    Download a file from a Zenodo public record.

    No authentication required for public records.

    Args:
        record_id: Zenodo record ID (e.g., "1234567")
        filename: Name of the file in the Zenodo record
        write_path: Local path to save the file
        chunk_size: Download chunk size in bytes (default: 8 MB)

    Example:
        >>> zenodo_downloader(
        ...     record_id="1234567",
        ...     filename="data.h5ad",
        ...     write_path="./data.h5ad"
        ... )
    """
    downloader = ZenodoDownloader(record_id=record_id, chunk_size=chunk_size)
    return downloader.download(filename=filename, write_path=write_path)
