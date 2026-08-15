#!/usr/bin/env python
"""Mirror the scDiffEq dataset artifacts from Figshare to a Zenodo record.

Maintainer tooling -- not shipped in the wheel and not imported by the package.

Why this exists: the direct Figshare download host sits behind an AWS WAF that
answers with an empty HTTP 202, so it cannot be relied on. Zenodo serves public
records anonymously and publishes a per-file checksum, which the runtime
downloader verifies.

Usage:

    export ZENODO_TOKEN=...                       # deposit:write scope

    # See what would happen, transfer nothing:
    python scripts/mirror_to_zenodo.py --dry-run

    # Practise against the sandbox first (separate token):
    python scripts/mirror_to_zenodo.py --sandbox

    # Real run; resumable, so re-running skips what is already uploaded:
    python scripts/mirror_to_zenodo.py --out-dir /path/with/10GB/free

    # One file at a time:
    python scripts/mirror_to_zenodo.py --only larry.h5ad

The deposition is created as a draft. Publishing it -- which mints the DOI and
makes the files immutable -- is left as a deliberate manual step in the Zenodo UI.

Afterwards, set ZENODO_RECORD_ID in src/scdiffeq/datasets/_figshare_downloader.py
to the published record ID.
"""

# -- import packages: ---------------------------------------------------------
import argparse
import logging
import os
import pathlib
import sys

import requests

# -- import local dependencies: -----------------------------------------------
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from scdiffeq.datasets._figshare_downloader import (  # noqa: E402
    ZENODO_FILES,
    FigshareDownloadError,
    figshare_downloader,
)

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger("mirror_to_zenodo")

TIMEOUT = 60


# -- Zenodo deposition API: ---------------------------------------------------
class ZenodoDeposition:
    """Thin wrapper over the Zenodo deposition + bucket upload API."""

    def __init__(self, token: str, sandbox: bool = False):
        self.base = (
            "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api"
        )
        self.session = requests.Session()
        # Send the token as a header, not a query parameter. As a query parameter
        # it ends up in server logs, proxies, and -- as this script demonstrated --
        # in the URL echoed by requests' own HTTPError messages.
        self.session.headers["Authorization"] = f"Bearer {token}"
        self.deposition_id = None
        self.bucket_url = None

    def create(self, metadata: dict) -> dict:
        response = self.session.post(
            f"{self.base}/deposit/depositions",
            json={"metadata": metadata},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return self._adopt(response.json())

    def fetch(self, deposition_id: str) -> dict:
        response = self.session.get(
            f"{self.base}/deposit/depositions/{deposition_id}", timeout=TIMEOUT
        )
        response.raise_for_status()
        return self._adopt(response.json())

    def unlock(self) -> bool:
        """Reopen a published record for metadata editing.

        A published deposition is locked; PUT returns 404 until it is unlocked
        with the ``edit`` action. Files stay immutable either way -- only metadata
        can change. Returns True if the record was published and is now editable.
        """
        response = self.session.post(
            f"{self.base}/deposit/depositions/{self.deposition_id}/actions/edit",
            timeout=TIMEOUT,
        )
        if response.status_code == 201:
            return True
        # 400 == already an open draft, which is fine.
        if response.status_code == 400:
            return False
        response.raise_for_status()
        return False

    def publish(self) -> dict:
        """Publish the deposition. Required to make edits to a published record live."""
        response = self.session.post(
            f"{self.base}/deposit/depositions/{self.deposition_id}/actions/publish",
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def update_metadata(self, metadata: dict) -> dict:
        """Replace the deposition's metadata."""
        response = self.session.put(
            f"{self.base}/deposit/depositions/{self.deposition_id}",
            json={"metadata": metadata},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def _adopt(self, payload: dict) -> dict:
        self.deposition_id = payload["id"]
        self.bucket_url = payload["links"]["bucket"]
        return payload

    def existing_filenames(self) -> set:
        response = self.session.get(
            f"{self.base}/deposit/depositions/{self.deposition_id}/files",
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return {entry["filename"] for entry in response.json()}

    def upload(self, path: pathlib.Path, filename: str) -> dict:
        """Stream a file into the deposition bucket."""
        with open(path, "rb") as handle:
            response = self.session.put(
                f"{self.bucket_url}/{filename}", data=handle, timeout=None
            )
        response.raise_for_status()
        return response.json()


# Files we publish ourselves rather than mirror: they have no Figshare source and
# must already be staged in --out-dir. Built by running the loader at its default
# flags, so users get X_pca without recomputing it.
ZENODO_ONLY_FILES = (
    "larry_unprocessed.processed.h5ad",
)


DESCRIPTION = """\
<p>Single-cell datasets used by the
<a href="https://github.com/scDiffEq/scDiffEq">scDiffEq</a> package, which models
single-cell dynamics with neural differential equations.</p>

<p>These files are mirrored here so that <code>scdiffeq.datasets</code> can fetch
them without authentication. The previous host sits behind a web application
firewall that rejects programmatic downloads, which left users unable to obtain
the data at all.</p>

<p><strong>Contents</strong></p>
<ul>
  <li><em>LARRY in vitro</em> (Weinreb et al.) &mdash; <code>larry.h5ad</code>
      (130,887 &times; 2,492, with precomputed PCA/UMAP),
      <code>larry_unprocessed.h5ad</code> (130,887 &times; 25,289, unfiltered),
      a prebuilt preprocessed form of the latter, precomputed CytoTRACE
      annotations, and KEGG-derived growth weights.</li>
  <li><em>Human hematopoiesis</em> &mdash; raw and processed objects plus the
      fitted scaler/PCA/UMAP models.</li>
  <li><em>Pancreatic endocrinogenesis</em> &mdash; raw and processed objects plus
      the fitted scaler/PCA/UMAP models.</li>
</ul>

<p><strong>Provenance.</strong> These are redistributed and derived copies of data
published by their original authors. <strong>Please cite the original
publications</strong> alongside this record:</p>
<ul>
  <li><em>LARRY in vitro</em> &mdash; Weinreb C, Rodriguez-Fraticelli A, Camargo
      FD, Klein AM. Lineage tracing on transcriptional landscapes links state to
      fate during differentiation. <em>Science</em> (2020).
      <a href="https://doi.org/10.1126/science.aaw3381">10.1126/science.aaw3381</a></li>
  <li><em>Human hematopoiesis</em> &mdash; Qiu X, Zhang Y, Martin-Rufino JD, et al.
      Mapping transcriptomic vector fields of single cells. <em>Cell</em> (2022).
      <a href="https://doi.org/10.1016/j.cell.2021.12.045">10.1016/j.cell.2021.12.045</a></li>
  <li><em>Pancreatic endocrinogenesis</em> &mdash; Bastidas-Ponce A, Tritschler S,
      Dony L, et al. Comprehensive single cell mRNA profiling reveals a detailed
      roadmap for pancreatic endocrinogenesis. <em>Development</em> (2019).
      <a href="https://doi.org/10.1242/dev.173849">10.1242/dev.173849</a></li>
</ul>

<p>The <code>.h5ad</code> objects carry preprocessing applied by scDiffEq (gene
filtering, standardization, and a 50-component PCA with a fixed random seed).
Note that the seeded PCA is reproducible going forward but does not reproduce the
basis distributed with the original publications, which was computed with an
unseeded randomized SVD.</p>
"""

DEFAULT_METADATA = {
    "title": (
        "scDiffEq datasets: LARRY in vitro, human hematopoiesis, "
        "and pancreatic endocrinogenesis"
    ),
    "upload_type": "dataset",
    "description": DESCRIPTION,
    "creators": [{"name": "Vinyard, Michael E."}],
    "keywords": [
        "single-cell",
        "scRNA-seq",
        "lineage tracing",
        "cell fate",
        "neural differential equations",
        "scDiffEq",
    ],
    # DOIs verified against Crossref (title, journal, year, first author) before
    # being written here -- these become part of the permanent record.
    "related_identifiers": [
        {
            "identifier": "https://github.com/scDiffEq/scDiffEq",
            "relation": "isSupplementTo",
            "scheme": "url",
        },
        # The paper this data supports.
        {
            "identifier": "10.1038/s42256-025-01150-3",
            "relation": "isSupplementTo",
            "scheme": "doi",
            "resource_type": "publication-article",
        },
        {
            "identifier": "10.1101/2023.12.06.570508",
            "relation": "isSupplementTo",
            "scheme": "doi",
            "resource_type": "publication-preprint",
        },
        # Originating publications for the redistributed datasets.
        {
            "identifier": "10.1126/science.aaw3381",  # LARRY in vitro
            "relation": "isDerivedFrom",
            "scheme": "doi",
            "resource_type": "publication-article",
        },
        {
            "identifier": "10.1016/j.cell.2021.12.045",  # human hematopoiesis
            "relation": "isDerivedFrom",
            "scheme": "doi",
            "resource_type": "publication-article",
        },
        {
            "identifier": "10.1242/dev.173849",  # pancreatic endocrinogenesis
            "relation": "isDerivedFrom",
            "scheme": "doi",
            "resource_type": "publication-article",
        },
    ],
}


def build_metadata(license_id: str) -> dict:
    metadata = dict(DEFAULT_METADATA)
    metadata["license"] = license_id
    return metadata


# -- steps: -------------------------------------------------------------------
def read_token(token_file: str) -> str:
    """Resolve the Zenodo token from the environment, else from a local file.

    Reading from a file keeps the token out of shell history and out of the
    process list, where a command-line argument would be visible to other users.
    """
    token = os.getenv("ZENODO_TOKEN")
    if token:
        logger.debug("Using ZENODO_TOKEN from the environment")
        return token.strip()

    path = pathlib.Path(token_file).expanduser()
    if path.exists():
        mode = path.stat().st_mode & 0o077
        if mode:
            logger.warning(
                f"{path} is group/world readable; consider `chmod 600 {path}`"
            )
        logger.debug(f"Using Zenodo token from {path}")
        return path.read_text().strip()

    return ""


def fetch_from_figshare(
    figshare_id: str, filename: str, out_dir: pathlib.Path
) -> pathlib.Path:
    """Download one artifact, reusing the package's validated downloader."""
    path = out_dir / filename
    if path.exists() and path.stat().st_size > 0:
        logger.info(f"[cached] {filename} ({path.stat().st_size / 1e6:.1f} MB)")
        return path

    logger.info(f"[fetch ] {filename} (figshare {figshare_id})")
    figshare_downloader(figshare_id=figshare_id, write_path=path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="./zenodo_mirror",
        help="Staging directory for downloads (needs ~9 GB free).",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="FILENAME",
        help="Mirror only this filename (repeatable).",
    )
    parser.add_argument(
        "--deposition-id",
        default=None,
        help="Resume into an existing draft deposition instead of creating one.",
    )
    parser.add_argument(
        "--token-file",
        default=".zenodo_token",
        help=(
            "File holding the Zenodo token (default: .zenodo_token, gitignored). "
            "Used only when ZENODO_TOKEN is not set. Keeps the token out of shell "
            "history and process listings."
        ),
    )
    parser.add_argument(
        "--license",
        default="cc-by-4.0",
        help=(
            "Zenodo license identifier for the record (default: cc-by-4.0). "
            "Required to publish, and immutable afterwards."
        ),
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help=(
            "Update the deposition's metadata and exit; transfer no files. "
            "Requires --deposition-id."
        ),
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Use sandbox.zenodo.org (needs a separate sandbox token).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be transferred, then exit.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Download from Figshare only; do not touch Zenodo.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    # (filename -> figshare_id or None for locally-staged, Zenodo-only files)
    targets = {name: fid for fid, name in ZENODO_FILES.items()}
    targets.update({name: None for name in ZENODO_ONLY_FILES})

    if args.only:
        wanted = set(args.only)
        unknown = wanted - set(targets)
        if unknown:
            parser.error(f"unknown filename(s): {sorted(unknown)}")
        targets = {name: fid for name, fid in targets.items() if name in wanted}

    if not targets:
        parser.error("nothing to do")

    logger.info(f"{len(targets)} artifact(s) selected:")
    for filename, figshare_id in sorted(targets.items()):
        origin = f"figshare {figshare_id}" if figshare_id else "local (Zenodo-only)"
        logger.info(f"  {filename}  [{origin}]")

    if args.dry_run:
        logger.info("Dry run - nothing transferred.")
        return 0

    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    deposition = None
    already_uploaded = set()

    if not args.skip_upload:
        token = read_token(args.token_file)
        if not token:
            parser.error(
                f"No Zenodo token found. Either set ZENODO_TOKEN, or write the token "
                f"to {args.token_file} (gitignored):\n"
                f"    printf %s 'YOUR_TOKEN' > {args.token_file}\n"
                f"    chmod 600 {args.token_file}\n"
                f"Create a token with the 'deposit:write' scope at "
                f"https://zenodo.org/account/settings/applications/tokens/new/"
            )

        metadata = build_metadata(args.license)
        deposition = ZenodoDeposition(token=token, sandbox=args.sandbox)

        if args.deposition_id:
            deposition.fetch(args.deposition_id)
            already_uploaded = deposition.existing_filenames()
            logger.info(
                f"Resuming deposition {deposition.deposition_id} "
                f"({len(already_uploaded)} file(s) already uploaded)"
            )
            was_published = deposition.unlock()
            if was_published:
                logger.info("Record is published; reopened it for metadata editing")

            deposition.update_metadata(metadata)
            logger.info(f"Metadata updated (license: {args.license})")

            if was_published:
                # An unlocked record stays in edit mode until republished, and its
                # public metadata does not reflect the change until then. Files are
                # untouched; this republishes metadata only, keeping the same DOI.
                deposition.publish()
                logger.info("Republished (same DOI; files unchanged)")
        else:
            deposition.create(metadata)
            logger.info(
                f"Created draft deposition {deposition.deposition_id} "
                f"(license: {args.license})"
            )

        if args.metadata_only:
            logger.info("Metadata-only run - no files transferred.")
            return 0

    failures = []
    for filename, figshare_id in sorted(targets.items()):
        try:
            if filename in already_uploaded:
                logger.info(f"[skip  ] {filename} (already in the deposition)")
                continue

            if figshare_id is None:
                path = out_dir / filename
                if not path.exists():
                    raise FileNotFoundError(
                        f"{filename} has no Figshare source and is not staged in "
                        f"{out_dir}. Build it with the loader at its default flags "
                        f"and copy it there."
                    )
                logger.info(f"[local ] {filename} ({path.stat().st_size / 1e6:.1f} MB)")
            else:
                path = fetch_from_figshare(figshare_id, filename, out_dir)

            if deposition is not None:
                size_mb = path.stat().st_size / 1e6
                logger.info(f"[upload] {filename} ({size_mb:.1f} MB)")
                deposition.upload(path, filename)

        except (FigshareDownloadError, requests.RequestException, OSError) as e:
            logger.error(f"[FAILED] {filename}: {e}")
            failures.append(filename)

    if deposition is not None:
        logger.info("")
        logger.info(f"Draft deposition: {deposition.deposition_id}")
        logger.info(
            "Review and publish it in the Zenodo UI, then set ZENODO_RECORD_ID in "
            "src/scdiffeq/datasets/_figshare_downloader.py to the published record ID."
        )

    if failures:
        logger.error(f"{len(failures)} artifact(s) failed: {failures}")
        logger.error("Re-run to retry; completed files are skipped.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
