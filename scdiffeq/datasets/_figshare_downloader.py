# -- import packages: ---------------------------------------------------------
import requests
import tqdm

# -- operational cls: ---------------------------------------------------------
class FigshareDownloader:
    def __init__(self, chunk_size: int = int(8 * 1024 * 1024)):
        """"""
        self.chunk_size = chunk_size

    @property
    def url(self):
        return f"https://figshare.com/ndownloader/files/{self.figshare_id}"

    def _get_size(self, response):
        return int(response.headers.get("Content-Length", 0))

    def download(self, response, write_path: str):

        total_size = self._get_size(response)

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
                    size = f.write(chunk)
                    pbar.update(size)

    def forward(self, write_path: str):
        response = requests.get(self.url, stream=True)
        response.raise_for_status()
        self.download(response=response, write_path=write_path)

    def __call__(self, figshare_id: str, write_path: str):

        self.figshare_id = figshare_id

        return self.forward(write_path=write_path)

# -- function: ----------------------------------------------------------------
def figshare_downloader(figshare_id: str, write_path: str):
    downloader = FigshareDownloader()
    return downloader(figshare_id=figshare_id, write_path=write_path)
