from invoke import task
import sys
from pathlib import Path
import requests as rq
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = sys.executable  # get correct python executable for all platforms


def _download_litbank(_fname: str, dest: Path):
    raw_url = (
        "https://raw.githubusercontent.com/dbamman/litbank/refs/heads/master/original/"
    )
    url = f"{raw_url}/{_fname}"
    response = rq.get(url)
    response.raise_for_status()
    success = response.status_code == 200
    if success:
        content = response.text
        file_path = dest / _fname
        file_path.write_text(content)
    return _fname, success


@task
def download_google_vectors(c):
    """
    Download Google News vectors using gensim-downloader.
    """
    cmd = f"{PY} -m gensim.downloader --download word2vec-google-news-300"
    c.run(cmd, echo=True)


@task
def download_spacy_model(c):
    """
    Download spaCy English small model.
    """
    cmd = f"{PY} -m spacy download en_core_web_sm"
    c.run(cmd, echo=True)


@task(pre=[download_google_vectors, download_spacy_model])
def init_models(c):
    """
    Download *all* NLP resources needed.
    """
    print("All NLP models downloaded successfully.")


@task
def download_litbank(c, overwrite: bool = False):
    """
    Download books from the LitBank repository.

    Args:
        c: The command.
        overwrite: Overwrite existing files.
    """

    url = "https://github.com/dbamman/litbank/blob/master/original"
    response = rq.get(url)
    success = response.status_code == 200

    if not success:
        print(
            f"Failed to parse the LitBank repository content. Please ensure that you have access to the Internet."
        )
        return

    # Parse the HTML content of the response
    soup = BeautifulSoup(response.content, "html.parser")
    # Find all links
    links = soup.find_all("a")
    fnames = set()
    for link in links:
        href = link["href"]
        if href.endswith("txt"):
            # Extract the href attribute of each link
            _fname = href.split("/")[-1]
            if len(_fname) > 0:
                fnames.add(_fname)

    # Ensure that the destination path exists
    dest = Path.cwd() / "data/litbank"
    dest.mkdir(parents=True, exist_ok=True)

    # Download the books in parallel
    pbar = tqdm(desc="Downloading LitBank", total=len(fnames))
    futures = []
    with pbar, ThreadPoolExecutor() as tp:
        for _fname in fnames:
            if (dest / _fname).exists() and not overwrite:
                pbar.update()
                continue
            futures.append(tp.submit(_download_litbank, _fname, dest))
        for fut in as_completed(futures):
            _fname, success = fut.result()
            pbar.update()
