from invoke import task
import sys

PY = sys.executable  # get correct python executable for all platforms

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