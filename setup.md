---
title: Setup
---

## Software Setup

::::::::::::::::::::::::::::::::::::::: discussion

### Installing Python

[Python](https://python.org) is a popular language for scientific computing, and a frequent choice
for machine learning as well.
To install Python, follow the [Beginner's Guide](https://wiki.python.org/moin/BeginnersGuide/Download) or head straight to the [download page](https://www.python.org/downloads/).

Please set up your python environment at least a day in advance of the workshop.
If you encounter problems with the installation procedure, ask your workshop organizers via e-mail for assistance so
you are ready to go as soon as the workshop begins.

:::::::::::::::::::::::::::::::::::::::::::::::::::

## Installing the required packages

[Pip](https://pip.pypa.io/en/stable/) is the package management system built into Python.
Pip should be available in your system once you installed Python successfully. Please note that installing the packages can take some time, in particular on Windows.


Open a terminal (Mac/Linux) or Command Prompt (Windows) and run the following commands.

1. Create a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments) called `nlp_workshop`:

::: spoiler

### On Linux/macOs

```shell
python3 -m venv nlp_workshop
```

:::

::: spoiler

### On Windows

```shell
py -m venv nlp_workshop
```

:::

2. Activate the newly created virtual environment:

::: spoiler

### On Linux/macOs

```shell
source nlp_workshop/bin/activate
```

:::

::: spoiler

### On Windows

```shell
nlp_workshop\Scripts\activate
```

:::

Remember that you need to activate your environment every time you restart your terminal!

3. Install the required packages:

::: spoiler

### On Linux/macOs

```shell
python3 -m pip install jupyterlab jieba spacy gensim matplotlib transformers  
```

:::

::: spoiler

### On Windows

```shell
py -m pip install jupyterlab jieba spacy gensim matplotlib transformers
```

:::

## Jupyter Lab

We will teach using Python in [Jupyter Lab](http://jupyter.org/), a programming environment that runs in a web browser.
Jupyter Lab is compatible with Firefox, Chrome, Safari and Chromium-based browsers.
Note that Internet Explorer and Edge are *not* supported.
See the [Jupyter Lab documentation](https://jupyterlab.readthedocs.io/en/latest/getting_started/accessibility.html#compatibility-with-browsers-and-assistive-technology) for an up-to-date list of supported browsers.

To start Jupyter Lab, open a terminal (Mac/Linux) or Command Prompt (Windows) and type the command:

```shell
jupyter lab
```

## Ollama
We will use Ollama to run large language models. It can be downloaded here:

https://ollama.com/download

Next, download the model that we will be using from a terminal (Mac/Linux) or Command Prompt (Windows) by typing the command:

```shell
ollama pull llama3.2:1b
```

## Data Sets
Datasets and example files are placed in the [episodes/data/](https://github.com/carpentries-incubator/Natural-language-processing/tree/main/episodes/data) directory.


### Word2Vec
Download Word2Vec models trained on 6 national Dutch newspaper data spanning a time period from 1950 to 1989 (Wevers, M., 2019). These models are available on [Zenodo](https://zenodo.org/records/3237380).


::: spoiler

### On Linux/macOs

```shell
python3 -m gensim.downloader --download word2vec-google-news-300 
```

:::

::: spoiler

### On Windows

```shell
py -m gensim.downloader --download word2vec-google-news-300
```

:::

### Spacy English
Download the [trained pipelines for English from Spacy](https://spacy.io/models/en/). To do so, open a terminal (Mac/Linux) or Command Prompt (Windows) and type the command:

::: spoiler

### On Linux/macOs

```shell
python3 -m spacy download en_core_web_sm
```

:::

::: spoiler

### On Windows

```shell
py -m spacy download en_core_web_sm
```

:::

