<div align="center"><img src="https://github.com/stanfordnlp/bclm/raw/dev/images/bclm-logo.png" height="100px"/></div>

<h2 align="center">bclm: A Python NLP Library for Many Human Languages</h2>

<div align="center">
    <a href="https://travis-ci.com/stanfordnlp/bclm">
        <img alt="Travis Status" src="https://travis-ci.com/stanfordnlp/bclm.svg?token=RPNzRzNDQRoq2x3J2juj&branch=master">
    </a>
    <a href="https://pypi.org/project/bclm/">
        <img alt="PyPI Version" src="https://img.shields.io/pypi/v/bclm?color=blue">
    </a>
    <a href="https://anaconda.org/stanfordnlp/bclm">
        <img alt="Conda Versions" src="https://img.shields.io/conda/vn/stanfordnlp/bclm?color=blue&label=conda">
    </a>
    <a href="https://pypi.org/project/bclm/">
        <img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/bclm?colorB=blue">
    </a>
</div>

The Stanford NLP Group's official Python NLP library. It contains support for running various accurate natural language processing tools on 60+ languages and for accessing the Java Stanford CoreNLP software from Python. For detailed information please visit our [official website](https://stanfordnlp.github.io/bclm/).

🔥 &nbsp;A new collection of **biomedical** and **clinical** English model packages are now available, offering seamless experience for syntactic analysis and named entity recognition (NER) from biomedical literature text and clinical notes. For more information, check out our [Biomedical models documentation page](https://stanfordnlp.github.io/bclm/biomed.html).

### References

If you use this library in your research, please kindly cite our [ACL2020 bclm system demo paper](https://arxiv.org/abs/2003.07082):

```bibtex
@inproceedings{qi2020bclm,
    title={bclm: A {Python} Natural Language Processing Toolkit for Many Human Languages},
    author={Qi, Peng and Zhang, Yuhao and Zhang, Yuhui and Bolton, Jason and Manning, Christopher D.},
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
    year={2020}
}
```

If you use our biomedical and clinical models, please also cite our [bclm Biomedical Models description paper](https://arxiv.org/abs/2007.14640):

```bibtex
@article{zhang2021biomedical,
    author = {Zhang, Yuhao and Zhang, Yuhui and Qi, Peng and Manning, Christopher D and Langlotz, Curtis P},
    title = {Biomedical and clinical {E}nglish model packages for the {S}tanza {P}ython {NLP} library},
    journal = {Journal of the American Medical Informatics Association},
    year = {2021},
    month = {06},
    issn = {1527-974X}
}
```

The PyTorch implementation of the neural pipeline in this repository is due to [Peng Qi](http://qipeng.me) (@qipeng), [Yuhao Zhang](http://yuhao.im) (@yuhaozhang), and [Yuhui Zhang](https://cs.stanford.edu/~yuhuiz/) (@yuhui-zh15), with help from [Jason Bolton](mailto:jebolton@stanford.edu) (@j38), [Tim Dozat](https://web.stanford.edu/~tdozat/) (@tdozat) and [John Bauer](https://www.linkedin.com/in/john-bauer-b3883b60/) (@AngledLuffa). Maintenance of this repo is currently led by [John Bauer](https://www.linkedin.com/in/john-bauer-b3883b60/).

If you use the CoreNLP software through bclm, please cite the CoreNLP software package and the respective modules as described [here](https://stanfordnlp.github.io/CoreNLP/#citing-stanford-corenlp-in-papers) ("Citing Stanford CoreNLP in papers"). The CoreNLP client is mostly written by [Arun Chaganty](http://arun.chagantys.org/), and [Jason Bolton](mailto:jebolton@stanford.edu) spearheaded merging the two projects together.

## Issues and Usage Q&A

To ask questions, report issues or request features 🤔, please use the [GitHub Issue Tracker](https://github.com/stanfordnlp/bclm/issues). Before creating a new issue, please make sure to search for existing issues that may solve your problem, or visit the [Frequently Asked Questions (FAQ) page](https://stanfordnlp.github.io/bclm/faq.html) on our website.

## Contributing to bclm

We welcome community contributions to bclm in the form of bugfixes 🛠️ and enhancements 💡! If you want to contribute, please first read [our contribution guideline](CONTRIBUTING.md).

## Installation

### pip

bclm supports Python 3.6 or later. We recommend that you install bclm via [pip](https://pip.pypa.io/en/stable/installing/), the Python package manager. To install, simply run:
```bash
pip install bclm
```
This should also help resolve all of the dependencies of bclm, for instance [PyTorch](https://pytorch.org/) 1.3.0 or above.

If you currently have a previous version of `bclm` installed, use:
```bash
pip install bclm -U
```

### Anaconda

To install bclm via Anaconda, use the following conda command:

```bash
conda install -c stanfordnlp bclm
```

Note that for now installing bclm via Anaconda does not work for Python 3.8. For Python 3.8 please use pip installation.

### From Source

Alternatively, you can also install from source of this git repository, which will give you more flexibility in developing on top of bclm. For this option, run
```bash
git clone https://github.com/stanfordnlp/bclm.git
cd bclm
pip install -e .
```

## Running bclm

### Getting Started with the neural pipeline

To run your first bclm pipeline, simply following these steps in your Python interactive interpreter:

```python
>>> import bclm
>>> bclm.download('en')       # This downloads the English models for the neural pipeline
>>> nlp = bclm.Pipeline('en') # This sets up a default neural pipeline in English
>>> doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
>>> doc.sentences[0].print_dependencies()
```

If you encounter `requests.exceptions.ConnectionError`, please try to use a proxy:

```python
>>> import bclm
>>> proxies = {'http': 'http://ip:port', 'https': 'http://ip:port'}
>>> bclm.download('en', proxies=proxies)  # This downloads the English models for the neural pipeline
>>> nlp = bclm.Pipeline('en')             # This sets up a default neural pipeline in English
>>> doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
>>> doc.sentences[0].print_dependencies()
```

The last command will print out the words in the first sentence in the input string (or [`Document`](https://stanfordnlp.github.io/bclm/data_objects.html#document), as it is represented in bclm), as well as the indices for the word that governs it in the Universal Dependencies parse of that sentence (its "head"), along with the dependency relation between the words. The output should look like:

```
('Barack', '4', 'nsubj:pass')
('Obama', '1', 'flat')
('was', '4', 'aux:pass')
('born', '0', 'root')
('in', '6', 'case')
('Hawaii', '4', 'obl')
('.', '4', 'punct')
```

See [our getting started guide](https://stanfordnlp.github.io/bclm/installation_usage.html#getting-started) for more details.

### Accessing Java Stanford CoreNLP software

Aside from the neural pipeline, this package also includes an official wrapper for accessing the Java Stanford CoreNLP software with Python code.

There are a few initial setup steps.

* Download [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) and models for the language you wish to use
* Put the model jars in the distribution folder
* Tell the Python code where Stanford CoreNLP is located by setting the `CORENLP_HOME` environment variable (e.g., in *nix): `export CORENLP_HOME=/path/to/stanford-corenlp-4.1.0`

We provide [comprehensive examples](https://stanfordnlp.github.io/bclm/corenlp_client.html) in our documentation that show how one can use CoreNLP through bclm and extract various annotations from it.

### Online Colab Notebooks

To get your started, we also provide interactive Jupyter notebooks in the `demo` folder. You can also open these notebooks and run them interactively on [Google Colab](https://colab.research.google.com). To view all available notebooks, follow these steps:

* Go to the [Google Colab website](https://colab.research.google.com)
* Navigate to `File` -> `Open notebook`, and choose `GitHub` in the pop-up menu
* Note that you do **not** need to give Colab access permission to your github account
* Type `stanfordnlp/bclm` in the search bar, and click enter

### Trained Models for the Neural Pipeline

We currently provide models for all of the [Universal Dependencies](https://universaldependencies.org/) treebanks v2.5, as well as NER models for a few widely-spoken languages. You can find instructions for downloading and using these models [here](https://stanfordnlp.github.io/bclm/models.html).

### Batching To Maximize Pipeline Speed

To maximize speed performance, it is essential to run the pipeline on batches of documents. Running a for loop on one sentence at a time will be very slow. The best approach at this time is to concatenate documents together, with each document separated by a blank line (i.e., two line breaks `\n\n`).  The tokenizer will recognize blank lines as sentence breaks. We are actively working on improving multi-document processing.

## Training your own neural pipelines

All neural modules in this library can be trained with your own data. The tokenizer, the multi-word token (MWT) expander, the POS/morphological features tagger, the lemmatizer and the dependency parser require [CoNLL-U](https://universaldependencies.org/format.html) formatted data, while the NER model requires the BIOES format. Currently, we do not support model training via the `Pipeline` interface. Therefore, to train your own models, you need to clone this git repository and run training from the source.

For detailed step-by-step guidance on how to train and evaluate your own models, please visit our [training documentation](https://stanfordnlp.github.io/bclm/training.html).

## LICENSE

bclm is released under the Apache License, Version 2.0. See the [LICENSE](https://github.com/stanfordnlp/bclm/blob/master/LICENSE) file for more details.
