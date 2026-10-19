# Offensive Expression Detection in Urdu

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2610.15795-b31b1b.svg)](https://arxiv.org/abs/2610.15795)

Transformer-based **offensive expression detection in Urdu**, with a research focus on explainable predictions.

**[Main paper][paper] · [Related study][related] · [Dataset](data/) · [Notebooks](notebooks/) · [Citations](#citation)**

## Task

Given an Urdu comment, predict one of two labels:

| Label | Class |
| --- | --- |
| `0` | Non-Offensive |
| `1` | Offensive |

## Dataset

The paper reports **7,604 Facebook comments**, balanced across **3,802 examples per class**, with **5,475 training**, **608 validation**, and **1,521 test** instances [1].

See [`data/`](data/) for the available files and [`docs/`](docs/) for annotation guidelines and dataset documentation. Check the dataset version before reproducing the published benchmark.

## Models and published results

The repository model list includes **XLM-RoBERTa**, **TwHIN-BERT**, and **MuRIL**. The paper's benchmark is listed separately below: **Urdu-RoBERTa and XLM-RoBERTa are distinct models**. Match checkpoint names against the notebooks before comparing results.

| Model in paper [1] | Accuracy (%) | Macro F1 (%) |
| --- | ---: | ---: |
| Urdu-BERT | 82.18 | 82.16 |
| Urdu-RoBERTa | 85.34 | 85.31 |
| Urdu-Twhin-BERT | 86.91 | 86.92 |
| **MuRIL** | **87.05** | **87.06** |

*Published values from Table 11 [1], rather than results independently reproduced here. The paper does not establish a statistically significant ranking among the leading transformers.*
`

*Conceptual explanation workflow [1]. Attributions explain model behavior; they do not prove a word is inherently offensive.*

## Repository structure

| Path | Purpose |
| --- | --- |
| [`README.md`](README.md) | Project overview and references |
| [`requirements.txt`](requirements.txt) | Python dependencies |
| [`data/`](data/) | Training and evaluation data |
| [`docs/`](docs/) | Annotation guidelines and dataset documentation |
| [`notebooks/`](notebooks/) | Model training and evaluation notebooks |

## Installation and setup

```bash
git clone https://github.com/MSIMALIK/Offensive-Expression-detection-In-Urdu.git
cd Offensive-Expression-detection-In-Urdu

python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell: use this activation command instead
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Open a notebook from [`notebooks/`](notebooks/), select the environment's Python kernel, and check its dataset paths and model checkpoint before running the cells in order. GPU availability and memory requirements depend on the selected notebook.

For a reproducible experiment, record the dataset version, split identifiers, random seed, checkpoint, package versions, hyperparameters, and hardware alongside the metrics.

## Related work and limitations

Hussain et al. [2] study semantic features, word embeddings, and ensemble methods for Urdu offensive-language identification. Their reported scores use a different experimental setup and should not be directly ranked against this benchmark.

The main paper identifies lexical and topical shortcuts; performance on other platforms remains uncertain [1]. Use human review when applying predictions to moderation decisions.

## Content notice

The dataset may contain offensive, abusive, or sensitive Urdu expressions required for training and evaluating offensive-language detection models. Such content is included for research and academic purposes.

## Citation

Please cite the main paper when using this project and acknowledge the study when using its underlying corpus or methods.


```bibtex
@misc{malik2026explainable,
  title         = {An Explainable System for Offensive Content Detection in Nastaliq Urdu: Corpus Curation, Transformer Benchmarking, and Word-Level Attribution},
  author        = {Malik, Muhammad Shahid Iqbal},
  year          = {2026},
  eprint        = {2610.15795},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2610.15795},
  url           = {https://arxiv.org/abs/2610.15795}
}
```


## License

Project code is licensed under the **MIT License**. See [`LICENSE`](LICENSE). Consult the dataset documentation and linked papers for terms applying to those materials.

## Author

**Muhammad Shahid Iqbal Malik**  
GitHub: [@MSIMALIK](https://github.com/MSIMALIK)

[paper]: https://arxiv.org/abs/2610.15795
[related]: https://doi.org/10.7717/peerj-cs.1169
