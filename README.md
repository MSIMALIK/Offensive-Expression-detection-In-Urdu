# Offensive Expression Detection in Urdu

A Natural Language Processing project for detecting **offensive expressions in Urdu text** using transformer-based models.

The task is formulated as a binary classification problem:

* **0 - Non-Offensive**
* **1 - Offensive**

## Models

The repository includes experiments with:

* XLM-RoBERTa
* TwHIN-BERT
* MuRIL

## Repository Structure

```text
Offensive-Expression-detection-In-Urdu/
│
├── README.md
├── requirements.txt
├── data/
├── docs/
└── notebooks/
```

* `data/` - Dataset used for training and evaluation
* `docs/` - Annotation guidelines and dataset documentation
* `notebooks/` - Model training and evaluation notebooks

## Installation

```bash
git clone https://github.com/MSIMALIK/Offensive-Expression-detection-In-Urdu.git
cd Offensive-Expression-detection-In-Urdu
pip install -r requirements.txt
```


## Content Notice

The dataset may contain offensive or sensitive Urdu expressions because they are required for training and evaluating offensive-language detection models.

## Author

**Muhammad Shahid Iqbal Malik**

GitHub: [@MSIMALIK](https://github.com/MSIMALIK)
