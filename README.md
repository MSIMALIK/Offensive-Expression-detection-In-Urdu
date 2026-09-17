# Offensive Expression Detection in Urdu

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C.svg)](https://pytorch.org/)

A Natural Language Processing project for detecting **offensive expressions in Urdu text** using transformer-based deep learning models.

The task is formulated as a binary classification problem:

* **0 - Non-Offensive**
* **1 - Offensive**


## 🏗️ Models & Architectures

This repository evaluates and fine-tunes fine-grained multilingual transformer architectures for Urdu offensive language understanding:

* **XLM-RoBERTa**
* **TwHIN-BERT**
* **MuRIL**

---

## 📁 Repository Structure

```text
Offensive-Expression-detection-In-Urdu/
│
├── README.md
├── requirements.txt
├── data/           # Dataset used for training and evaluation
├── docs/           # Annotation guidelines and dataset documentation
└── notebooks/      # Model training and evaluation notebooks
```

* `data/` - Dataset used for training and evaluation
* `docs/` - Annotation guidelines and dataset documentation
* `notebooks/` - Model training and evaluation notebooks

---

## 🚀 Installation & Setup

Clone the repository and set up the dependencies:

```bash
git clone https://github.com/MSIMALIK/Offensive-Expression-detection-In-Urdu.git
cd Offensive-Expression-detection-In-Urdu
pip install -r requirements.txt
```


## ⚠️ Content Notice

The dataset may contain offensive, abusive, or sensitive Urdu expressions because such examples are required for training and evaluating offensive-language detection models.

The content is included solely for research and academic purposes.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 👤 Author

**Muhammad Shahid Iqbal Malik**  
GitHub: [@MSIMALIK](https://github.com/MSIMALIK)
