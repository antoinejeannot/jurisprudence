<p align="center"><img src="https://raw.githubusercontent.com/antoinejeannot/jurisprudence/artefacts/jurisprudence.svg" width=650></p>

[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg)](https://huggingface.co/datasets/antoinejeannot/jurisprudence)

# ✨ Jurisprudence, release v2024.09.07 🏛️

Jurisprudence is an open-source project that automates the collection and distribution of French legal decisions. It leverages the Judilibre API provided by the Cour de Cassation to:

- Fetch rulings from major French courts (Cour de Cassation, Cour d'Appel, Tribunal Judiciaire)
- Process and convert the data into easily accessible formats
- Publish & version updated datasets on Hugging Face every 3 days

This project aims to democratize access to legal information, enabling researchers, legal professionals, and the public to easily access and analyze French court decisions.
Whether you're conducting legal research, developing AI models, or simply interested in French jurisprudence, this project provides a valuable, open resource for exploring the French legal landscape.

## 📊 Exported Data

| Jurisdiction        | Size         | Jurisprudences | Oldest     | Latest     | Tokens              | Download                                                                                                                       |
| ------------------- | ------------ | -------------- | ---------- | ---------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Cour d'Appel        | 7.58 GB      | 375,549        | 1996-03-25 | 2024-08-13 | 1,884,985,718 +     | [Download](https://huggingface.co/datasets/antoinejeannot/jurisprudence/resolve/main/cour_d_appel.tar.gz?download=true)        |
| Tribunal Judiciaire | 830.98 MB    | 56,530         | 2023-12-14 | 2024-08-13 | 204,326,755 +       | [Download](https://huggingface.co/datasets/antoinejeannot/jurisprudence/resolve/main/tribunal_judiciaire.tar.gz?download=true) |
| Cour de Cassation   | 4.79 GB      | 533,827        | 1860-08-01 | 2024-08-07 | 1,103,124,295 +     | [Download](https://huggingface.co/datasets/antoinejeannot/jurisprudence/resolve/main/cour_de_cassation.tar.gz?download=true)   |
| **Total**           | **13.18 GB** | **965,906**    | -          | -          | **3,192,436,768 +** | -                                                                                                                              |

<i>Latest update date: 2024-09-07</i>

<i># Tokens are computed GPT-4 using tiktoken </i>

## 🤗 Hugging Face Dataset

The updated dataset is available at: <https://huggingface.co/datasets/antoinejeannot/jurisprudence>

## 🪪 Citing & Authors

If you use this code in your research, please use the following BibTeX entry:

```bibtex
@misc{antoinejeannot2024,
author = {Jeannot Antoine and {Cour de Cassation}},
title = {Jurisprudence},
year = {2024},
howpublished = {\url{https://github.com/antoinejeannot/jurisprudence}},
note = {Data source: API Judilibre, \url{https://www.data.gouv.fr/en/datasets/api-judilibre/}}
}
```

This project relies on the [Judilibre API par la Cour de Cassation](https://www.data.gouv.fr/en/datasets/api-judilibre/), which is made available under the Open License 2.0 (Licence Ouverte 2.0)

It scans the API every 3 days at 2am UTC and exports its data in various formats to Hugging Face, without any fundamental transformation but conversions.

<p align="center"><a href="https://www.etalab.gouv.fr/licence-ouverte-open-licence/"><img src="https://raw.githubusercontent.com/antoinejeannot/jurisprudence/artefacts/license.png" width=50  alt="license ouverte / open license"></a></p>
