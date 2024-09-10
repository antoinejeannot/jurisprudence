<p align="center"><img src="https://raw.githubusercontent.com/antoinejeannot/jurisprudence/artefacts/jurisprudence.svg" width=650></p>

[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg)](https://huggingface.co/datasets/antoinejeannot/jurisprudence)

# ✨ Jurisprudence, release v2024.09.10 🏛️

Jurisprudence is an open-source project that automates the collection and distribution of French legal decisions. It leverages the Judilibre API provided by the Cour de Cassation to:

- Fetch rulings from major French courts (Cour de Cassation, Cour d'Appel, Tribunal Judiciaire)
- Process and convert the data into easily accessible formats
- Publish & version updated datasets on Hugging Face every 3 days

This project aims to democratize access to legal information, enabling researchers, legal professionals, and the public to easily access and analyze French court decisions.
Whether you're conducting legal research, developing AI models, or simply interested in French jurisprudence, this project provides a valuable, open resource for exploring the French legal landscape.

## 📊 Exported Data

| Jurisdiction | Jurisprudences | Oldest | Latest | Tokens | JSONL (gzipped) | Parquet |
|--------------|----------------|--------|--------|--------|-----------------|---------|
| Cour d'Appel | 9,136 | 2024-08-01 | 2024-09-04 | 30,657,415 | [Download (26.06 MB)](https://huggingface.co/datasets/antoinejeannot/jurisprudence/resolve/main/cour_d_appel.jsonl.gz?download=true) | [Download (43.89 MB)](https://huggingface.co/datasets/antoinejeannot/jurisprudence/resolve/main/cour_d_appel.parquet?download=true) |
| Cour de Cassation | 689 | 2024-08-01 | 2024-09-05 | 792,251 | [Download (565.66 KB)](https://huggingface.co/datasets/antoinejeannot/jurisprudence/resolve/main/cour_de_cassation.jsonl.gz?download=true) | [Download (770.69 KB)](https://huggingface.co/datasets/antoinejeannot/jurisprudence/resolve/main/cour_de_cassation.parquet?download=true) |
| Tribunal Judiciaire | 15,309 | 2024-08-01 | 2024-08-13 | 51,662,019 | [Download (46.28 MB)](https://huggingface.co/datasets/antoinejeannot/jurisprudence/resolve/main/tribunal_judiciaire.jsonl.gz?download=true) | [Download (76.56 MB)](https://huggingface.co/datasets/antoinejeannot/jurisprudence/resolve/main/tribunal_judiciaire.parquet?download=true) |
| **Total** | **25,134** | **2024-08-01** | **2024-09-05** | **83,111,685** | **72.90 MB** | **121.20 MB** |

<i>Latest update date: 2024-09-10</i>

<i># Tokens are computed using GPT-4 tiktoken </i>

## 🤗 Hugging Face Dataset

The updated dataset is available at: https://huggingface.co/datasets/antoinejeannot/jurisprudence

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

<p align="center"><a href="https://www.etalab.gouv.fr/licence-ouverte-open-licence/"><img src="https://raw.githubusercontent.com/antoinejeannot/jurisprudence/artefacts/license.png" width=50 alt="license ouverte / open license"></a></p>