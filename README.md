# Study 1 — Diachronic Word Embeddings for Sino-Japanese Relations

Word embedding models trained on **People's Daily** (1950–2019) and **Google Books Chinese** corpora, used to study how language about Japan has evolved over time in Chinese media and literature.

This repository accompanies:

> Hamamura, T., Kobayashi, T., Chen, Z., Chen, S. X., & Chan, C. S. (2025). Geopolitics in language: A psychological view of Sino-Japanese relations over time. *Journal of Social Issues*.
>
> OSF: <https://osf.io/t8mg9/>

## Data Sources & Downloads

### 1a) People's Daily Embeddings

Yearly Word2Vec SGNS models (1950–2019), stored as a Python dict keyed by `"year{YYYY}"`. Each value is a Gensim `Word2Vec` model (access vectors via `.wv`).

| Host | Link | Passcode |
|------|------|----------|
| BaiduYun | <https://pan.baidu.com/s/11kTFJDt7-HUIbrgHIZEKrQ?pwd=9u2s> | 9u2s |
| Dropbox | <https://www.dropbox.com/scl/fi/4zlt5j8voocesnhm0awzx/peoplesdaily_sgns_model_dict.pkl?rlkey=muzi1rqyq92bk7333jlpsylnj&st=4a3s5uw0&dl=0> | 2025 |

### 1b) Google Books Chinese Embeddings

Yearly Word2Vec SGNS models trained on the Chinese portion of Google Books Ngram corpus.

| Host | Link | Passcode |
|------|------|----------|
| BaiduYun | <https://pan.baidu.com/s/1WRpg1jmI7cIUu9C-Cfl8sA?pwd=5dku> | 5dku |
| Dropbox | <https://www.dropbox.com/scl/fi/lyruo80d9afblf4w3lvjc/google_sgns_model_dict.pkl?rlkey=r45qf30fwkzohstue5kcuhb6o&st=8iag14p4&dl=0> | 2025 |

## Installation

Requires Python >= 3.10.

```bash
pip install -r requirements.txt
```

## Quick Start

### Run WEFAT Analysis

1. Download a model pickle and place it at `Models/peoplesdaily_sgns_model_dict.pkl`.
2. Ensure `positive_words.csv` and `negative_words.csv` are in the project root (available from OSF).
3. Run:

```bash
python -m analysis.run_analysis
```

Outputs (CSVs and plots) are saved to `Outputs/`.

For the Google Books corpus:

```bash
python -m analysis.run_analysis --model Models/google_sgns_model_dict.pkl --label google
```

### Train Models from Scratch

**People's Daily:**

```bash
python -m training.train_peoplesdaily --input-dir /path/to/peoplesdaily_raw/
```

**Google Books (per year):**

```bash
python -m training.train_google 2019
```

## Project Structure

```
histwords-chinese/
├── README.md
├── requirements.txt
├── .gitignore
├── training/
│   ├── common.py              # Shared hyperparameters, callback, model builder
│   ├── train_peoplesdaily.py   # People's Daily training pipeline
│   └── train_google.py        # Google Books training pipeline
├── analysis/
│   ├── config.py              # Paths, plot settings, word lists
│   ├── wefat.py               # WEFAT computation (cosine sim, effect size)
│   ├── validate.py            # Vocabulary size plots & availability heatmaps
│   └── run_analysis.py        # Main: WEFAT over time + trend plots
├── Models/                    # (gitignored) model pickles
└── Outputs/                   # (gitignored) CSVs, PNGs
```

## Citation

```bibtex
@article{hamamura2025geopolitics,
  title={Geopolitics in language: A psychological view of Sino-Japanese relations over time},
  author={Hamamura, Takeshi and Kobayashi, Tetsuro and Chen, Zhicong and Chen, Sylvia Xiaohua and Chan, Chi Shun},
  journal={Journal of Social Issues},
  year={2025}
}
```

## License

- **Code**: MIT
- **Model bundles**: Research use only
- **Analysis outputs**: CC BY 4.0

## Ethics Note

Some terms in the "People" word list (e.g., 倭寇, 小日本, 日本鬼子) are historical slurs. They are included strictly for empirical measurement of how language usage has shifted over time, not as endorsement. See the paper for discussion.

## Contact

Zhicong Chen — zcchen@nus.edu.sg
