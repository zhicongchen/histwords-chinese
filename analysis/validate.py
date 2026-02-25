"""Validation utilities: vocabulary sizes and word availability heatmaps."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from analysis.config import OUTPUTS_DIR, NATION_STATE_WORDS, PEOPLE_WORDS


def plot_vocab_sizes(kvmodel_dict, years):
    """Plot vocabulary sizes over time and save CSV + PNG.

    Parameters
    ----------
    kvmodel_dict : dict[str, KeyedVectors]
        Keyed by ``"year{YYYY}"``.
    years : list[int]
        Sorted list of years.
    """
    vocab_sizes = []
    for y in years:
        key = f"year{y}"
        vocab_sizes.append({"year": y, "size_of_vocab": len(kvmodel_dict[key])})

    df_vocab = pd.DataFrame(vocab_sizes)
    df_vocab.to_csv(OUTPUTS_DIR / "validation_vocab_sizes.csv", index=False)

    plt.figure(figsize=(7, 3))
    sns.lineplot(data=df_vocab, x="year", y="size_of_vocab")
    plt.title("People's Daily Embeddings: Vocabulary Size Over Time")
    plt.xlabel("")
    plt.ylabel("Vocab Size")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "validation_vocab_sizes.png", dpi=200)
    plt.show()

    return df_vocab


def _availability_matrix(words, kvmodel_dict, years):
    """Build a binary matrix of word presence across years."""
    data = {}
    for y in years:
        key = f"year{y}"
        kv = kvmodel_dict[key]
        data[y] = [1 if w in kv else 0 for w in words]
    return pd.DataFrame(data, index=words)


def plot_availability_heatmaps(kvmodel_dict, years):
    """Plot availability heatmaps for nation-state and people terms.

    Parameters
    ----------
    kvmodel_dict : dict[str, KeyedVectors]
        Keyed by ``"year{YYYY}"``.
    years : list[int]
        Sorted list of years.
    """
    df_ns = _availability_matrix(NATION_STATE_WORDS, kvmodel_dict, years)
    df_pe = _availability_matrix(PEOPLE_WORDS, kvmodel_dict, years)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), dpi=200)
    sns.heatmap(df_ns, cmap="Blues", cbar=False, ax=axes[0])
    axes[0].set_title("Availability: Nation-State Terms")
    sns.heatmap(df_pe, cmap="Blues", cbar=False, ax=axes[1])
    axes[1].set_title("Availability: People Terms")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "availability_heatmaps.png", dpi=200)
    plt.show()

    return df_ns, df_pe
