"""Main analysis script: compute WEFAT scores over time and plot trends.

Usage:
    python -m analysis.run_analysis
    python -m analysis.run_analysis --model Models/google_sgns_model_dict.pkl --label google
"""

import argparse
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from analysis.config import (
    MODELS_DIR,
    OUTPUTS_DIR,
    NATION_STATE_WORDS,
    PEOPLE_WORDS,
)
from analysis.wefat import compute_wefat


def load_kvmodels(pickle_path):
    """Load a model pickle and extract KeyedVectors."""
    with open(pickle_path, "rb") as f:
        model_dict = pickle.load(f)

    kvmodel_dict = {key: model.wv for key, model in model_dict.items()}
    years = sorted(int(k.replace("year", "")) for k in kvmodel_dict.keys())
    print(f"Loaded yearly embeddings for {len(years)} years: {years[0]}-{years[-1]}")
    return kvmodel_dict, years


def count_years_present(words, kvmodel_dict, years):
    """Count how many years each word appears in the vocabulary."""
    counts = {}
    for w in words:
        c = sum(1 for y in years if w in kvmodel_dict[f"year{y}"])
        counts[w] = c
    return counts


def filter_terms(words, kvmodel_dict, years, min_years=20):
    """Keep only words present in at least *min_years* years."""
    counts = count_years_present(words, kvmodel_dict, years)
    return [w for w, c in counts.items() if c >= min_years]


def compute_wefat_over_time(kvmodel_dict, years, ns_terms, pe_terms,
                            positive_words, negative_words):
    """Compute WEFAT scores for all terms across all years."""
    records = []
    for category, term_list in [("Nation-State", ns_terms), ("People", pe_terms)]:
        for w in tqdm(term_list, desc=f"WEFAT {category}"):
            for y in years:
                kv = kvmodel_dict[f"year{y}"]
                score = compute_wefat(kv, w, positive_words, negative_words)
                records.append({
                    "year": y,
                    "word": w,
                    "category": category,
                    "wefat": score,
                })
    return pd.DataFrame(records)


def plot_wefat_trends(df_wefat, output_path):
    """Plot WEFAT trends with 95% CI bands."""
    plt.figure(figsize=(8, 7), dpi=200)
    for i, cat in enumerate(["Nation-State", "People"], start=1):
        ax = plt.subplot(2, 1, i)
        sub = df_wefat[df_wefat["category"] == cat]
        sns.lineplot(data=sub, x="year", y="wefat", errorbar=("ci", 95))
        ax.axhline(0, ls="--", color="gray")
        ax.set_title(f"WEFAT Trends \u2014 {cat} Terms")
        ax.set_ylim(-1, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run WEFAT analysis")
    parser.add_argument(
        "--model",
        default=str(MODELS_DIR / "peoplesdaily_sgns_model_dict.pkl"),
        help="Path to model pickle",
    )
    parser.add_argument(
        "--positive-words",
        default="positive_words.csv",
        help="CSV file with positive sentiment words (column 'x')",
    )
    parser.add_argument(
        "--negative-words",
        default="negative_words.csv",
        help="CSV file with negative sentiment words (column 'x')",
    )
    parser.add_argument(
        "--min-years",
        type=int,
        default=20,
        help="Minimum years a term must be present (default: 20)",
    )
    parser.add_argument(
        "--label",
        default="peoplesdaily",
        help="Label used in output filenames (default: peoplesdaily)",
    )
    args = parser.parse_args()

    # Load models
    kvmodel_dict, years = load_kvmodels(args.model)

    # Load sentiment lexicons
    positive_words = pd.read_csv(args.positive_words)["x"].dropna().astype(str).tolist()
    negative_words = pd.read_csv(args.negative_words)["x"].dropna().astype(str).tolist()

    # Filter terms by minimum presence
    ns_terms = filter_terms(NATION_STATE_WORDS, kvmodel_dict, years, args.min_years)
    pe_terms = filter_terms(PEOPLE_WORDS, kvmodel_dict, years, args.min_years)
    print(f"Nation-State terms retained: {len(ns_terms)} / {len(NATION_STATE_WORDS)}")
    print(f"People terms retained: {len(pe_terms)} / {len(PEOPLE_WORDS)}")

    # Compute WEFAT
    df_wefat = compute_wefat_over_time(
        kvmodel_dict, years, ns_terms, pe_terms, positive_words, negative_words
    )

    csv_path = OUTPUTS_DIR / f"WEFAT_{args.label}_by_category.csv"
    df_wefat.to_csv(csv_path, index=False)
    print(f"Saved WEFAT scores to {csv_path}")

    # Plot
    plot_path = OUTPUTS_DIR / f"wefat_trends_{args.label}_by_category.png"
    plot_wefat_trends(df_wefat, plot_path)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
