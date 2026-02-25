"""Train yearly Word2Vec SGNS models on People's Daily corpus (1950-2019).

Usage:
    python -m training.train_peoplesdaily --input-dir F:/Datasets/Embeddings/peoplesdaily/peoplesdaily_raw/
"""

import argparse
import logging
import pickle
import re

import jieba
import numpy as np
from tqdm import tqdm

from training.common import build_and_train_model

logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def load_and_clean(filepath):
    """Read a yearly text file and return cleaned text."""
    with open(filepath, "r", encoding="utf-8") as fp:
        data = fp.read()

    data = (
        data.replace("\u3000", " ")
        .replace("\xa0", " ")
        .replace("</s>", " ")
        .replace(" ", "")
    )
    data = re.sub(r"[a-zA-Z0-9.*‧\\\-卩Ė]+", "", data)
    return data


def segment_sentences(text):
    """Split text on sentence boundaries and segment with jieba."""
    sentences = []
    for sentence in tqdm(text.split("。"), desc="Segmenting", leave=False):
        tokens = jieba.lcut(sentence, cut_all=False)
        if tokens:
            sentences.append(tokens)
    return sentences


def main():
    parser = argparse.ArgumentParser(description="Train People's Daily embeddings")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing yearly text files (1950.txt ... 2019.txt)",
    )
    parser.add_argument(
        "--output",
        default="Models/peoplesdaily_sgns_model_dict.pkl",
        help="Output pickle path (default: Models/peoplesdaily_sgns_model_dict.pkl)",
    )
    parser.add_argument("--start-year", type=int, default=1950)
    parser.add_argument("--end-year", type=int, default=2020)
    args = parser.parse_args()

    model_dict = {}

    for year in np.arange(args.start_year, args.end_year):
        print(f"\n{'='*40}")
        print(f"Year {year}")
        print(f"{'='*40}")

        filepath = f"{args.input_dir}{year}.txt"
        text = load_and_clean(filepath)

        print("Segmenting ...")
        sentences = segment_sentences(text)

        print("Training ...")
        model = build_and_train_model(sentences, seed=42)

        key = f"year{year}"
        model_dict[key] = model

    with open(args.output, "wb") as fp:
        pickle.dump(model_dict, fp)

    print(f"\nAll models saved to {args.output}")


if __name__ == "__main__":
    main()
