"""Train Word2Vec SGNS model on Google Books Chinese 5-grams for a single year.

Usage:
    python -m training.train_google 2019
"""

import logging
import os
import sys

import gensim
import numpy as np
import pandas as pd

from training.common import build_and_train_model


class MySentences:
    """Memory-friendly iterator over Google Books 5-gram files.

    Each line has format: ``gram_str\\tmatch_count\\tvolume_count``
    Yields the gram tokens ``match_count`` times.
    """

    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename, "r", encoding="utf-8"):
            gram_str, match_count, volume_count = line.strip("\n").split("\t")
            gram_list = [i.split("_")[0] for i in gram_str.split(" ")]
            match_count = int(match_count)
            for _ in np.arange(match_count):
                yield gram_list


def make_trim_function(allowed_words):
    """Create a vocabulary trimming function that keeps only allowed words."""

    def trim_function(word, count, min_count):
        if word in allowed_words:
            return gensim.utils.RULE_KEEP
        else:
            return gensim.utils.RULE_DISCARD

    return trim_function


def main():
    year = sys.argv[1].strip("\r")
    print(year)

    filename = f"./5grams_merge_by_year/{year}.txt"
    logfilename = f"word2vec_training_{year}_trim_iter_5.log"

    if os.path.exists(logfilename):
        os.remove(logfilename)

    logging.basicConfig(
        filename=logfilename,
        format="%(asctime)s:%(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    # Load vocabulary whitelist
    top_df = pd.read_csv("top_100000_frequent_words.csv")
    top_words = set(top_df["word"].values)

    trim_rule = make_trim_function(top_words)
    sentences = MySentences(filename)

    model = build_and_train_model(sentences, trim_rule=trim_rule)

    model.save(f"word2vec_{year}.model")
    print("Model saved!")


if __name__ == "__main__":
    main()
