"""WEFAT (Word Embedding Fairness Association Test) utilities."""

import numpy as np
from scipy.spatial.distance import cosine


def cosine_sim(kv, a, b):
    """Compute cosine similarity between two words in a KeyedVectors model.

    Returns NaN if either word is missing from the vocabulary.
    """
    try:
        return 1 - cosine(kv[a], kv[b])
    except Exception:
        return np.nan


def compute_wefat(kv, w, A, B):
    """Compute the WEFAT effect size for target word *w*.

    Parameters
    ----------
    kv : gensim.models.KeyedVectors
        Word vectors.
    w : str
        Target word.
    A : list[str]
        Attribute word set A (e.g., positive sentiment words).
    B : list[str]
        Attribute word set B (e.g., negative sentiment words).

    Returns
    -------
    float
        ``(mean(sim_A) - mean(sim_B)) / std(sim_A + sim_B)``, or NaN if
        either set produces no valid similarities.
    """
    sA = [cosine_sim(kv, w, a) for a in A]
    sB = [cosine_sim(kv, w, b) for b in B]

    sA = [x for x in sA if not np.isnan(x)]
    sB = [x for x in sB if not np.isnan(x)]

    if len(sA) == 0 or len(sB) == 0:
        return np.nan

    pooled = sA + sB
    denom = np.std(pooled) if len(pooled) > 1 else np.nan

    if denom is None or np.isnan(denom) or denom == 0:
        return np.nan

    return (np.mean(sA) - np.mean(sB)) / denom
