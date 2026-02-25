"""Shared training utilities for Word2Vec SGNS models."""

import logging
import multiprocessing
from time import time

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

HYPERPARAMS = {
    "vector_size": 300,
    "window": 4,
    "min_count": 5,
    "sg": 1,
    "hs": 0,
    "negative": 5,
    "ns_exponent": 0.75,
    "sample": 1e-5,
    "epochs": 5,
}


class EpochLossCallback(CallbackAny2Vec):
    """Callback to print loss after each epoch."""

    def __init__(self):
        self.epoch = 1
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        logging.info(f"Loss after epoch {self.epoch}: {loss_now}")
        self.epoch += 1


def build_and_train_model(sentences, trim_rule=None, seed=None):
    """Build vocabulary and train a Word2Vec SGNS model.

    Parameters
    ----------
    sentences : iterable
        Iterable of tokenized sentences (lists of strings).
    trim_rule : callable, optional
        Custom vocabulary trimming function for build_vocab().
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    gensim.models.Word2Vec
        The trained model.
    """
    n_cores = multiprocessing.cpu_count()

    params = {**HYPERPARAMS, "workers": n_cores - 1}
    if seed is not None:
        params["seed"] = seed

    model = Word2Vec(**params)

    logging.info("Hyperparameters: " + str(params))

    print("Building Vocabulary ...")
    t = time()
    build_kwargs = {"corpus_iterable": sentences, "progress_per": 1_000_000}
    if trim_rule is not None:
        build_kwargs["trim_rule"] = trim_rule
    model.build_vocab(**build_kwargs)
    logging.info(f"Time to build vocab: {round((time() - t) / 60, 2)} mins")

    print("Training ...")
    t = time()
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=model.epochs,
        compute_loss=True,
        callbacks=[EpochLossCallback()],
    )
    logging.info(f"Time to train the model: {round((time() - t) / 60, 2)} mins")

    return model
