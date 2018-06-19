#!/usr/bin/env python
import numpy as np
from scipy.spatial.distance import cdist


def cosine_distance(row_patterns, col_patterns, e=1e-10):
    """A fast and naive implementation of cosine distance.

    Numerical stability issues might arise.

    :param row_patterns: A matrix with row vectors of dimensions MxN.
    :param col_patterns: A matrix of column vectors of dimensions NxK. The
    matrix can also be the transpose of row_patterns.
    :param e: A small constant to deal with numerical instability.
    :return: A matrix of MxK containing the pair-wise distance of all vectors
        given.
    """
    row_l2 = np.linalg.norm(row_patterns, axis=1)
    col_l2 = np.linalg.norm(col_patterns, axis=0)
    row_l2[row_l2 == 0] = e
    col_l2[col_l2 == 0] = e
    magnitude_mat = np.dot(row_l2[:, None], col_l2[None, :])
    similarity_mat = np.dot(row_patterns, col_patterns) / magnitude_mat
    return 1 - similarity_mat


def get_pr_rec_at(q_features, q_labels,
                  ret_features=None, ret_labels=None,
                  e=1e-32, mode="unspecified", metric="cosine",
                  singleton_samples="chop"):
    """Provides Precision recall matrices needed for computing mAP.

    :param q_features: A matrix with the the query row-vectors
    :param q_labels: A vector with the numerical labels corresponding to each
        query vector.
    :param ret_features: A matrix of the same width as q_features containing the
        row-vectors of the retrieval database.
    :param ret_labels: A vector with the numerical labels corresponding to each
        retrieval vector.
    :param e: A small constant, smaller that any meaningful distance between
        samples used to control sorting ambiguity. The constant is also used to
        deal with divisions by 0.
    :param mode: How to deal with ambiguous sorts. Must be one of
        ["unspecified", "optimistic", "pessimistic"]. Optimistic will return the
        most favorable sorting and pessimistic will return the leat favorable
        sorting.
    :param metric: The metric by which distances are computed between vectors.
        This must be one that scipy.spatial.distance.cdist accepts. Popular
        choices are ['cityblock', 'euclidean', 'cosine']. Look into cdist's
        documentation for details.
    :param singleton_samples: What to do with query labels that don't exist in
        the database. It must be one of ["chop", "perfect", "balanced", "null"].
        Selecting "chop" means that row's referring to queries that did not
        exist, will be omitted from the result matrix. Selecting "perfect" means
        that the non-retrievable rows will get always 100% precision and recall.
        Selecting "balanced" means that the non-retrievable rows will get always
        0% precision and 100% recall. Selecting "null" means that the
        non-retrievable rows will get always 0% precision and 0% recall.

    :return: Two matrices of the same size where the rows refer to query samples
        and columns refer to retrieval samples. The first one contains
        precition_@ rows while the second one contains recall_@.
    """
    assert singleton_samples in ["chop", "perfect", "balanced", "null"]
    assert mode in ["unspecified", "optimistic", "pessimistic"]
    if ret_labels is None:
        assert ret_features is None  # labels and features must be coupled
        ret_features = q_features
        ret_labels = q_labels
        leave_one_out = True
    else:
        leave_one_out = False
    dist_mat = cdist(q_features, ret_features, metric=metric).astype(
        np.double)
    correct = (q_labels.reshape(-1, 1) == ret_labels.reshape(1,
                                                             -1)).astype(
        "float")
    if mode == "optimistic":
        dist_mat -= correct * e
    elif mode == "pessimistic":
        dist_mat += correct * e

    nearest_idx = np.argsort(dist_mat, axis=1)
    correct_predictions = q_labels[:, None] == ret_labels[nearest_idx]
    # each row in correct_predictions contains the index of the retrieval
    # samples sorted by distance for each query.
    if leave_one_out:
        # In leave one out we retrive all samples from all samples
        # The nearest sample to each query will be the query its self.
        correct_predictions = correct_predictions[:, 1:]
    correct_predictions = correct_predictions.astype("double")
    correct_at = np.cumsum(correct_predictions, axis=1)
    recall_at = (correct_at + e) / (
            correct_predictions.sum(axis=1)[:, None] + e)
    precision_at = correct_at / np.cumsum(np.ones_like(recall_at), axis=1)

    if singleton_samples == "chop":
        exist_idx = np.nonzero(correct_predictions.sum(axis=1) > 0)[0]
        return precision_at[exist_idx, :], recall_at[exist_idx, :]
    elif singleton_samples == "perfect":
        non_existent = np.nonzero(correct_predictions.sum(axis=1) < 1)[0]
        precision_at[non_existent, :] = 1
        recall_at[non_existent, :] = 1
        return precision_at, recall_at
    elif singleton_samples == "null":
        non_existent = np.nonzero(correct_predictions.sum(axis=1) < 1)[0]
        precision_at[non_existent, :] = 0
        recall_at[non_existent, :] = 0
        return precision_at, recall_at
    elif singleton_samples == "balanced":
        non_existent = np.nonzero(correct_predictions.sum(axis=1) < 1)[0]
        precision_at[non_existent, :] = 1
        recall_at[non_existent, :] = 1
        return precision_at, recall_at
    else:
        raise Exception()


def get_map(q_features, q_labels, ret_features=None,
            ret_labels=None, mode="bounds", e=1e-20, metric="euclidean",
            singleton_samples="chop"):
    """Provides mean Average Precision mAP estimates.

    :param q_features: A matrix with the the query row-vectors.
    :param q_labels: A vector with the numerical labels corresponding to each
        query vector.
    :param ret_features: A matrix of the same width as q_features containing the
        row-vectors of the retrieval database.
    :param ret_labels: A vector with the numerical labels corresponding to each
        retrieval vector.
    :param e: A small constant, smaller that any meaningful distance between
        samples used to control sorting ambiguity. The constant is also used to
        deal with divisions by 0.
    :param mode: How to deal with ambiguous sorts. Must be one of
        ["unspecified", "optimistic", "pessimistic", "bounds"]. Optimistic will
        return the most favorable sorting and "pessimistic" will return the
        least favorable sorting. When mode is "bounds" than both
        "optimistic" and "pessimistic" are returned.
    :param metric: The metric by which distances are computed between vectors.
        This must be one that scipy.spatial.distance.cdist accepts. Popular
        choices are ['cityblock', 'euclidean', 'cosine']. Look into cdist's
        documentation for details.
    :param singleton_samples: What to do with query labels that don't exist in
        the database. It must be one of ["chop", "perfect", "balanced", "null"].
        Selecting "chop" means that row's referring to queries that did not
        exist, will be omitted from the result matrix. Selecting "perfect" means
        that the non-retrievable rows will get always 100% precision and recall.
        Selecting "balanced" means that the non-retrievable rows will get always
        0% precision and 100% recall. Selecting "null" means that the
        non-retrievable rows will get always 0% precision and 0% recall.

    :return: A floating point number containing the estimate of mAP. If mode was
        "bounds" than a tuple with the lower and upper bounds of mAP.
    """

    assert mode in ["unspecified", "optimistic", "pessimistic", "mean",
                    "bounds"]
    if mode in ["mean", "bounds"]:
        pr_at, rec_at = get_pr_rec_at(q_features,
                                      q_labels,
                                      ret_features,
                                      ret_labels,
                                      mode="optimistic",
                                      e=e,
                                      metric=metric, singleton_samples=singleton_samples)
        padded_rec_at = np.concatenate(
            (np.zeros([rec_at.shape[0], 1]), rec_at), axis=1)
        rec_changes_at = padded_rec_at[:, 1:] > padded_rec_at[:, :-1]

        average_pr = (pr_at * rec_changes_at).sum(
            axis=1) / rec_changes_at.sum(axis=1)
        optimist = average_pr.mean()

        pr_at, rec_at = get_pr_rec_at(q_features,
                                      q_labels,
                                      ret_features,
                                      ret_labels,
                                      mode="pessimistic",
                                      e=e,
                                      metric=metric, singleton_samples=singleton_samples)
        padded_rec_at = np.concatenate(
            (np.zeros([rec_at.shape[0], 1]), rec_at), axis=1)
        rec_changes_at = padded_rec_at[:, 1:] > padded_rec_at[:, :-1]
        average_pr = (pr_at * rec_changes_at).sum(
            axis=1) / rec_changes_at.sum(axis=1)
        pessimist = average_pr.mean()
        if mode == "mean":
            return (optimist + pessimist) / 2
        else:
            return optimist, pessimist
    else:
        pr_at, rec_at = get_pr_rec_at(q_features,
                                      q_labels,
                                      ret_features,
                                      ret_labels,
                                      mode=mode, e=e,
                                      metric=metric, singleton_samples=singleton_samples)
        padded_rec_at = np.concatenate(
            (np.zeros([rec_at.shape[0], 1]), rec_at), axis=1)
        rec_changes_at = padded_rec_at[:, 1:] > padded_rec_at[:, :-1]
        average_pr = (pr_at * rec_changes_at).sum(
            axis=1) / rec_changes_at.sum(axis=1)
        return average_pr.mean()
