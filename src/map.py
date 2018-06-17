#!/usr/bin/env python
import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

def cosine_distance(row_patterns,col_patterns,e=1e-10):
    row_l2 = np.linalg.norm(row_patterns, axis=1)
    col_l2 = np.linalg.norm(col_patterns, axis=0)
    row_l2[row_l2==0] = e
    col_l2[col_l2 == 0] = e
    magnitude_mat=np.dot(row_l2[:,None],col_l2[None,:])
    similarity_mat=np.dot(row_patterns,col_patterns)/magnitude_mat
    return 1-similarity_mat

def get_query_precision_recall_at(query_features, query_labels, retrival_features=None, retrieval_labels=None,e=1e-32,mode="unspecified",metric="cosine",singleton_samples="chop"):
    assert singleton_samples in ["chop", "perfect"]
    assert mode in ["unspecified","optimistic","pessimistic"]
    if retrieval_labels is None:
        assert retrival_features is None # labels and features must be coupled
        retrival_features=query_features
        retrieval_labels=query_labels
        leave_one_out=True
    else:
        leave_one_out=False
    dist_mat=cdist(query_features, retrival_features,metric=metric).astype(np.double)
    correct=(query_labels.reshape(-1, 1) == retrieval_labels.reshape(1, -1)).astype("float")
    if mode == "optimistic":
        dist_mat-=correct*e
    elif mode == "pessimistic":
        dist_mat+=correct*e

    nearest_idx=np.argsort(dist_mat, axis=1)
    correct_predictions=query_labels[:, None] == retrieval_labels[nearest_idx]
    # each row in correct_predictions contains the index of the retrieval
    # samples sorted by distance for each query.
    if leave_one_out:
        # In leave one out we retrive all samples from all samples
        # The nearest sample to each query will be the query its self.
        correct_predictions=correct_predictions[:,1:]
    correct_predictions=correct_predictions.astype("double")
    correct_at=np.cumsum(correct_predictions,axis=1)
    recall_at=(correct_at+e)/(correct_predictions.sum(axis=1)[:,None]+e)
    precision_at = correct_at/np.cumsum(np.ones_like(recall_at),axis=1)


    if singleton_samples=="chop":
        exist_idx = np.nonzero(correct_predictions.sum(axis=1) > 0)[0]
        return precision_at[exist_idx,:], recall_at[exist_idx,:]
    elif singleton_samples=="perfect":
        non_existent=np.nonzero(correct_predictions.sum(axis=1) < 1)[0]
        precision_at[non_existent,:]=1
        recall_at[non_existent,:]=1
        return precision_at, recall_at
    else:
        raise Exception()


def get_map(query_features, query_labels, retrival_features=None, retrieval_labels=None,mode="bounds",e=1e-20,metric="euclidean"):
    assert mode in  ["unspecified","optimistic","pessimistic","mean","bounds"]
    if mode in ["mean","bounds"]:
        precision_at, recall_at = get_query_precision_recall_at(query_features,
                                                                query_labels,
                                                                retrival_features,
                                                                retrieval_labels,
                                                                mode="optimistic",e=e,metric=metric)
        padded_recall_at=np.concatenate((np.zeros([recall_at.shape[0],1]),recall_at),axis=1)
        recall_changes_at=padded_recall_at[:,1:]>padded_recall_at[:,:-1]

        average_precison=(precision_at*recall_changes_at).sum(axis=1)/recall_changes_at.sum(axis=1)
        optimist= average_precison.mean()

        precision_at, recall_at = get_query_precision_recall_at(query_features,
                                                                query_labels,
                                                                retrival_features,
                                                                retrieval_labels,
                                                                mode="pessimistic",e=e,metric=metric)
        padded_recall_at=np.concatenate((np.zeros([recall_at.shape[0],1]),recall_at),axis=1)
        recall_changes_at=padded_recall_at[:,1:]>padded_recall_at[:,:-1]
        average_precison=(precision_at*recall_changes_at).sum(axis=1)/recall_changes_at.sum(axis=1)
        pessimist= average_precison.mean()
        if mode == "mean":
            return (optimist+pessimist)/2
        else:
            return (optimist,pessimist)
    else:
        precision_at, recall_at=get_query_precision_recall_at(query_features, query_labels, retrival_features, retrieval_labels,mode=mode,e=e,metric=metric)
        padded_recall_at=np.concatenate((np.zeros([recall_at.shape[0],1]),recall_at),axis=1)
        recall_changes_at=padded_recall_at[:,1:]>padded_recall_at[:,:-1]
        average_precison=(precision_at*recall_changes_at).sum(axis=1)/recall_changes_at.sum(axis=1)
        return average_precison.mean()

def get_accuracy(query_features, query_labels, retrival_features=None, retrieval_labels=None,mode="bounds",e=1e-20,metric="cosine"):
    assert mode in  ["unspecified","optimistic","pessimistic","mean","bounds"]
    if mode in ["mean","bounds"]:
        precision_at, recall_at = get_query_precision_recall_at(query_features,
                                                                query_labels,
                                                                retrival_features,
                                                                retrieval_labels,
                                                                mode="optimistic",e=e,metric=metric)
        optimist=precision_at[:,0].mean()

        precision_at, recall_at = get_query_precision_recall_at(query_features,
                                                                query_labels,
                                                                retrival_features,
                                                                retrieval_labels,
                                                                mode="pessimistic",e=e,metric=metric)
        pessimist= precision_at[:,0].mean()
        if mode == "mean":
            return (optimist+pessimist)/2
        else:
            return (optimist,pessimist)
    else:
        precision_at, recall_at=get_query_precision_recall_at(query_features, query_labels, retrival_features, retrieval_labels,mode=mode,e=e,metric=metric)
        return precision_at[:,0].mean()

