# -*- coding: utf-8 -*-
"""
credit to:
Algan, G., & Ulusoy, I. (2020). Label noise types and their effects on deep learning. arXiv preprint arXiv:2003.10471.
https://github.com/gorkemalgan/corrupting_labels_with_distillation
"""


import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils.multiclass import unique_labels
from numpy.testing import assert_array_almost_equal
from sklearn.metrics import confusion_matrix
import collections

NOISETYPES = set(['class-dependent', 'uniform', 'locally-concentrated'])


def get_noisy_labels(datasetName, x_train, y_train_int, x_test, y_test_int, num_classes, datasize, noise_type, noise_ratio, classStr):
    assert noise_type in NOISETYPES, "invalid noise type"
    probs = None

    if noise_ratio > 1 and noise_ratio < 100:
        noise_ratio = noise_ratio / 100.
  
    if noise_type == 'none':
        y_train_noisy = y_train_int
        y_test_noisy =  y_test_int
    elif noise_type == 'uniform':
        P = cm_uniform(num_classes, noise_ratio)
        y_train_noisy, _ = noise_cm(y_train_int, P)
    elif noise_type == 'class-dependent':
        test_logits = np.load(datasetName + '_logits_and_preds/'+ datasetName + '_'+ datasize + '_' + str(num_classes) + 'cls' + classStr + '_soft_pred_test.npy')
        P = cm_model_prediction(x_test, y_test_int, test_logits, noise_ratio)
        y_train_noisy, _ = noise_cm(y_train_int, P)
    elif noise_type == 'locally-concentrated':

        train_logits = np.load(datasetName + '_logits_and_preds/'+ datasetName + '_'+ datasize + '_' + str(num_classes) + 'cls' + classStr +  '_logits_train.npy')
        y_train_noisy = noise_xy_localized(train_logits, y_train_int, noise_ratio)
     
    
    return y_train_noisy, probs





def get_sorted_idx(probs, labels, class_id=None):
    '''
    Returns indices of samples beloning to class_id. Indices are sorted according to probs. First one is least confidently class_id
    and last one is most confidently class_id.
    If class_id is None, then just sorts all samples according to given probability
    '''
    # indices of samples which belong to class i
    if class_id is None:
        idx_i = labels
    else:
        idx_i = np.where(labels == class_id)[0]
    # order according to probabilities of confidence. First one is least likely for class i and last one is most likely
    idx_tmp = np.argsort(probs[idx_i])
    idx_sorted = idx_i[idx_tmp]

    # make sure sorted idx indeed belongs to given class
    if class_id is not None:
        assert np.sum(labels[idx_sorted] == class_id) == len(idx_sorted)
    # make sure idx are indeed sorted
    assert np.sum(np.diff(probs[idx_sorted])<0) == 0

    return idx_sorted

def noise_softmax(x_train, y_train_int, probs, noise_ratio):
  y_noisy = np.copy(y_train_int)
  num_classes = len(unique_labels(y_train_int))
  num_noisy = int(x_train.shape[0]*noise_ratio)
  # find maximum noise ratio and change original noise ratio if necessary
  non_zeros = np.count_nonzero(probs, axis=1)
  num_multiple_guess = np.sum(non_zeros > 1)
  # class ids sorted according to their probabilities for each instance shape=(num_samples,num_classes)
  prob_preds = np.argsort(probs, axis=1)
  # first and second predicted classes for each instance shape=(num_samples)
  prob_pred1, prob_pred2 = prob_preds[:,-1], prob_preds[:,-2]
  # indices of wrong predictions for first prediction 
  idx_wrong = np.where(prob_pred1 != y_train_int)[0]
  # change mis-predicted instances to their first prediction because it is most similer to that class
  if len(idx_wrong) >= num_noisy:
    # get the probabilities of first predictions for each sample shape=(num_samples)
    prob1 = np.array([probs[i,prob_pred1[i]] for i in range(len(prob_pred1))])
    # sorted list of second prediction probabilities
    idx_sorted = np.argsort(prob1)
    # sort them according to prob1
    idx_wrong2 = get_sorted_idx(prob1, idx_wrong)
    # get elements with highest probability on second prediciton because they are closest to other classes
    idx2change = idx_wrong2[-num_noisy:]
    # change them to their most likely class which is second most probable prediction
    y_noisy[idx2change] = prob_pred1[idx2change]
  else:
    y_noisy[idx_wrong] = prob_pred1[idx_wrong]
    # remaining number of elements to be mislabeled
    num_noisy_remain = num_noisy - len(idx_wrong)
    # get the probabilities of second predictions for each sample shape=(num_samples)
    prob2 = np.array([probs[i,prob_pred2[i]] for i in range(len(prob_pred2))])
    # sorted list of second prediction probabilities
    idx_sorted = np.argsort(prob2)
    # remove already changed indices for wrong first prediction
    idx_wrong2 = np.setdiff1d(idx_sorted, idx_wrong)
    # sort them according to prob2
    idx_wrong2 = get_sorted_idx(prob2, idx_wrong2)
    # get elements with highest probability on second prediciton because they are closest to other classes
    idx2change = idx_wrong2[-num_noisy_remain:]
    # change them to their most likely class which is second most probable prediction
    y_noisy[idx2change] = prob_pred2[idx2change]
    # get indices where second prediction has zero probability
    idx_tmp = np.where(prob2[idx2change] == 0)[0]
    idx_prob0 = idx2change[idx_tmp]
    assert np.sum(prob2[idx_prob0] != 0) == 0
    # since there is no information in second prediction, to prevent all samples with zero probability on second prediction to have same class
    # we will choose a random class for that sample
    for i in idx_prob0:
        classes = np.arange(num_classes)
        classes_clipped = np.delete(classes, y_train_int[i])
        y_noisy[i] = np.random.choice(classes_clipped, 1)

  return y_noisy, probs

def noise_xy_localized(features, y_train_int, noise_ratio):
  print("noise_xy, features.shape: ", features.shape)
  print("noise_xy, y_train_int.shape: ", y_train_int.shape)
  print("noise_xy, noise_ratio: ", noise_ratio)
  
  y_noisy = np.copy(y_train_int)
  n_clusters = 20
  num_classes = len(unique_labels(y_train_int))
  for i in range(num_classes):
    idx = y_train_int == i
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features[idx])
    y_pred = kmeans.labels_

    # number of samples for clusters
    n_samples = [np.sum(y_pred==k) for k in range(n_clusters)]
    sorted_idx = np.argsort(n_samples)
    # find clusters idx whose sum is equal for noise ratio
    n_tobecorrupted = round(np.sum(idx)*noise_ratio)

    for j in range(len(sorted_idx)):
      if n_samples[sorted_idx[j]] > n_tobecorrupted:
        break
    if j > 0:
      mid = sorted_idx[j-1]
    else:
      mid = sorted_idx[0]
    idx_class=np.where(idx==True)[0]
    idx2change=idx_class[y_pred==mid]
    y_noisy[idx2change] = np.random.choice(np.delete(np.arange(num_classes),i),1)
    num_corrupted = len(idx2change)

    for k in reversed(range(j-1)):
      if n_samples[sorted_idx[k]] + num_corrupted < n_tobecorrupted:
        idx2change=idx_class[y_pred==sorted_idx[k]]
        y_noisy[idx2change] = np.random.choice(np.delete(np.arange(num_classes),i),1)
        num_corrupted += len(idx2change)

  return y_noisy
  

def cm_uniform(num_classes, noise_ratio):
    # if noise ratio is integer, convert it to float
    if noise_ratio > 1 and noise_ratio < 100:
        noise_ratio = noise_ratio / 100.
    assert (noise_ratio >= 0.) and (noise_ratio <= 1.)

    P = noise_ratio / (num_classes - 1) * np.ones((num_classes, num_classes))
    np.fill_diagonal(P, (1 - noise_ratio) * np.ones(num_classes))

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def cm_model_prediction(x_test, y_test, logits, noise_ratio):
    # if noise ratio is integer, convert it to float
    if noise_ratio > 1 and noise_ratio < 100:
        noise_ratio = noise_ratio / 100.

    y_pred = np.argmax(logits, axis=1)
    # extract confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # set diagonal entries to 0 for now
    np.fill_diagonal(cm, 0)
    # find probability of each misclassification with avoiding zero division
    sums = cm.sum(axis=1)
    idx_zeros = np.where(sums == 0)[0]
    sums[idx_zeros] = 1
    cm = (cm.T / sums).T
    # weight them with noise
    cm = cm * noise_ratio
    # set diagonal entries
    np.fill_diagonal(cm, (1-noise_ratio))
    # if noise was with zero probabiilty, set the coresponding class probability to 1
    for idx in idx_zeros:
        cm[idx,idx] = 1

    assert_array_almost_equal(cm.sum(axis=1), 1, 1)
    return cm


def noise_cm(y, cm=None):
    assert_array_almost_equal(cm.sum(axis=1), 1, 1)

    y_noisy = np.copy(y)
    num_classes = cm.shape[0]

    for i in range(num_classes):
        # indices of samples belonging to class i
        idx = np.where(y == i)[0]
        # number of samples belonging to class i
        n_samples = len(idx)
        for j in range(num_classes):
            if i != j:
                # number of noisy samples according to confusion matrix
                n_noisy = round(n_samples*cm[i,j])
                if n_noisy > 0:
                    # indices of noisy samples
                    noisy_idx = np.random.choice(len(idx), n_noisy, replace=False)
                    # change their classes
                    y_noisy[idx[noisy_idx]] = j
                    # update indices
                    idx = np.delete(idx, noisy_idx)

    return y_noisy, None




