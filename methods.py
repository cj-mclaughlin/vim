import argparse
from sklearn.mixture import GaussianMixture
import torch
import numpy as np
from tqdm import tqdm
import mmcv
from numpy.linalg import norm, pinv
from scipy.special import softmax
from sklearn import metrics
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.covariance import EmpiricalCovariance
from os.path import basename, splitext
from scipy.special import logsumexp
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import copy

RECALL = 0.95
CLIP_QUANTILE = 0.99

label_mapper = {}
with open('superclass.txt') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        label_mapper[i] = int(line)  # original: new

inv_map = {}
for k, v in label_mapper.items():
    inv_map[v] = inv_map.get(v, []) + [k]

def num_fp_at_recall(ind_conf, ood_conf, tpr):
    num_ind = len(ind_conf)

    if num_ind == 0 and len(ood_conf) == 0:
        return 0, 0.
    if num_ind == 0:
        return 0, np.max(ood_conf) + 1

    recall_num = int(np.floor(tpr * num_ind))
    thresh = np.sort(ind_conf)[-recall_num]
    num_fp = np.sum(ood_conf >= thresh)
    return num_fp, thresh

def fpr_recall(ind_conf, ood_conf, tpr):
    num_fp, thresh = num_fp_at_recall(ind_conf, ood_conf, tpr)
    num_ood = len(ood_conf)
    fpr = num_fp / max(1, num_ood)
    return fpr, thresh

def auc(ind_conf, ood_conf):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))

    fpr, tpr, _ = metrics.roc_curve(ind_indicator, conf)
    precision_in, recall_in, _ = metrics.precision_recall_curve(
        ind_indicator, conf)
    precision_out, recall_out, _ = metrics.precision_recall_curve(
        1 - ind_indicator, 1 - conf)

    auroc = metrics.auc(fpr, tpr)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out

def kl(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def l1_ce_gradnorm(x, w, b, dim=None):
    fc = torch.nn.Linear(*w.shape[::-1])
    fc.weight.data[...] = torch.from_numpy(w)
    fc.bias.data[...] = torch.from_numpy(b)
    fc.cuda()

    x = torch.from_numpy(x).float().cuda()
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()

    confs = []

    for i in tqdm(x):
        targets = torch.ones((1, 1000)).cuda()
        fc.zero_grad()
        loss = torch.mean(torch.sum(-targets * logsoftmax(fc(i[None])), dim=-1))
        loss.backward()
        layer_grad_norm = torch.sum(torch.abs(fc.weight.grad.data), dim=dim).cpu().numpy()
        confs.append(layer_grad_norm)

    return np.array(confs)

def gradnorm_noreduce(x, w, b):
    fc = torch.nn.Linear(*w.shape[::-1])
    fc.weight.data[...] = torch.from_numpy(w)
    fc.bias.data[...] = torch.from_numpy(b)
    fc.cuda()

    x = torch.from_numpy(x).float().cuda()
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()

    confs = []

    for i in tqdm(x):
        targets = fc(i[None]).argmax(-1).detach()
        fc.zero_grad()
        loss = torch.nn.CrossEntropyLoss()(fc(i[None]), targets)
        #loss = torch.mean(torch.sum(-targets * logsoftmax(fc(i[None])), dim=-1))
        loss.backward()
        layer_grad_norm = torch.sum(torch.abs(fc.weight.grad.data), dim=-1).cpu().numpy()
        confs.append(layer_grad_norm)

    return np.array(confs)

def tta_gradshift(data_train, data_val, data_oods, w, b):
    score_id = data_val["tta_grads_delta"]
    scores_oods = data_oods["tta_grads_delta"]
    result = []
    for name, score_ood in scores_oods.items():
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, RECALL)
        result.append(dict(method=f"TTA_gradient_delta", oodset=name, auroc=auc_ood, fpr=fpr_ood))
    return result

def tta_logitshift(data_train, data_val, data_oods, w, b):
    score_id = data_val["tta_logits_delta"]
    scores_oods = data_oods["tta_logits_delta"]
    result = []
    for name, score_ood in scores_oods.items():
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, RECALL)
        result.append(dict(method=f"TTA_logit_delta", oodset=name, auroc=auc_ood, fpr=fpr_ood))
    return result

def mahalanobis_distance(train, val, oods, labels):
    result = []

    train_means = []
    train_feat_centered = []
    for i in tqdm(range(max(labels)+1)):
        fs = train[labels == i]
        _m = fs.mean(axis=0)
        train_means.append(_m)
        train_feat_centered.extend(fs - _m)

    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(np.array(train_feat_centered).astype(np.float64))

    mean = torch.from_numpy(np.array(train_means)).cuda().float()
    prec = torch.from_numpy(ec.precision_).cuda().float()

    score_id = -np.array([(((f - mean)@prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in tqdm(torch.from_numpy(val).cuda().float())])
    for name, ood in oods.items():
        score_ood = -np.array([(((f - mean)@prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in tqdm(torch.from_numpy(ood).cuda().float())])
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, RECALL)
        result.append(dict(method=f"Mahalanobis", oodset=name, auroc=auc_ood, fpr=fpr_ood))
    return result

def mahalanobis_ood(data_train, data_val, data_oods, w, b, type="feat", labels="true", pca=False, remap=False):
    """
    Max-Mahalanobis Distance
    """
    assert type in ["feat", "logit", "softmax", "gradnorm"]
    assert labels in ["true", "hierarchical", "argmax"]

    train = data_train[type]
    val = data_val[type]
    oods = copy.deepcopy(data_oods[type])

    logit_mapper = np.zeros(shape=(1000, 558))
    for new_key in inv_map.keys():
        values = inv_map[new_key]
        for v in values:
            logit_mapper[v, new_key] = 1.0 / len(values)

    if remap:
        train = np.matmul(train, logit_mapper)
        val = np.matmul(val, logit_mapper)
        for key in oods.keys():
            oods[key] = np.matmul(oods[key], logit_mapper)

    if pca:
        pca = PCA(n_components=0.99).fit(train)
        train = pca.transform(train)
        val = pca.transform(val)
        for key in oods.keys():
            oods[key] = pca.transform(oods[key])

    if labels == "true":
        labels = data_train["labels"]
    elif labels == "hierarchical":
        labels = data_train["labels2"]
    elif labels == "argmax":
        labels = np.argmax(data_train["softmax"], axis=-1)

    return mahalanobis_distance(train, val, oods, labels)

def mixture_likelihood(data_train, data_val, data_oods, w, b, type="feat", label_type="true", n_mixtures=1, pca=False):
    assert type in ["feat", "logit", "softmax", "gradnorm"]
    assert label_type in ["true", "hierarchical", "argmax"]

    train = data_train[type]
    val = data_val[type]
    oods = copy.deepcopy(data_oods[type])
    if pca:
        pca = PCA(n_components=0.99).fit(train)
        train = pca.transform(train)
        val = pca.transform(val)
        for key in oods.keys():
            oods[key] = pca.transform(oods[key])
    if label_type == "true":
        labels = data_train["labels"]
    elif label_type == "hierarchical":
        labels = data_train["labels2"]
    elif label_type == "argmax":
        labels = np.argmax(data_train["softmax"], axis=-1)
    result = []

    mixtures = []
    for i in tqdm(range(max(labels)+1)):
        fs = train[labels == i]
        _m = fs.mean(axis=0)
        mixture = GaussianMixture(covariance_type="diag", reg_covar=1e-4, n_components=n_mixtures).fit(fs)
        mixtures.append(mixture)

    # we can do one of two things
    # 1) use the likelihood of the mixture of the argmax class for each instance
    # 2) use the max-likelihood value across all mixtures (similar to min-mahalanobis distance.)
    val_argmax = np.argmax(data_val["softmax"], -1)
    if label_type == "hierarchical":
        val_argmax = np.array([label_mapper[l] for l in val_argmax])
    print("computing id mixture likelihoods")
    id_likelihoods = np.array([mixtures[i].score_samples(val) for i in tqdm(range(len(mixtures)))])  # c x n

    # 1)
    #score_id = np.array([mixtures[val_argmax[i]].score(val[i].reshape(1,-1)) for i in range(len(val))])
    # 2)
    score_id = np.max(id_likelihoods, axis=0)

    for name, ood in oods.items():
        ood_argmax = np.argmax(data_oods["softmax"][name], -1)
        if label_type == "hierarchical":
            ood_argmax = np.array([label_mapper[l] for l in ood_argmax])
        print("computing ood mixture likelihoods")
        ood_likelihoods = np.array([mixtures[i].score_samples(ood) for i in tqdm(range(len(mixtures)))])  # c x n
        # 1)
        #score_ood = np.array([mixtures[ood_argmax[i]].score(ood[i].reshape(1,-1)) for i in range(len(ood))])
        # 2)
        score_ood = np.max(ood_likelihoods, axis=0)
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, RECALL)
        result.append(dict(method=f"Mahalanobis_{type}", oodset=name, auroc=auc_ood, fpr=fpr_ood))
    return result

def msp(data_train, data_val, data_oods, w, b):
    softmax_id_val = data_val["softmax"]
    softmax_oods = data_oods["softmax"]
    result = []
    score_id = softmax_id_val.max(axis=-1)
    for name, softmax_ood in softmax_oods.items():
        score_ood = softmax_ood.max(axis=-1)
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, RECALL)
        result.append(dict(method="MSP", oodset=name, auroc=auc_ood, fpr=fpr_ood))
    return result

def maxlogit(data_train, data_val, data_oods, w, b, tta=False):
    logit_id_val = data_val["logit" if not tta else "tta_logits"]
    logit_oods = data_oods["logit" if not tta else "tta_logits"]
    result = []
    score_id = logit_id_val.max(axis=-1)
    for name, logit_ood in logit_oods.items():
        score_ood = logit_ood.max(axis=-1)
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, RECALL)
        result.append(dict(method="MaxLogit" if not tta else "MaxLogit+TTA", oodset=name, auroc=auc_ood, fpr=fpr_ood))
    return result

def energy(data_train, data_val, data_oods, w, b):
    logit_id_val = data_val["logit"]
    logit_oods = data_oods["logit"]
    result = []
    score_id = logsumexp(logit_id_val, axis=-1)
    for name, logit_ood in logit_oods.items():
        score_ood = logsumexp(logit_ood, axis=-1)
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, RECALL)
        result.append(dict(method="Energy", oodset=name, auroc=auc_ood, fpr=fpr_ood))
    return result

def energy_react(data_train, data_val, data_oods, w, b):
    feature_id_train = data_train["feat"]
    feature_id_val = data_val["feat"]
    feature_oods = data_oods["feat"]
    result = []
    clip = np.quantile(feature_id_train, CLIP_QUANTILE)
    logit_id_val_clip = np.clip(feature_id_val, a_min=None, a_max=clip) @ w.T + b
    score_id = logsumexp(logit_id_val_clip, axis=-1)
    for name, feature_ood in feature_oods.items():
        logit_ood_clip = np.clip(feature_ood, a_min=None, a_max=clip) @ w.T + b
        score_ood = logsumexp(logit_ood_clip, axis=-1)
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, RECALL)
        result.append(dict(method="Energy+ReAct", oodset=name, auroc=auc_ood, fpr=fpr_ood))
    return result

def vim(data_train, data_val, data_oods, w, b):
    feature_id_train = data_train["feat"]
    feature_id_val = data_val["feat"]
    feature_oods = data_oods["feat"]
    logit_id_train = data_train["logit"]
    logit_id_val = data_val["logit"]
    logit_oods = data_oods["logit"]
    u = -np.matmul(pinv(w), b)

    result = []
    DIM = 1000 if feature_id_val.shape[-1] >= 2048 else 512

    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feature_id_train - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

    vlogit_id_train = norm(np.matmul(feature_id_train - u, NS), axis=-1)
    alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()

    vlogit_id_val = norm(np.matmul(feature_id_val - u, NS), axis=-1) * alpha
    energy_id_val = logsumexp(logit_id_val, axis=-1)
    score_id = -vlogit_id_val + energy_id_val

    ood_names = logit_oods.keys()
    for name in ood_names:
        logit_ood = logit_oods[name]
        feature_ood = feature_oods[name]
        energy_ood = logsumexp(logit_ood, axis=-1)
        vlogit_ood = norm(np.matmul(feature_ood - u, NS), axis=-1) * alpha
        score_ood = -vlogit_ood + energy_ood
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, RECALL)
        result.append(dict(method="ViM", oodset=name, auroc=auc_ood, fpr=fpr_ood))
    return result

def residual(data_train, data_val, data_oods, w, b):
    feature_id_train = data_train["feat"]
    feature_id_val = data_val["feat"]
    feature_oods = data_oods["feat"]
    
    result = []
    DIM = 1000 if feature_id_val.shape[-1] >= 2048 else 512

    u = -np.matmul(pinv(w), b)

    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feature_id_train - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

    score_id = -norm(np.matmul(feature_id_val - u, NS), axis=-1)
    for name, feature_ood in feature_oods.items():
        score_ood = -norm(np.matmul(feature_ood - u, NS), axis=-1)
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, RECALL)
        result.append(dict(method="Residual", oodset=name, auroc=auc_ood, fpr=fpr_ood))
    return result

def gradnorm(data_train, data_val, data_oods, w, b):
    feature_id_val = data_val["feat"]
    feature_oods = data_oods["feat"]
    result = []
    score_id = l1_ce_gradnorm(feature_id_val, w, b)
    for name, feature_ood in feature_oods.items():
        score_ood = l1_ce_gradnorm(feature_ood, w, b)
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, RECALL)
        result.append(dict(method="GradNorm", oodset=name, auroc=auc_ood, fpr=fpr_ood))
    return result