#!/usr/bin/env python
import argparse
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
from methods import *

from sklearn.mixture import GaussianMixture

from tta import compute_tta_scores

def parse_args():
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('fc', help='Path to config')
    parser.add_argument('id_train_feature', help='Path to data')
    parser.add_argument('id_val_feature', help='Path to output file')
    parser.add_argument('ood_features', nargs="+", help='Path to ood features')
    parser.add_argument('--train_label', default='datalists/imagenet2012_train_random_200k.txt', help='Path to train labels')
    parser.add_argument('--clip_quantile', default=0.99, help='Clip quantile to react')

    return parser.parse_args()

def main():
    args = parse_args()

    ood_names = [splitext(basename(ood))[0] for ood in args.ood_features]
    print(f"ood datasets: {ood_names}")

    w, b = mmcv.load(args.fc)
    print(f'{w.shape=}, {b.shape=}')

    train_labels = np.array([int(line.rsplit(' ', 1)[-1]) for line in mmcv.list_from_file(args.train_label)], dtype=int)

    label_mapper = {}
    with open('superclass.txt') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            label_mapper[i] = int(line)
    
    train_labels_remapped = np.array([label_mapper[l] for l in train_labels])
    recall = 0.95

    print('load features')
    feature_id_train = mmcv.load(args.id_train_feature).squeeze()
    feature_id_val = mmcv.load(args.id_val_feature).squeeze()
    feature_oods = {name: mmcv.load(feat).squeeze() for name, feat in zip(ood_names, args.ood_features)}
    print(f'{feature_id_train.shape=}, {feature_id_val.shape=}')
    for name, ood in feature_oods.items():
        print(f'{name} {ood.shape}')
    print('computing logits...')
    logit_id_train = feature_id_train @ w.T + b
    logit_id_val = feature_id_val @ w.T + b
    logit_oods = {name: feat @ w.T + b for name, feat in feature_oods.items()}

    print('computing softmax...')
    softmax_id_train = softmax(logit_id_train, axis=-1)
    softmax_id_val = softmax(logit_id_val, axis=-1)
    softmax_oods = {name: softmax(logit, axis=-1) for name, logit in logit_oods.items()}

    print("computing gradnorm...")
    gradnorm_id_train = None #gradnorm_noreduce(feature_id_train, w, b)
    gradnorm_id_val = None #gradnorm_noreduce(feature_id_val, w, b)
    gradnorm_oods = None #{name: gradnorm_noreduce(feat, w, b) for name, feat in feature_oods.items()}

    print("computing tta scores")
    TTA_STEPS = 10
    tta_id_val = compute_tta_scores(feature_id_val, w, b, loss="ce", proximal_loss=3e-2, learning_rate=0.1, num_steps=TTA_STEPS)
    tta_oods = {
    name:
        compute_tta_scores(feat_ood, w, b, loss="ce", proximal_loss=3e-2, learning_rate=0.1, num_steps=TTA_STEPS) 
        for name, feat_ood in feature_oods.items() 
    }

    all_results = []

    data_train = {"softmax": softmax_id_train, "logit": logit_id_train, "feat": feature_id_train, "labels": train_labels, "labels2":train_labels_remapped, "gradnorm": gradnorm_id_train}
    data_val = {"softmax": softmax_id_val, "logit": logit_id_val, "feat": feature_id_val, "gradnorm": gradnorm_id_val}
    data_oods = {"softmax": softmax_oods, "logit": logit_oods, "feat": feature_oods, "gradnorm": gradnorm_oods}

    data_val["tta_logits"] = tta_id_val[0][f"logits_step{TTA_STEPS}"]
    data_val["tta_logits_delta"] = np.sum(np.abs(tta_id_val[0]["logits_step1"] - tta_id_val[0][f"logits_step{TTA_STEPS}"]), -1)
    data_val["tta_grads"] = tta_id_val[1][f"grads_step{TTA_STEPS}"]
    data_val["tta_grads_delta"] = np.abs(tta_id_val[1]["grads_step1"] - tta_id_val[1][f"grads_step{TTA_STEPS}"])
    data_oods["tta_logits"] = {
        name: value[0][f"logits_step{TTA_STEPS}"] for name, value in tta_oods.items()
    }
    data_oods["tta_logits_delta"] = {
        name: np.sum(np.abs(value[0]["logits_step1"] - value[0][f"logits_step{TTA_STEPS}"]), -1) for name, value in tta_oods.items()
    }
    data_oods["tta_grads"] = {
        name: value[1][f"grads_step{TTA_STEPS}"] for name, value in tta_oods.items()
    }
    data_oods["tta_grads_delta"] = {
        name: np.abs(value[1]["grads_step1"] - value[1][f"grads_step{TTA_STEPS}"]) for name, value in tta_oods.items()
    }
    data_oods["tta"] = tta_oods
    
    # ---------------------------------------
    result_msp = msp(data_train, data_val, data_oods, w, b)
    all_results += result_msp

    # ---------------------------------------
    result_maxlogit = maxlogit(data_train, data_val, data_oods, w, b)
    all_results += result_maxlogit

    # result_maxlogit_true = maxlogit(data_train, data_val, data_oods, w, b, tta=True)
    # all_results += result_maxlogit_true

    # ---------------------------------------
    result_energy = energy(data_train, data_val, data_oods, w, b)
    all_results += result_energy

    # ---------------------------------------
    result_react = energy_react(data_train, data_val, data_oods, w, b)
    all_results += result_react

    # ---------------------------------------
    result_vim = vim(data_train, data_val, data_oods, w, b)
    all_results += result_vim

    # ---------------------------------------
    result_residual = residual(data_train, data_val, data_oods, w, b)
    all_results += result_residual

    # ---------------------------------------
    result_gradnorm = gradnorm(data_train, data_val, data_oods, w, b)
    all_results += result_gradnorm

    # ---------------------------------------
    result_tta_grads = tta_gradshift(data_train, data_val, data_oods, w, b)
    all_results += result_tta_grads

    # ---------------------------------------
    result_tta_logits = tta_logitshift(data_train, data_val, data_oods, w, b)
    all_results += result_tta_logits

    # ---------------------------------------
    result_mahalanobis = mahalanobis_ood(data_train, data_val, data_oods, w, b, type="logit")
    all_results += result_mahalanobis

    # print(result_mahalanobis)

    # likelihood_mahalanobis = mixture_likelihood(data_train, data_val, data_oods, w, b, n_mixtures=2, type="logit", pca=False, label_type="true")
    # print(likelihood_mahalanobis)

    # ---------------------------------------
    # experimental metrics
    # for type in ["logit", "softmax", "gradnorm"]:
    #     for pca in [False]:
    #         for remap in [True, False]:
    #             for labels in ["hierarchical"]:
    #                 result = mahalanobis_ood(data_train, data_val, data_oods, w, b, type=type, pca=pca, labels=labels, remap=remap)
    #                 print(f"Using {type} to fit Mahalanobis, with {labels} labels")
    #                 print(result)

    # for type in ["feat", "logit", "softmax", "gradnorm"]:
    #     for n_mixtures in [1]:
    #         for pca in [False]:
    #             for labels in ["true", "hierarchical", "argmax"]:
    #                 print(f"Using {type} to fit GMM with {n_mixtures} comp, with {labels} labels")
    #                 result_mixture1 = mixture_likelihood(data_train, data_val, data_oods, w, b, n_mixtures=n_mixtures, type=type, pca=pca, label_type=labels)
    #                 print(result_mixture1)

    # mahalanobis_logit, id_logit, ood_logit = mahalanobis_ood(data_train, data_val, data_oods, w, b, type="logit")
    # all_results += mahalanobis_logit

    # mahalanobis_softmax, id_softmax, ood_softmax = mahalanobis_ood(data_train, data_val, data_oods, w, b, type="softmax")
    # all_results += mahalanobis_softmax

    # mahalanobis_logit, id_logit, ood_logit = mahalanobis_ood(data_train, data_val, data_oods, w, b, type="logit", pca=True)
    # all_results += mahalanobis_logit

    # mahalanobis_logit, id_logit, ood_logit = mahalanobis_ood(data_train, data_val, data_oods, w, b, type="logit")
    # all_results += mahalanobis_logit

    result_df = pd.DataFrame(all_results)
    result_df = result_df.round(decimals=4)
    result_df.to_csv("results.csv")
    print("")
    # method = 'Mahalanobis Logit'
    # print(f'\n{method}')
    # result = []

    # print('computing classwise mean feature...')
    # train_means = []
    # train_feat_centered = []
    # for i in tqdm(range(1000)):
    #     fs = logit_id_train[train_labels == i]
    #     mask = np.ones_like(fs, dtype=bool)
    #     mask[range(fs.shape[0]), np.argmax(fs, -1)] = False
    #     fs = fs[mask].reshape(fs.shape[0], fs.shape[1]-1)
    #     fs = fs / fs.sum(axis=-1, keepdims=True)
    #     _m = fs.mean(axis=0)
    #     train_means.append(_m)
    #     train_feat_centered.extend(fs - _m)

    # print('computing precision matrix...')
    # ec = EmpiricalCovariance(assume_centered=True)
    # ec.fit(np.array(train_feat_centered).astype(np.float64))

    # print('go to gpu...')
    # mean = torch.from_numpy(np.array(train_means)).cuda().float()
    # prec = torch.from_numpy(ec.precision_).cuda().float()

    # val_mask = np.ones_like(logit_id_val, dtype=bool)
    # val_mask[range(logit_id_val.shape[0]), np.argmax(logit_id_val, -1)] = False
    # logit_id_val_filter = logit_id_val[val_mask].reshape(logit_id_val.shape[0], logit_id_val.shape[1]-1)
    # logit_id_val_filter = logit_id_val_filter / logit_id_val_filter.sum(axis=-1, keepdims=True)

    # score_id = -np.array([(((f - mean)@prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in tqdm(torch.from_numpy(logit_id_val_filter).cuda().float())])
    # for name, logit_ood in logit_oods.items():
    #     ood_mask = np.ones_like(logit_ood, dtype=bool)
    #     ood_mask[range(logit_ood.shape[0]), np.argmax(logit_ood, -1)] = False
    #     logit_ood_filter = logit_ood[ood_mask].reshape(logit_ood.shape[0], logit_ood.shape[1]-1)
    #     logit_ood_filter = logit_ood_filter / logit_ood_filter.sum(axis=-1, keepdims=True)
    #     score_ood = -np.array([(((f - mean)@prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in tqdm(torch.from_numpy(logit_ood_filter).cuda().float())])
    #     auc_ood = auc(score_id, score_ood)[0]
    #     fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
    #     result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
    #     print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    # df = pd.DataFrame(result)
    # dfs.append(df)
    # print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    # method = 'Grad Mahalanobis'
    # print(f'\n{method}')
    # result = []

    # print('computing classwise mean logit...')
    # means = []
    # precs = []
    # for i in tqdm(range(1000)):
    #     fs = gradnorm_id_train[train_labels == i]
    #     means.append(torch.from_numpy(fs.mean(axis=0)).cuda().float())
    #     precs.append(torch.from_numpy(np.linalg.pinv(np.cov(fs.T))).cuda().float())

    # # print('computing precision matrix...')
    # # ec = EmpiricalCovariance(assume_centered=True)
    # # ec.fit(np.array(train_feat_centered).astype(np.float64))

    # # print('go to gpu...')
    # # mean = torch.from_numpy(np.array(train_means)).cuda().float()
    # # prec = torch.from_numpy(ec.precision_).cuda().float()

    # score_id = -np.array([(((f - means[f.argmax(-1)])@precs[f.argmax(-1)]) * (f - means[f.argmax(-1)])).sum(axis=-1).cpu().numpy() for f in tqdm(torch.from_numpy(gradnorm_id_val).cuda().float())])

    # # score_id = -np.array([(((f - mean)@prec) * (f - mean)).sum(axis=-1).cpu().numpy() for f in tqdm(torch.from_numpy(logit_id_val).cuda().float())])
    # # idx_maxlogit = logit_id_val.argmax(-1).reshape(-1, 1)
    # # score_id = score_id[np.arange(score_id.shape[0])[:,None], idx_maxlogit]
    # for name, gradnorm_ood in gradnorm_oods.items():
    #     # score_ood = -np.array([(((f - mean)@prec) * (f - mean)).sum(axis=-1).cpu().numpy() for f in tqdm(torch.from_numpy(logit_ood).cuda().float())])
    #     # idx = logit_ood.argmax(-1).reshape(-1, 1)
    #     # score_ood = score_ood[np.arange(score_ood.shape[0])[:,None], idx]
    #     score_ood = -np.array([(((f - means[f.argmax(-1)])@precs[f.argmax(-1)]) * (f - means[f.argmax(-1)])).sum(axis=-1).cpu().numpy() for f in tqdm(torch.from_numpy(gradnorm_ood).cuda().float())])
    #     auc_ood = auc(score_id, score_ood)[0]
    #     fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
    #     result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
    #     print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    # df = pd.DataFrame(result)
    # dfs.append(df)
    # print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')


    # ---------------------------------------
    # method = 'Mixture Softmax Likelihood'
    # print(f'\n{method}')
    # result = []

    # max_comp = 2
    # chosen_comp = [0] * max_comp
    # mixtures = []
    # for i in tqdm(range(1000)):
    #     # remove highest logit element
    #     fs = softmax(logit_id_train[train_labels == i], -1)
    #     mask = np.ones_like(fs, dtype=bool)
    #     mask[range(fs.shape[0]), np.argmax(fs, -1)] = False
    #     fs = fs[mask].reshape(fs.shape[0], fs.shape[1]-1)
    #     best_bic = np.inf
    #     best_mixture = None
    #     best_comp = 1
    #     for comp in range(1, max_comp+1):
    #         mixture = GaussianMixture(n_components=comp, random_state=42, covariance_type="diag", reg_covar=1e-5).fit(fs)
    #         bic = mixture.bic(fs)
    #         if bic < best_bic:
    #             best_mixture = mixture
    #             best_bic = bic
    #             best_comp = comp
    #     chosen_comp[best_comp-1] = chosen_comp[best_comp-1] + 1
    #     mixtures.append(best_mixture)

    # val_mask = np.ones_like(logit_id_val, dtype=bool)
    # val_mask[range(logit_id_val.shape[0]), np.argmax(logit_id_val, -1)] = False
    # softmax_id_val_filter = softmax(logit_id_val[val_mask].reshape(logit_id_val.shape[0], logit_id_val.shape[1]-1), -1)

    # score_id = np.array([(mixtures[f.argmax(-1)].score_samples(f.reshape(1, -1))).sum(axis=-1) for f in tqdm(softmax_id_val_filter)])
    # for name, logit_ood in logit_oods.items():
    #     ood_mask = np.ones_like(logit_ood, dtype=bool)
    #     ood_mask[range(logit_ood.shape[0]), np.argmax(logit_ood, -1)] = False
    #     softmax_ood_filter = softmax(logit_ood[ood_mask].reshape(logit_ood.shape[0], logit_ood.shape[1]-1), -1)
    #     score_ood = np.array([(mixtures[f.argmax(-1)].score_samples(f.reshape(1, -1))).sum(axis=-1) for f in tqdm(softmax_ood_filter)])
    #     auc_ood = auc(score_id, score_ood)[0]
    #     fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
    #     result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
    #     print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    # df = pd.DataFrame(result)
    # dfs.append(df)
    # print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    # method = 'Mixture Softmax Likelihood'
    # print(f'\n{method}')
    # result = []

    # print('computing classwise mean logit...')
    # max_comp = 3
    # # chosen_comp = [0] * max_comp
    # # mixtures = []
    # for comp in range(1, max_comp+1):
    #     print("NUM COMPONENTS", comp)
    #     mixtures = []
    #     lls_val = []
    #     lls_oods = {}
    #     for i in tqdm(range(1000)):
    #         fs = softmax_id_train[train_labels == i]
    #         mixture = GaussianMixture(n_components=comp, random_state=42, covariance_type="full", reg_covar=1e-5).fit(fs)
    #         lls_val.append(mixture.score_samples(softmax_id_val))
    #         for name, softmax_ood in softmax_oods.items():
    #             if name not in lls_oods.keys():
    #                 lls_oods[name] = [mixture.score_samples(softmax_ood)]
    #             else:
    #                 lls_oods[name].append(mixture.score_samples(softmax_ood))
    #         # mixtures.append(mixture)

    #     score_id = -np.array([lls_val[f.argmax(-1)] for f in tqdm(softmax_id_val)])
    #     # score_id = -np.array([(mixtures[f.argmax(-1)].score_samples(f.reshape(1, -1))).sum(axis=-1) for f in tqdm(softmax_id_val)])
    #     for name, softmax_ood in softmax_oods.items():
    #         score_ood = -np.array([lls_oods[name][f.argmax(-1)] for f in tqdm(softmax_ood)])
    #         # score_ood = -np.array([(mixtures[f.argmax(-1)].score_samples(f.reshape(1, -1))).sum(axis=-1) for f in tqdm(softmax_ood)])
    #         auc_ood = auc(score_id, score_ood)[0]
    #         fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
    #         result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
    #         print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    #     df = pd.DataFrame(result)
    #     dfs.append(df)
    #     print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # # ---------------------------------------
    # method = 'Logit Mahalanobis'
    # print(f'\n{method}')
    # result = []

    # print('computing classwise mean logit...')
    # means = []
    # precs = []
    # for i in tqdm(range(1000)):
    #     fs = logit_id_train[train_labels == i]
    #     means.append(torch.from_numpy(fs.mean(axis=0)).cuda().float())
    #     precs.append(torch.from_numpy(np.linalg.pinv(np.cov(fs.T))).cuda().float())

    # # print('computing precision matrix...')
    # # ec = EmpiricalCovariance(assume_centered=True)
    # # ec.fit(np.array(train_feat_centered).astype(np.float64))

    # # print('go to gpu...')
    # # mean = torch.from_numpy(np.array(train_means)).cuda().float()
    # # prec = torch.from_numpy(ec.precision_).cuda().float()

    # score_id = np.array([(((f - means[f.argmax(-1)])@precs[f.argmax(-1)]) * (f - means[f.argmax(-1)])).sum(axis=-1).cpu().numpy() for f in tqdm(torch.from_numpy(logit_id_val).cuda().float())])

    # # score_id = -np.array([(((f - mean)@prec) * (f - mean)).sum(axis=-1).cpu().numpy() for f in tqdm(torch.from_numpy(logit_id_val).cuda().float())])
    # # idx_maxlogit = logit_id_val.argmax(-1).reshape(-1, 1)
    # # score_id = score_id[np.arange(score_id.shape[0])[:,None], idx_maxlogit]
    # for name, logit_ood in logit_oods.items():
    #     # score_ood = -np.array([(((f - mean)@prec) * (f - mean)).sum(axis=-1).cpu().numpy() for f in tqdm(torch.from_numpy(logit_ood).cuda().float())])
    #     # idx = logit_ood.argmax(-1).reshape(-1, 1)
    #     # score_ood = score_ood[np.arange(score_ood.shape[0])[:,None], idx]
    #     score_ood = np.array([(((f - means[f.argmax(-1)])@precs[f.argmax(-1)]) * (f - means[f.argmax(-1)])).sum(axis=-1).cpu().numpy() for f in tqdm(torch.from_numpy(logit_ood).cuda().float())])
    #     auc_ood = auc(score_id, score_ood)[0]
    #     fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
    #     result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
    #     print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    # df = pd.DataFrame(result)
    # dfs.append(df)
    # print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # # ---------------------------------------
    # method = 'Softmax Mahalanobis'
    # print(f'\n{method}')
    # result = []

    # print('computing classwise mean logit...')
    # means = []
    # precs = []
    # for i in tqdm(range(1000)):
    #     fs = softmax_id_train[train_labels == i]
    #     means.append(torch.from_numpy(fs.mean(axis=0)).cuda().float())
    #     precs.append(torch.from_numpy(np.linalg.pinv(np.cov(fs.T))).cuda().float())

    # # print('computing precision matrix...')
    # # ec = EmpiricalCovariance(assume_centered=True)
    # # ec.fit(np.array(train_feat_centered).astype(np.float64))

    # # print('go to gpu...')
    # # mean = torch.from_numpy(np.array(train_means)).cuda().float()
    # # prec = torch.from_numpy(ec.precision_).cuda().float()

    # score_id = -np.array([(((f - means[f.argmax(-1)])@precs[f.argmax(-1)]) * (f - means[f.argmax(-1)])).sum(axis=-1).cpu().numpy() for f in tqdm(torch.from_numpy(softmax_id_val).cuda().float())])

    # # score_id = -np.array([(((f - mean)@prec) * (f - mean)).sum(axis=-1).cpu().numpy() for f in tqdm(torch.from_numpy(logit_id_val).cuda().float())])
    # # idx_maxlogit = logit_id_val.argmax(-1).reshape(-1, 1)
    # # score_id = score_id[np.arange(score_id.shape[0])[:,None], idx_maxlogit]
    # for name, softmax_ood in softmax_oods.items():
    #     # score_ood = -np.array([(((f - mean)@prec) * (f - mean)).sum(axis=-1).cpu().numpy() for f in tqdm(torch.from_numpy(logit_ood).cuda().float())])
    #     # idx = logit_ood.argmax(-1).reshape(-1, 1)
    #     # score_ood = score_ood[np.arange(score_ood.shape[0])[:,None], idx]
    #     score_ood = -np.array([(((f - means[f.argmax(-1)])@precs[f.argmax(-1)]) * (f - means[f.argmax(-1)])).sum(axis=-1).cpu().numpy() for f in tqdm(torch.from_numpy(softmax_ood).cuda().float())])
    #     auc_ood = auc(score_id, score_ood)[0]
    #     fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
    #     result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
    #     print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    # df = pd.DataFrame(result)
    # dfs.append(df)
    # print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # # ---------------------------------------
    # # method = 'KL-Matching'
    # # print(f'\n{method}')
    # # result = []

    # # print('computing classwise mean softmax...')
    # # pred_labels_train = np.argmax(softmax_id_train, axis=-1)
    # # mean_softmax_train = [softmax_id_train[pred_labels_train==i].mean(axis=0) for i in tqdm(range(1000))]

    # # score_id = -pairwise_distances_argmin_min(softmax_id_val, np.array(mean_softmax_train), metric=kl)[1]

    # # for name, softmax_ood in softmax_oods.items():
    # #     score_ood = -pairwise_distances_argmin_min(softmax_ood, np.array(mean_softmax_train), metric=kl)[1]
    # #     auc_ood = auc(score_id, score_ood)[0]
    # #     fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
    # #     result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
    # #     print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    # # df = pd.DataFrame(result)
    # # dfs.append(df)
    # print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

if __name__ == '__main__':
    main()
