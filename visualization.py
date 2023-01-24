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
    gradnorm_id_train = 0# gradnorm_noreduce(feature_id_train, w, b)
    gradnorm_id_val = 0#gradnorm_noreduce(feature_id_val, w, b)
    gradnorm_oods = 0#{name: gradnorm_noreduce(feat, w, b) for name, feat in feature_oods.items()}

    u = -np.matmul(pinv(w), b)

    df = pd.DataFrame(columns = ['method', 'oodset', 'auroc', 'fpr'])

    all_results = []

    data_train = {"softmax": softmax_id_train, "logit": logit_id_train, "feat": feature_id_train, "labels": train_labels, "gradnorm": gradnorm_id_train}
    data_val = {"softmax": softmax_id_val, "logit": logit_id_val, "feat": feature_id_val, "gradnorm": gradnorm_id_val}
    data_oods = {"softmax": softmax_oods, "logit": logit_oods, "feat": feature_oods, "gradnorm": gradnorm_oods}

    from robustness import datasets
    from robustness.tools.imagenet_helpers import common_superclass_wnid, ImageNetHierarchy

    META_PATH = "/home/connor/dev/Data/ImageNet/meta"
    IN_PATH = "/home/connor/dev/Data/ImageNet"
    in_hier = ImageNetHierarchy(IN_PATH, META_PATH)
    # superclass_wnid = common_superclass_wnid('mixed_10')
    # class_ranges, label_map = in_hier.get_subclasses(superclass_wnid, balanced=True)
    superclass_wnid, class_ranges, label_map = in_hier.get_superclasses(n_superclasses=10)
    all_classes = []
    for s in class_ranges:
        all_classes += list(s)
    print("")

if __name__ == "__main__":
    main()