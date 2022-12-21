# Imports.
import argparse
import os
import time
import copy
import json
import cv2
import random
import warnings
from collections import defaultdict
warnings.filterwarnings("ignore", category=UserWarning)

from tqdm import tqdm
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler, ConcatDataset
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from sklearn import metrics
from skimage.feature import greycomatrix, greycoprops
from sklearn.utils import class_weight
from sklearn import svm
from sklearn.model_selection import GridSearchCV

from common import *



# Constants.
NAME_STR = ""
DESCRIP_STR = ""




def restructure_for_svm(ds):
    x = []
    y = []
    for b_mode, b_texture, _, _, _, label, day in tqdm(ds):
        vec = np.concatenate([b_texture.numpy(), day.numpy()])
        x.append(vec)
        y.append([label])
    x = np.array(x)
    y = np.array(y)
    return (x, y)




def main(args):
    train_ds_0 = BurnDS(os.path.join(args.data_dir, "train_d0"), t = 0)
    train_ds_3 = BurnDS(os.path.join(args.data_dir, "train_d3"), t = 3)
    train_ds_7 = BurnDS(os.path.join(args.data_dir, "train_d7"), t = 7)
    train_ds = ConcatDataset([train_ds_0, train_ds_3, train_ds_7])
    test_ds_0 = BurnDS(os.path.join(args.data_dir, "test_d0"))
    test_ds_3 = BurnDS(os.path.join(args.data_dir, "test_d3"))
    test_ds_7 = BurnDS(os.path.join(args.data_dir, "test_d7"))
    test_ds = ConcatDataset([test_ds_0, test_ds_3, test_ds_7])
    print("Test dataset size:   %d." % len(test_ds))
    print("Train dataset size:  %d." % len(train_ds))
    if args.mode == "svm":
        classifier = svm.SVC(gamma=0.0001)
        x, y = restructure_for_svm(train_ds)
        classifier.fit(x, y)
        predicted = classifier.predict(x)
        disp = metrics.ConfusionMatrixDisplay.from_predictions(y, predicted)
        disp.figure_.suptitle("Confusion Matrix")
        accu = metrics.accuracy_score(y, predicted)
        reca = metrics.recall_score(y, predicted, average="macro", zero_division=0)
        prec = metrics.precision_score(y, predicted, average="macro", zero_division=0)
        f1 = metrics.f1_score(y, predicted, average="macro", zero_division=0)
        print("Training Confusion Matrix:")
        print(disp.confusion_matrix)
        print()
        print("Training Stats:")
        print("  Accuracy:   %.4f." % accu)
        print("  Precision:  %.4f." % prec)
        print("  Recall:     %.4f." % reca)
        print("  F1 score:   %.4f." % f1)
        print()
        x, y = restructure_for_svm(test_ds)
        predicted = classifier.predict(x)
        disp = metrics.ConfusionMatrixDisplay.from_predictions(y, predicted)
        disp.figure_.suptitle("Confusion Matrix")
        accu = metrics.accuracy_score(y, predicted)
        reca = metrics.recall_score(y, predicted, average="macro", zero_division=0)
        prec = metrics.precision_score(y, predicted, average="macro", zero_division=0)
        f1 = metrics.f1_score(y, predicted, average="macro", zero_division=0)
        print("Testing Confusion Matrix:")
        print(disp.confusion_matrix)
        print()
        print("Testing Stats:")
        print("  Accuracy:   %.4f." % accu)
        print("  Precision:  %.4f." % prec)
        print("  Recall:     %.4f." % reca)
        print("  F1 score:   %.4f." % f1)
        print()
    elif args.mode == "svm_grid":
        x, y = restructure_for_svm(train_ds)
        param_grid = {"C": [0.1, 1, 10, 100, 1000], "gamma": [1, 0.1, 0.01, 0.001, 0.0001], "kernel": ["rbf"]}
        grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3)
        grid.fit(x, y)
        print("Best Hyperparams:")
        print(grid.best_params_)
        print(grid.best_estimator_)
    print("Done.")




#--------------------------------[module setup]---------------------------------
def config_cli_parser(parser):
    parser.add_argument("--mode", help="Runmode of the application.", choices = ["svm", "svm_grid"], default = "svm")
    parser.add_argument("--data_dir", help="", default = r"../data/frames/")
    return parser



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = NAME_STR, description = DESCRIP_STR)   # Create module's cli parser.
    parser = config_cli_parser(parser)
    args = parser.parse_args()
    main(args)

#===============================================================================
