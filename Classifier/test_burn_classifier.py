
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
#import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler, ConcatDataset
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import sklearn.metrics
from skimage.feature import greycomatrix, greycoprops
from sklearn.utils import class_weight

from torchvision.utils import save_image

from common import *


# Constants.
NAME_STR = ""
DESCRIP_STR = ""



def test_model(model, dataloader, dev):
    model.eval()
    confusion = defaultdict(lambda: np.zeros((4, 4), dtype=np.int64))
    acc = dict()
    recall = dict()
    precision = dict()
    f1 = dict()
    all_preds = []
    all_targs = []
    for phase in dataloader.keys():
        preds = []
        targs = []
        for b_mode, b_texture, tdi_mode, tdi_stats, tdi_bar, label, day in tqdm(dataloader[phase]):
            b_mode = b_mode.to(dev)
            b_texture = b_texture.to(dev)
            tdi_mode = tdi_mode.to(dev)
            tdi_stats = tdi_stats.to(dev)
            tdi_bar = tdi_bar.to(dev)
            label = label.to(dev)
            day = day.to(dev)
            outputs = model(b_mode, b_texture, tdi_mode, tdi_stats, tdi_bar, day)
            _, predictions = torch.max(outputs, 1)
            preds.append(predictions.data.cpu().numpy())
            targs.append(label.data.cpu().numpy())
        preds = np.concatenate(preds)
        targs = np.concatenate(targs)
        all_preds.append(preds)
        all_targs.append(targs)
        acc[phase] = sklearn.metrics.accuracy_score(targs.flatten(), preds.flatten())
        recall[phase] = sklearn.metrics.recall_score(targs.flatten(), preds.flatten(), average="macro", zero_division=0)
        precision[phase] = sklearn.metrics.precision_score(targs.flatten(), preds.flatten(), average="macro", zero_division=0)
        f1[phase] = sklearn.metrics.f1_score(targs.flatten(), preds.flatten(), average="macro", zero_division=0)
        for p, t in zip(preds, targs):
            confusion[phase][p, t] += 1
            confusion["all"][p, t] += 1
    all_preds = np.concatenate(all_preds)
    all_targs = np.concatenate(all_targs)
    acc["all"] = sklearn.metrics.accuracy_score(all_targs.flatten(), all_preds.flatten())
    recall["all"] = sklearn.metrics.recall_score(all_targs.flatten(), all_preds.flatten(), average="macro", zero_division=0)
    precision["all"] = sklearn.metrics.precision_score(all_targs.flatten(), all_preds.flatten(), average="macro", zero_division=0)
    f1["all"] = sklearn.metrics.f1_score(all_targs.flatten(), all_preds.flatten(), average="macro", zero_division=0)
    return (acc, precision, recall, f1, confusion)




def main(args):
    dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:%d" % args.gpu_id)
    batch_size = 1
    lr = 0.00003
    lrfe = lr * 0.25
    max_samples = 1400
    full_freeze = False
    num_texture_feats = 6
    print("Device: %s." % dev)
    test_ds_0 = ConcatDataset([
        BurnDS(os.path.join(args.data_dir, "pig1/d0"), t = 0),
        #BurnDS(os.path.join(args.data_dir, "pig6/d0"), t = 0),
    ])
    test_ds_3 = ConcatDataset([
        BurnDS(os.path.join(args.data_dir, "pig1/d3"), t = 3),
        #BurnDS(os.path.join(args.data_dir, "pig6/d3"), t = 3),
    ])
    test_ds_7 = ConcatDataset([
        BurnDS(os.path.join(args.data_dir, "pig1/d7"), t = 7),
        #BurnDS(os.path.join(args.data_dir, "pig6/d7"), t = 7),
    ])
    #test_ds_14 = ConcatDataset([
    #    BurnDS(os.path.join(args.data_dir, "pig5/d14"), t = 14),
    #    #BurnDS(os.path.join(args.data_dir, "pig6/d14"), t = 14),
    #])
    #test_ds_28 = ConcatDataset([
    #    BurnDS(os.path.join(args.data_dir, "pig3/d28"), t = 28),
    #    BurnDS(os.path.join(args.data_dir, "pig4/d28"), t = 28),
    #])
    cw = get_class_weights(test_ds_0)
    print("CW:   %s." % str(cw))
    #print("Test dataset size:  %d." % len(test_ds_0) + )
    #print("Class labels:  %s." % test_ds_0.class_to_idx)
    if args.check_full_ds:
        return
    #test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle = True, drop_last = True)
    test_dl_0 = DataLoader(test_ds_0, batch_size = batch_size, shuffle = True, drop_last = True)
    test_dl_3 = DataLoader(test_ds_3, batch_size = batch_size, shuffle = True, drop_last = True)
    test_dl_7 = DataLoader(test_ds_7, batch_size = batch_size, shuffle = True, drop_last = True)
    #test_dl_14 = DataLoader(test_ds_14, batch_size = batch_size, shuffle = True, drop_last = True)
    #test_dl_28 = DataLoader(test_ds_28, batch_size = batch_size, shuffle = True, drop_last = True)
    dataloader = {0: test_dl_0, 3: test_dl_3, 7: test_dl_7}#, 14: test_dl_14, }#28: test_dl_28}
    model = BurnClassifierType1(4, freeze_fe = full_freeze, num_texture_feats = num_texture_feats).to(dev)
    model.load_state_dict(torch.load(args.load_path))
    acc, precision, recall, f1, confusion = test_model(model, dataloader, dev)
    for k in [0, 3, 7, 14, "all"]:   #[0, 3, 7, 14, 28, "all"]:
        print("Results for %s." % str(k))
        print(confusion[k])
        print("Testing Stats.")
        print("  Accuracy:   %.4f." % acc[k])
        print("  Precision:  %.4f." % precision[k])
        print("  Recall:     %.4f." % recall[k])
        print("  F1 score:   %.4f." % f1[k])
        print()







#--------------------------------[module setup]---------------------------------
def config_cli_parser(parser):
    parser.add_argument("--cpu", help="Specify whether the CPU should be used.", type=bool, nargs='?', const=True, default=False)
    parser.add_argument("--check_full_ds", help="Stop after checking the size of the full training dataset.", type=bool, nargs='?', const=True, default=False)
    parser.add_argument("--gpu_id", help="Device ID for GPU to use if running on CUDA.", type = int, default = 0)
    parser.add_argument("--data_dir", help="", default = r"../../../ambush/Datasets/burns/iu_pig_ultrasound/")
    parser.add_argument("--load_path", help="", default = r"./resnet_text_std.pth")
    return parser



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = NAME_STR, description = DESCRIP_STR)   # Create module's cli parser.
    parser = config_cli_parser(parser)
    args = parser.parse_args()
    main(args)

#===============================================================================
