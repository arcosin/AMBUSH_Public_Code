
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
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import sklearn.metrics
from skimage.feature import greycomatrix, greycoprops
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold

from torchvision.utils import save_image

from common import *


# Constants.
NAME_STR = ""
DESCRIP_STR = ""







def train_model(model, dataloader, optimizer, num_epochs, dev, cw = None):
    start = time.time()
    if cw is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=cw.to(dev), reduction='mean')
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    #best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    if "valid" in dataloader and "train" in dataloader:  # Bad code but i'll fix later.
        phases = ["train", "valid"]
    elif "valid" in dataloader:
        phases = ["valid"]
    else:
        phases = ["train"]
    for epoch in range(num_epochs):
        print("Epoch %d / %d." % (epoch + 1, num_epochs))
        for phase in phases:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0.0
            for b_mode, b_texture, tdi_mode, tdi_stats, tdi_bar, label, day in tqdm(dataloader[phase]):
                #if phase == "train":
                    #b_mode = b_mode + torch.randn_like(b_mode) * 0.3
                b_mode = b_mode.to(dev)
                b_texture = b_texture.to(dev)
                tdi_mode = tdi_mode.to(dev)
                tdi_stats = tdi_stats.to(dev)
                tdi_bar = tdi_bar.to(dev)
                label = label.to(dev)
                day = day.to(dev)
                with torch.set_grad_enabled(phase == "train"):
                    if phase == "train":
                        optimizer.zero_grad()
                        outputs = model(b_mode, b_texture, tdi_mode, tdi_stats, tdi_bar, day)
                        loss = criterion(outputs, label)
                        _, predictions = torch.max(outputs, 1)
                        loss.backward()
                        optimizer.step()
                    else:
                        outputs = model(b_mode, b_texture, tdi_mode, tdi_stats, tdi_bar, day)
                        loss = criterion(outputs, label)
                        _, predictions = torch.max(outputs, 1)
                running_loss += loss.item()
                running_corrects += torch.sum(predictions == label.data)
            epoch_loss = running_loss / len(dataloader[phase].dataset)
            epoch_acc = (running_corrects / len(dataloader[phase].dataset)).cpu().numpy()
            print("  %s   Loss: %.4f   Acc: %.4f." % (phase.capitalize(), epoch_loss, epoch_acc))
            if phase == "train":
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc.item())
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc.item())
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
        print()
    train_time = time.time() - start
    return (model, train_loss, train_acc, val_loss, val_acc)




def get_inds_from_region(inds, val):
    res = []
    neg_res = []
    for k, v in inds.items():
        if k.endswith(str(val)):
            res += v
        else:
            neg_res += v
    return (res, neg_res)





def make_trainplot(save_dir, epochs, train_loss, val_loss, train_acc, val_acc, p = "./training_plot.png"):
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, color="skyblue", label="Train")
    plt.plot(val_loss, color="orange", label="Val")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, color="skyblue", label="Train")
    plt.plot(val_acc, color="orange", label="Val")
    plt.legend()
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.savefig(os.path.join(save_dir, p))
    plt.clf()





def main(args):
    dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:%d" % args.gpu_id)
    num_models = 10
    sds_size = 8000
    batch_size = 32
    lr = 0.0005#0.00003
    lrfe = lr * 0.25
    max_samples = 1400
    full_freeze = False
    num_texture_feats = 6
    train_on = False
    print("Device: %s." % dev)
    aug = transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
        ])
    dss = [
        BurnDS(os.path.join(args.data_dir, "pig1/d0"), aug_transform_b = aug, t = 0),
        BurnDS(os.path.join(args.data_dir, "pig1/d3"), aug_transform_b = aug, t = 3),
        BurnDS(os.path.join(args.data_dir, "pig1/d7"), aug_transform_b = aug, t = 7),
        BurnDS(os.path.join(args.data_dir, "pig1/d14"), aug_transform_b = aug, t = 14),
        #BurnDS(os.path.join(args.data_dir, "pig1/d28"), aug_transform_b = aug, t = 28),
        #BurnDS(os.path.join(args.data_dir, "pig1/d35"), aug_transform_b = aug, t = 35),
        #BurnDS(os.path.join(args.data_dir, "pig1/d42"), aug_transform_b = aug, t = 42),
        BurnDS(os.path.join(args.data_dir, "pig2/d0"), aug_transform_b = aug, t = 0),
        BurnDS(os.path.join(args.data_dir, "pig2/d3"), aug_transform_b = aug, t = 3),
        BurnDS(os.path.join(args.data_dir, "pig2/d7"), aug_transform_b = aug, t = 7),
        BurnDS(os.path.join(args.data_dir, "pig2/d14"), aug_transform_b = aug, t = 14),
        #BurnDS(os.path.join(args.data_dir, "pig2/d28"), aug_transform_b = aug, t = 28),
        #BurnDS(os.path.join(args.data_dir, "pig2/d35"), aug_transform_b = aug, t = 35),
        #BurnDS(os.path.join(args.data_dir, "pig2/d42"), aug_transform_b = aug, t = 42),
        BurnDS(os.path.join(args.data_dir, "pig3/d0"), aug_transform_b = aug, t = 0),
        BurnDS(os.path.join(args.data_dir, "pig3/d3"), aug_transform_b = aug, t = 3),
        BurnDS(os.path.join(args.data_dir, "pig3/d7"), aug_transform_b = aug, t = 7),
        BurnDS(os.path.join(args.data_dir, "pig3/d14"), aug_transform_b = aug, t = 14),
        #BurnDS(os.path.join(args.data_dir, "pig3/d28"), aug_transform_b = aug, t = 28),
        #BurnDS(os.path.join(args.data_dir, "pig3/d35"), aug_transform_b = aug, t = 35),
        #BurnDS(os.path.join(args.data_dir, "pig3/d42"), aug_transform_b = aug, t = 42),
        BurnDS(os.path.join(args.data_dir, "pig6/d0"), aug_transform_b = aug, t = 0),
        BurnDS(os.path.join(args.data_dir, "pig6/d3"), aug_transform_b = aug, t = 3),
        BurnDS(os.path.join(args.data_dir, "pig6/d7"), aug_transform_b = aug, t = 7),
        BurnDS(os.path.join(args.data_dir, "pig6/d14"), aug_transform_b = aug, t = 14),
    ]
    valid_dss = [
        BurnDS(os.path.join(args.data_dir, "pig4/d0"), aug_transform_b = aug, t = 0),
        BurnDS(os.path.join(args.data_dir, "pig4/d3"), aug_transform_b = aug, t = 3),
        BurnDS(os.path.join(args.data_dir, "pig4/d7"), aug_transform_b = aug, t = 7),
        BurnDS(os.path.join(args.data_dir, "pig4/d14"), aug_transform_b = aug, t = 14),
        BurnDS(os.path.join(args.data_dir, "pig5/d0"), aug_transform_b = aug, t = 0),
        BurnDS(os.path.join(args.data_dir, "pig5/d3"), aug_transform_b = aug, t = 3),
        BurnDS(os.path.join(args.data_dir, "pig5/d7"), aug_transform_b = aug, t = 7),
        BurnDS(os.path.join(args.data_dir, "pig5/d14"), aug_transform_b = aug, t = 14),
    ]
    train_ds = ConcatDataset(dss)
    valid_ds = ConcatDataset(valid_dss)
    print("Train dataset size (full):  %d." % len(train_ds))
    valid_dl = DataLoader(valid_ds, batch_size = batch_size, shuffle = False, drop_last = False)
    print("Valid dataset size (split):  %d." % len(valid_ds))
    #cw = get_class_weights(train_ds)
    #print("CW:   %s." % str(cw))
    models = nn.ModuleList([])
    if train_on:
        for m in range(num_models):
            train_ds_sub = Subset(train_ds, random.sample(range(0, len(train_ds)), sds_size))
            print("Train dataset size (split):  %d." % len(train_ds_sub))
            train_dl = DataLoader(train_ds_sub, batch_size = batch_size, shuffle = True, drop_last = True)
            dataloader = {"train": train_dl, "valid": valid_dl}
            model = BurnClassifierEns(4).to(dev)
            model_pars = model.params_split()
            if full_freeze:
                opt = optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-4)
            else:
                pars = [{'params': model_pars["fc"]}, {'params': model_pars["fe"], 'lr': lrfe}]
                opt = optim.Adam(pars, lr = lr, weight_decay = 1e-4)
            model, train_l, train_a, val_l, val_a = train_model(model, dataloader, opt, args.epochs, dev)
            models.append(model)
            print("Model %d." % m)
            make_trainplot(args.save_dir, args.epochs, train_l, val_l, train_a, val_a, p = "./training_plot_mod%d.png" % m)
            print("Train loss:  %s.\n" % train_l)
            print("Train accu:  %s.\n" % train_a)
            print("Valid loss:  %s.\n" % val_l)
            print("Valid accu:  %s.\n" % val_a)
        torch.save(models.state_dict(), os.path.join(args.save_dir, "bagger.pth"))
    else:
        for m in range(num_models):
            model = BurnClassifierEns(4).to(dev)
            models.append(model)
        models.load_state_dict(torch.load(os.path.join(args.save_dir, "bagger.pth")))
    model = Bagger(4, models, dev).to(dev)
    dataloader = {"valid": valid_dl}
    _, train_l, train_a, val_l, val_a = train_model(model, dataloader, None, 1, dev)
    print("Bagging.")
    make_trainplot(args.save_dir, args.epochs, train_l, val_l, train_a, val_a, p = "./training_plot_bagging.png")
    print("Train loss:  %s.\n" % train_l)
    print("Train accu:  %s.\n" % train_a)
    print("Valid loss:  %s.\n" % val_l)
    print("Valid accu:  %s.\n" % val_a)
    print("Done.\n")











def flatten(l):
    return [item for sublist in l for item in sublist]


def main_crossval(args):
    dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:%d" % args.gpu_id)
    batch_size = 32
    lr = 0.0002
    lrfe = lr * 0.25
    max_samples = 1400
    full_freeze = False
    num_texture_feats = 6
    print("Device: %s." % dev)
    aug = transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0)])
    val_size = 2
    for fold_id in range(6):
        tr_on = list(range(6))
        tr_on = tr_on[:fold_id] + tr_on[fold_id + val_size:]
        print("Training on %s, validating on %d." % (str(tr_on), fold_id))
        dss = [
            [BurnDS(os.path.join(args.data_dir, "pig1/d0"), aug_transform_b = aug, t = 0),
            BurnDS(os.path.join(args.data_dir, "pig1/d3"), aug_transform_b = aug, t = 3)],
            [BurnDS(os.path.join(args.data_dir, "pig2/d0"), aug_transform_b = aug, t = 0),
            BurnDS(os.path.join(args.data_dir, "pig2/d3"), aug_transform_b = aug, t = 3)],
            [BurnDS(os.path.join(args.data_dir, "pig3/d0"), aug_transform_b = aug, t = 0),
            BurnDS(os.path.join(args.data_dir, "pig3/d3"), aug_transform_b = aug, t = 3)],
            [BurnDS(os.path.join(args.data_dir, "pig4/d0"), aug_transform_b = aug, t = 0),
            BurnDS(os.path.join(args.data_dir, "pig4/d3"), aug_transform_b = aug, t = 3)],
            [BurnDS(os.path.join(args.data_dir, "pig5/d0"), aug_transform_b = aug, t = 0),
            BurnDS(os.path.join(args.data_dir, "pig5/d3"), aug_transform_b = aug, t = 3)],
            [BurnDS(os.path.join(args.data_dir, "pig6/d0"), aug_transform_b = aug, t = 0),
            BurnDS(os.path.join(args.data_dir, "pig6/d3"), aug_transform_b = aug, t = 3)],
        ]
        train_ds = ConcatDataset(flatten(dss[:fold_id] + dss[fold_id + val_size:]))
        valid_ds = ConcatDataset(dss[fold_id])
        print("Train dataset size (full):  %d." % len(train_ds))
        train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True, drop_last = True)
        valid_dl = DataLoader(valid_ds, batch_size = batch_size, shuffle = False, drop_last = False)
        dataloader = {"train": train_dl, "valid": valid_dl}
        cw = get_class_weights(train_ds)
        print("CW:   %s." % str(cw))
        model = BurnClassifierType1(4, freeze_fe = full_freeze, num_texture_feats = num_texture_feats).to(dev)
        model_pars = model.params_split()
        if full_freeze:
            opt = optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-5)
        else:
            pars = [{'params': model_pars["fc"]}, {'params': model_pars["fe"], 'lr': lrfe}]
            opt = optim.Adam(pars, lr = lr, weight_decay = 1e-5)
        model, train_l, train_a, val_l, val_a = train_model(model, dataloader, opt, args.epochs, dev, cw = cw)
        torch.save(model.state_dict(), os.path.join(args.save_dir, "model_fold_%d.pth" % fold_id))
        make_trainplot(args.save_dir, args.epochs, train_l, val_l, train_a, val_a, p = "./training_plot_fold_%d.png" % fold_id)
        print("Train loss:  %s.\n" % train_l)
        print("Train accu:  %s.\n" % train_a)
        print("Valid loss:  %s.\n" % val_l)
        print("Valid accu:  %s.\n" % val_a)
    print("Done.\n")














#--------------------------------[module setup]---------------------------------
def config_cli_parser(parser):
    parser.add_argument("--cpu", help="Specify whether the CPU should be used.", type=bool, nargs='?', const=True, default=False)
    parser.add_argument("--check_full_ds", help="Stop after checking the size of the full training dataset.", type=bool, nargs='?', const=True, default=False)
    parser.add_argument("--validate", help="Perform cross validation.", type=bool, nargs='?', const=True, default=False)
    parser.add_argument("--vid_validate", help="Perform cross validation on video level.", type=bool, nargs='?', const=True, default=False)
    parser.add_argument("--gpu_id", help="Device ID for GPU to use if running on CUDA.", type = int, default = 0)
    parser.add_argument("--fold_file", help="JSON file holding fold list for validation.", default = r"./folds_0-3-7_8294.json")
    parser.add_argument("--data_dir", help="", default = r"../../../ambush/Datasets/burns/iu_pig_ultrasound/")
    parser.add_argument("--save_dir", help="", default = r"./")
    parser.add_argument("--val_fold", help="Fold to use for validation.", type = int, default = 0)
    parser.add_argument("--epochs", help="Epochs to train for.", type = int, default = 100)
    return parser



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = NAME_STR, description = DESCRIP_STR)   # Create module's cli parser.
    parser = config_cli_parser(parser)
    args = parser.parse_args()
    #main_crossval(args)
    main(args)

#===============================================================================
