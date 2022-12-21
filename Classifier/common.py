
import os
import random
import statistics
from collections import defaultdict
from PIL import Image, ImageStat, ImagePalette
from PIL.ImagePalette import ImagePalette

from tqdm import tqdm
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler, ConcatDataset
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import cv2
import sklearn.metrics
from sklearn.utils import class_weight
from skimage.feature import greycomatrix, greycoprops





def crop_b_mode(img):
    return transforms.functional.crop(img, 140, 30, 240, 250) #-200h

def crop_tdi_mode(img):
    return transforms.functional.crop(img, 140, 480, 240, 460)

def crop_b_tdi_mode(img):
    return transforms.functional.crop(img, 100, 30, 280, 900)

def crop_bar(img):
    return transforms.functional.crop(img, 80, 470, 60, 30)

def recolor_tdi(img):
    colors = torch.tensor([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255]
    ])
    colors_normalized = colors / 255
    return quantize(img, colors_normalized)

def distance(color_a, color_b):
    differences = torch.stack([(c_a - c_b) ** 2 for c_a, c_b in zip(color_a, color_b)])
    return torch.sum(differences, dim = -2) ** 0.5

#https://discuss.pytorch.org/t/color-quantization/104528
def quantize(image, palette):
    flat_img = image.view(1, 3,-1) # [C, H, W] -> [1, C, H*W]
    img_per_palette_color = torch.cat(len(palette)*[flat_img]) # [1, C, H*W] -> [n_colors, C, H*W]
    distance_per_pixel = distance(img_per_palette_color, palette.unsqueeze(-1)) # [n_colors, C, H*W] -> [n_colors, H*W]
    color_indices = torch.argmin(distance_per_pixel, dim=0) # [n_colors, H*W] -> [H*W]
    new_colors = palette[color_indices].T # [H*W] -> [C, H*W]
    return new_colors.view(image.shape) # [C, H*W] -> [C, H, W]

def get_targets(ds):
    if isinstance(ds, Subset):
        targets = get_targets(ds.dataset)
        return torch.as_tensor(targets)[ds.indices]
    if isinstance(ds, ConcatDataset):
        return torch.cat([get_targets(sub_dataset) for sub_dataset in ds.datasets])
    if isinstance(ds, (datasets.MNIST, datasets.ImageFolder)):
        return torch.as_tensor(ds.targets)
    if isinstance(ds, datasets.SVHN):
        return ds.labels
    raise NotImplementedError(f"Unknown dataset {dataset}!")

def get_class_weights(ds):
    targs = get_targets(ds)
    print("counts  ", targs.unique(return_counts=True)[1])
    cw = class_weight.compute_class_weight('balanced', classes = np.unique(targs), y = targs.numpy())
    return torch.tensor(cw, dtype = torch.float)

def append_dropout(model, rate = 0.2):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            append_dropout(module)
        if isinstance(module, nn.ReLU):
            new_model = nn.Sequential(module, nn.Dropout2d(p=rate, inplace=False))
            setattr(model, name, new_model)

class PrintBlock(nn.Module):
    def forward(self, x):
        #print(x.size())
        return x







def trim_ds(train_dataset, n):
    class1 = np.where(np.array(train_dataset.targets) == 0)[0]
    try:
        class1 = random.sample(class1.tolist(), n)
    except ValueError:
        class1 = class1.tolist()
    count1 = len(class1)
    class2 = np.where(np.array(train_dataset.targets) == 1)[0]
    try:
        class2 = random.sample(class2.tolist(), n)
    except ValueError:
        class2 = class2.tolist()
    count2 = len(class2)
    class3 = np.where(np.array(train_dataset.targets) == 2)[0]
    try:
        class3 = random.sample(class3.tolist(), n)
    except ValueError:
        class3 = class3.tolist()
    count3 = len(class3)
    class4 = np.where(np.array(train_dataset.targets) == 3)[0]
    try:
        class4 = random.sample(class4.tolist(), n)
    except ValueError:
        class4 = class4.tolist()
    count4 = len(class4)
    subset1 = Subset(train_dataset, class1)
    subset2 = Subset(train_dataset, class2)
    subset3 = Subset(train_dataset, class3)
    subset4 = Subset(train_dataset, class4)
    return (ConcatDataset([subset1, subset2, subset3, subset4]), [count1, count2, count3, count4])






class BurnDS(datasets.ImageFolder):
    def __init__(self, root, aug_transform_b = (lambda x: x), aug_transform_tdi = (lambda x: x), feat_mode = "contrast", t = 0):
        super().__init__(root)
        self.feat_mode = feat_mode
        self.t = t
        self.b_mode_pre = transforms.Compose([
                crop_b_mode,
                aug_transform_b,
                transforms.Resize((224,224)), #transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.tdi_mode_pre = transforms.Compose([
                crop_tdi_mode,
                aug_transform_tdi,
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                recolor_tdi,
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.tdi_bar_pre = transforms.Compose([
                crop_bar,
                transforms.Resize((9,1)),
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        inp, label = super(BurnDS, self).__getitem__(index)
        path = self.imgs[index][0]
        b_mode = self.b_mode_pre(inp)
        b_texture = extract_text_features(b_mode, self.feat_mode)
        tdi_mode = self.tdi_mode_pre(inp)
        tdi_stats = ImageStat.Stat(crop_tdi_mode(inp))
        tdi_stats_list = list(tdi_stats.mean) + list(tdi_stats.median) + list(tdi_stats.stddev)
        tdi_stats_vec = torch.FloatTensor(tdi_stats_list)
        tdi_bar_mean = torch.mean(self.tdi_bar_pre(inp)).unsqueeze(0)
        md = statistics.mean([abs(tdi_stats.mean[0] - tdi_stats.mean[1]), abs(tdi_stats.mean[0] - tdi_stats.mean[2]), abs(tdi_stats.mean[1] - tdi_stats.mean[2])])
        if md < 2.0:
            tdi_mode = torch.zeros_like(tdi_mode)
        day = torch.FloatTensor([self.t])
        return (b_mode, b_texture, tdi_mode, tdi_stats_vec, tdi_bar_mean, label, day)

    #def get_video_indices(self):
    #    vidmap = defaultdict(list)
    #    for i, img in enumerate(self.imgs):
    #        fpath = os.path.basename(img[0])
    #        prefix = fpath.split('_')[0]
    #        vidmap[prefix].append(i)
    #    return vidmap




class BurnDS_TDI(datasets.ImageFolder):
    def __init__(self, root, aug_transform_b = (lambda x: x), aug_transform_tdi = (lambda x: x), feat_mode = "contrast", t = 0):
        super().__init__(root)
        self.feat_mode = feat_mode
        self.t = t
        self.tdi_mode_pre = transforms.Compose([
                crop_tdi_mode,
                aug_transform_tdi,
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                recolor_tdi,
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.tdi_bar_pre = transforms.Compose([
                crop_bar,
                transforms.Resize((9,1)),
                transforms.ToTensor()
            ])
        self.tdi_inds = []
        print("Trimming DS %s." % root)
        rem_count = 0
        for index in range(super(BurnDS_TDI, self).__len__()):
            inp, label = super(BurnDS_TDI, self).__getitem__(index)
            tdi_mode = self.tdi_mode_pre(inp)
            tdi_stats = ImageStat.Stat(crop_tdi_mode(inp))
            tdi_stats_list = list(tdi_stats.mean) + list(tdi_stats.median) + list(tdi_stats.stddev)
            tdi_stats_vec = torch.FloatTensor(tdi_stats_list)
            tdi_bar_mean = torch.mean(self.tdi_bar_pre(inp)).unsqueeze(0)
            md = statistics.mean([abs(tdi_stats.mean[0] - tdi_stats.mean[1]), abs(tdi_stats.mean[0] - tdi_stats.mean[2]), abs(tdi_stats.mean[1] - tdi_stats.mean[2])])
            if md > 2.0:
                self.tdi_inds.append(index)
            else:
                rem_count += 1
        print("Removed %d." % rem_count)

    def __len__(self):
        return len(self.tdi_inds)

    def __getitem__(self, index):
        inp, label = super(BurnDS_TDI, self).__getitem__(self.tdi_inds[index])
        #path = self.imgs[self.tdi_inds[index]][0]
        tdi_mode = self.tdi_mode_pre(inp)
        tdi_stats = ImageStat.Stat(crop_tdi_mode(inp))
        tdi_stats_list = list(tdi_stats.mean) + list(tdi_stats.median) + list(tdi_stats.stddev)
        tdi_stats_vec = torch.FloatTensor(tdi_stats_list)
        tdi_bar_mean = torch.mean(self.tdi_bar_pre(inp)).unsqueeze(0)
        md = statistics.mean([abs(tdi_stats.mean[0] - tdi_stats.mean[1]), abs(tdi_stats.mean[0] - tdi_stats.mean[2]), abs(tdi_stats.mean[1] - tdi_stats.mean[2])])
        assert md > 2.0
        day = torch.FloatTensor([self.t])
        return (0, 0, tdi_mode, tdi_stats_vec, tdi_bar_mean, label, day)







def extract_text_features(img_tensor, feat_mode):
    numpy_img = img_tensor.cpu().detach().permute(1,2,0).numpy()
    image = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2GRAY)
    im_array = np.array(image, dtype=np.uint8)[:300,:]
    dist = [1, 2]
    angles = [0, np.pi / 4, np.pi / 2]
    g = greycomatrix(im_array, dist, angles, 256, symmetric=True, normed=True)
    contrast = greycoprops(g, "contrast")
    homogeneity = greycoprops(g, "homogeneity")
    ASM = greycoprops(g, "ASM")
    energy = greycoprops(g, "energy")
    dissimilarity = greycoprops(g, "dissimilarity")
    text_features = []
    if feat_mode == "contrast":
        for i in range(len(dist)):
            for j in range(len(angles)):
                text_features.append(contrast[i][j])
    elif feat_mode == "homogeneity":
        for i in range(len(dist)):
            for j in range(len(angles)):
                text_features.append(homogeneity[i][j])
    elif feat_mode == "asm":
        for i in range(len(dist)):
            for j in range(len(angles)):
                text_features.append(ASM[i][j])
    elif feat_mode == "energy":
        for i in range(len(dist)):
            for j in range(len(angles)):
                text_features.append(energy[i][j])
    elif feat_mode == "dissimilarity":
        for i in range(len(dist)):
            for j in range(len(angles)):
                text_features.append(dissimilarity[i][j])
    else:
        for i in range(len(dist)):
            for j in range(len(angles)):
                text_features.append(contrast[i][j])
                text_features.append(homogeneity[i][j])
                text_features.append(ASM[i][j])
                text_features.append(energy[i][j])
                text_features.append(dissimilarity[i][j])
    feat_array = np.expand_dims(np.array((text_features)), 1)
    features = torch.from_numpy(feat_array).type(torch.FloatTensor)
    features = torch.squeeze(features, 1)
    return features







class BurnClassifierType1(nn.Module):
    def __init__(self, num_classes, freeze_fe = True, fc1_width = 512, droprate = 0.3, num_texture_feats = 30):
        super().__init__()
        self.num_classes = num_classes
        self.dr = droprate
        self.num_texture_feats = num_texture_feats
        self.fe_feats = 512#196   #5824
        self.fe = torch.nn.Sequential(*(list(models.resnet18(pretrained=True).children())[:-1]))
        #self.fe = torch.nn.Sequential(
        #    nn.Conv2d(3, 32, 4, padding = 1, stride = 2),
        #    nn.ReLU(),
        #    nn.MaxPool2d(2),
        #    nn.Conv2d(32, 16, 4, padding = 1, stride = 2),
        #    nn.ReLU(),
        #    nn.MaxPool2d(2),
        #    nn.Conv2d(16, 4, 4, padding = 1, stride = 2),
        #    nn.ReLU(),
        #    )
        #self.fe = torch.nn.Sequential(*(list(models.vgg16_bn(pretrained=True).children())[:-1]))
        if freeze_fe:
            for p in self.fe.parameters():
                p.requires_grad = False
            append_dropout(self.fe, droprate)
        self.fc1 = nn.Linear(in_features=self.fe_feats * 1 + num_texture_feats + 1, out_features=fc1_width)
        #self.fc1 = nn.Linear(in_features=self.fe_feats + 2, out_features=fc1_width)
        self.fc2 = nn.Linear(in_features=fc1_width, out_features=self.num_classes)

    def forward(self, b_mode, b_texture, tdi_mode, tdi_stats, tdi_bar, day):
        x1 = self.fe(b_mode)
        x1 = torch.flatten(x1, 1)
        if self.num_texture_feats == 0:
            x = torch.cat([x1, day], dim = 1)
        else:
            x = torch.cat([x1, b_texture, day], dim = 1)
        x = nn.functional.dropout(x, self.dr)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, self.dr)
        x = self.fc2(x)
        return x

    def params_split(self):
        #return {"fe": list(self.fe1.parameters()) + list(self.fe2.parameters()), "fc": list(self.fc1.parameters()) + list(self.fc2.parameters())}
        return {"fe": list(self.fe.parameters()), "fc": list(self.fc1.parameters()) + list(self.fc2.parameters())}






class BurnClassifierType2(nn.Module):
    def __init__(self, num_classes, freeze_fe = False, fc1_width = 512, droprate = 0.2, num_texture_feats = 30):
        super().__init__()
        self.num_classes = num_classes
        self.dr = droprate
        self.fe1 = torch.nn.Sequential(*(list(models.resnet18(pretrained=True).children())[:-1]))
        if freeze_fe:
            for p in self.fe1.parameters():
                p.requires_grad = False
        else:
            append_dropout(self.fe1, droprate * 0.5)
        self.fe2 = torch.nn.Sequential(*(list(models.resnet18(pretrained=True).children())[:-1]))
        if freeze_fe:
            for p in self.fe2.parameters():
                p.requires_grad = False
        else:
            append_dropout(self.fe2, droprate * 0.5)
        self.fc1 = nn.Linear(in_features=512 * 2 + num_texture_feats + 1, out_features=fc1_width)
        self.fc2 = nn.Linear(in_features=fc1_width, out_features=self.num_classes)

    def forward(self, b_mode, b_texture, tdi_mode, tdi_stats, tdi_bar, day):
        x1 = self.fe1(b_mode)
        x1 = torch.flatten(x1, 1)
        x2 = self.fe2(tdi_mode)
        x2 = torch.flatten(x2, 1)
        x = torch.cat([x1, x2, b_texture, day], dim = 1)
        x = nn.functional.dropout(x, self.dr)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, self.dr)
        x = self.fc2(x)
        return x

    def params_split(self):
        return {"fe": list(self.fe1.parameters()) + list(self.fe2.parameters()), "fc": list(self.fc1.parameters()) + list(self.fc2.parameters())}






class BurnClassifierType3(nn.Module):
    def __init__(self, num_classes, freeze_fe = True, fc1_width = 512, droprate = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.dr = droprate
        self.fe_feats = 512#196   #5824
        self.fe = torch.nn.Sequential(*(list(models.resnet18(pretrained=True).children())[:-1]))
        if freeze_fe:
            for p in self.fe.parameters():
                p.requires_grad = False
            append_dropout(self.fe, droprate)
        self.fc1 = nn.Linear(in_features=self.fe_feats + 2, out_features=fc1_width)
        self.fc2 = nn.Linear(in_features=fc1_width, out_features=self.num_classes)

    def forward(self, b_mode, b_texture, tdi_mode, tdi_stats, tdi_bar, day):
        x1 = self.fe(tdi_mode)
        x1 = torch.flatten(x1, 1)
        x = torch.cat([x1, tdi_bar, day], dim = 1)
        x = nn.functional.dropout(x, self.dr)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, self.dr)
        x = self.fc2(x)
        return x

    def params_split(self):
        return {"fe": list(self.fe.parameters()), "fc": list(self.fc1.parameters()) + list(self.fc2.parameters())}






class BurnClassifierEns(nn.Module):
    def __init__(self, num_classes, droprate = 0.2, num_texture_feats = 30):
        super().__init__()
        self.num_classes = num_classes
        self.dr = droprate
        self.cnn = torch.nn.Sequential(
            PrintBlock(),
            nn.Conv2d(3, 4, 4, padding = 1, stride = 2),   # 112.
            nn.BatchNorm2d(4),
            nn.ReLU(),
            PrintBlock(),
            nn.Dropout2d(p=droprate),
            nn.Conv2d(4, 8, 4, padding = 1, stride = 2),   # 56.
            nn.BatchNorm2d(8),
            nn.ReLU(),
            PrintBlock(),
            nn.Dropout2d(p=droprate),
            nn.Conv2d(8, 16, 4, padding = 1, stride = 2),   # 28.
            nn.BatchNorm2d(16),
            nn.ReLU(),
            PrintBlock(),
            nn.Dropout2d(p=droprate),
            nn.Conv2d(16, 32, 4, padding = 1, stride = 2),   # 14.
            nn.BatchNorm2d(32),
            nn.ReLU(),
            PrintBlock(),
            nn.Dropout2d(p=droprate),
            nn.Conv2d(32, 16, 4, padding = 1, stride = 2),   # 7.
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(p=droprate),
            PrintBlock(),
        )
        self.fc = nn.Linear(in_features=7*7*16 + 1, out_features=self.num_classes)

    def forward(self, b_mode, b_texture, tdi_mode, tdi_stats, tdi_bar, day):
        x = self.cnn(b_mode)
        x = torch.flatten(x, 1)
        x = torch.cat([x, day], dim = 1)
        return self.fc(x)

    def params_split(self):
        return {"fe": list(self.cnn.parameters()), "fc": list(self.fc.parameters())}




class Bagger(nn.Module):
    def __init__(self, num_classes, models, dev):
        super().__init__()
        self.num_classes = num_classes
        self.models = models
        self.dev = dev

    def forward(self, b_mode, b_texture, tdi_mode, tdi_stats, tdi_bar, day):
        activ = nn.Softmax(dim=1)
        start = torch.FloatTensor([0.0] * self.num_classes).unsqueeze(0).repeat(b_mode.size(0), 1).to(self.dev)
        for model in self.models:
            start += activ(model(b_mode, b_texture, tdi_mode, tdi_stats, tdi_bar, day))
        y = torch.div(start, float(len(self.models)))
        return y



#===============================================================================
