
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
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler, ConcatDataset
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.utils import class_weight
from vegans.utils import get_input_dim
from vegans.utils.layers import LayerReshape, LayerPrintSize






def crop_b_mode(img):
    return transforms.functional.crop(img, 140, 30, 240, 450)
    #return transforms.functional.crop(img, 55, 45, 360, 350)

def crop_tdi_mode(img):
    return transforms.functional.crop(img, 140, 480, 240, 460)
    #return transforms.functional.crop(img, 55, 410, 360, 350)

def crop_b_tdi_mode(img):
    return transforms.functional.crop(img, 100, 30, 280, 900)

def crop_bar(img):
    return transforms.functional.crop(img, 80, 470, 60, 30)
    #return transforms.functional.crop(img, 58, 399, 63, 9)

def recolor_tdi(img, dev = None):
    colors = torch.FloatTensor([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
    ])
    if dev is not None:
        colors = colors.to(dev)
    colors_normalized = colors / 255
    ret = quantize(img, colors_normalized)
    return ret

def recolor_batch(imgs, dev = None):
    colors = torch.FloatTensor([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
    ])
    if dev is not None:
        colors = colors.to(dev)
    colors_normalized = colors / 255
    for i in range(imgs.size(0)):
        img = imgs[i]
        imgs[i] = quantize(img, colors_normalized)
    return imgs

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
    cw = class_weight.compute_class_weight('balanced', classes = np.unique(targs), y = targs.numpy())
    return torch.tensor(cw, dtype = torch.float)

def append_dropout(model, rate = 0.2):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            append_dropout(module)
        if isinstance(module, nn.ReLU):
            new_model = nn.Sequential(module, nn.Dropout2d(p=rate, inplace=False))
            setattr(model, name, new_model)








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

def seperate_ds(train_dataset):
    targ_list = get_targets(train_dataset)
    targ_unq = torch.unique(targ_list)
    dss = []
    for t in targ_unq:
        cla = np.where(np.array(targ_list) == np.array(t))[0]
        dss.append(Subset(train_dataset, cla))
    return dss







class BurnDS(datasets.ImageFolder):
    def __init__(self, root, aug_transform_b = (lambda x: x), aug_transform_tdi = (lambda x: x), feat_mode = "all", t = 0):
        super().__init__(root)
        self.feat_mode = feat_mode
        self.t = t
        self.b_mode_pre = transforms.Compose([
                crop_b_mode,
                aug_transform_b,
                transforms.Resize((128, 256)),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.tdi_mode_pre = transforms.Compose([
                crop_tdi_mode,
                aug_transform_tdi,
                transforms.Resize((128, 256)),
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
        tdi_mode = self.tdi_mode_pre(inp)
        tdi_stats = ImageStat.Stat(crop_tdi_mode(inp))
        tdi_stats_list = list(tdi_stats.mean) + list(tdi_stats.median) + list(tdi_stats.stddev)
        tdi_stats_vec = torch.FloatTensor(tdi_stats_list)
        tdi_bar_mean = torch.mean(self.tdi_bar_pre(inp)).unsqueeze(0)
        md = statistics.mean([abs(tdi_stats.mean[0] - tdi_stats.mean[1]), abs(tdi_stats.mean[0] - tdi_stats.mean[2]), abs(tdi_stats.mean[1] - tdi_stats.mean[2])])
        tdi_flag = True
        if md < 2.0:
            tdi_mode = torch.zeros_like(tdi_mode)
            tdi_flag = False
        #plt.imsave("temp.png", b_mode.permute(1,2,0).cpu().numpy())
        day = torch.FloatTensor([self.t])
        return (b_mode, tdi_mode, tdi_flag, tdi_stats_vec, tdi_bar_mean, label, day)

    def get_video_indices(self):
        vidmap = defaultdict(list)
        for i, img in enumerate(self.imgs):
            fpath = os.path.basename(img[0])
            prefix = fpath.split('_')[0]
            vidmap[prefix].append(i)
        return vidmap






class BurnDS_B(datasets.ImageFolder):
    def __init__(self, root, aug_transform_b = (lambda x: x), t = 0):
        super().__init__(root)
        self.t = t
        self.b_mode_pre = transforms.Compose([
                crop_b_mode,
                aug_transform_b,
                transforms.Resize((512,512)),
                #transforms.Resize((224, 420)),
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        inp, label = super(BurnDS_B, self).__getitem__(index)
        label = F.one_hot(torch.tensor([label]), num_classes = 4).squeeze(0)
        path = self.imgs[index][0]
        b_mode = self.b_mode_pre(inp)
        day = torch.FloatTensor([self.t])
        return (b_mode, label)




class P(nn.Module):
    def forward(self, x):
        print(x.size())
        return x

'''
class Generator(nn.Module):
    def __init__(self, x_dim, gen_in_dim, dr = 0.2):
        super().__init__()
        if len(gen_in_dim) == 1:
            out_shape = (128, 4, 4)
            self.linear_part = nn.Sequential(
                nn.Linear(in_features=gen_in_dim[0], out_features=256),
                nn.LeakyReLU(0.1),
                nn.Dropout(dr),
                nn.Linear(in_features=256, out_features=np.prod(out_shape)),
                nn.LeakyReLU(0.1),
                nn.Dropout(dr),
                LayerReshape(shape=out_shape),
            )
            gen_in_dim = out_shape
        else:
            self.linear_part = nn.Identity()
        self.hidden_part = nn.Sequential(
            nn.ConvTranspose2d(in_channels=gen_in_dim[0], out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dr),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum = 0.01),
            nn.LeakyReLU(0.1),
            nn.Dropout(dr),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dr),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum = 0.01),
            nn.LeakyReLU(0.1),
            nn.Dropout(dr),
        )
        desired_output = x_dim[1]
        current_output = gen_in_dim[1]
        in_channels = 128
        i = 3
        while current_output != desired_output:
            out_channels = in_channels // 2
            current_output *= 2
            if current_output != desired_output:
                self.hidden_part.add_module("ConvTraspose{}".format(i), nn.ConvTranspose2d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1
                    )
                )
                self.hidden_part.add_module("Batchnorm{}".format(i), nn.BatchNorm2d(num_features=out_channels))
                self.hidden_part.add_module("LeakyRelu{}".format(i), nn.LeakyReLU(0.1))
                self.hidden_part.add_module("Dropout{}".format(i), nn.Dropout(dr))
                self.hidden_part.add_module("Conv{}".format(i), nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),)
                self.hidden_part.add_module("Batchnorm_again{}".format(i), nn.BatchNorm2d(num_features=out_channels))
                self.hidden_part.add_module("LeakyRelu_again{}".format(i), nn.LeakyReLU(0.1))
                self.hidden_part.add_module("Dropout{}".format(i), nn.Dropout(dr))
            else: # Last layer
                self.hidden_part.add_module("ConvTraspose{}".format(i), nn.ConvTranspose2d(in_channels=in_channels, out_channels=3, kernel_size=4, stride=2, padding=1))
                self.hidden_part.add_module("Batchnorm{}".format(i), nn.BatchNorm2d(num_features=3))
                self.hidden_part.add_module("LeakyRelu{}".format(i), nn.LeakyReLU(0.1))
                self.hidden_part.add_module("Dropout{}".format(i), nn.Dropout(dr))
                self.hidden_part.add_module("Conv{}".format(i), nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),)
            in_channels = in_channels // 2
            i += 1
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.linear_part(x)
        x = self.hidden_part(x)
        return self.output(x)
'''


def up_block_type1(ch_in, ch_out, dr = 0.2):
    x = nn.Sequential(
        nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(ch_out, momentum = 0.01),
        nn.LeakyReLU(0.1),
        nn.Dropout(dr),
        nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(ch_out, momentum = 0.01),
        nn.LeakyReLU(0.1),
        nn.Dropout(dr),
    )
    return x


def up_block_type2(ch_in, ch_out, dr = 0.2):
    x = nn.Sequential(
        nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(ch_out, momentum = 0.01),
        nn.LeakyReLU(0.1),
        nn.Dropout(dr),
        nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(ch_out, momentum = 0.01),
        nn.LeakyReLU(0.1),
        nn.Dropout(dr),
    )
    return x



def down_block_type1(ch_in, ch_out, dr = 0.2):
    x = nn.Sequential(
        nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(ch_out, momentum = 0.01),
        nn.LeakyReLU(0.1),
        nn.Dropout(dr),
        nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(ch_out, momentum = 0.01),
        nn.LeakyReLU(0.1),
        nn.Dropout(dr),
    )
    return x


def down_block_type2(ch_in, ch_out, dr = 0.2):
    x = nn.Sequential(
        nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(ch_out, momentum = 0.01),
        nn.LeakyReLU(0.1),
        nn.Dropout(dr),
        nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(ch_out, momentum = 0.01),
        nn.LeakyReLU(0.1),
        nn.Dropout(dr),
    )
    return x


class Upsample(nn.Module):
    def forward(self, x):
        F.interpolate(x, scale_factor = 2, mode='bilinear')

class Downsample(nn.Module):
    def forward(self, x):
        F.interpolate(x, scale_factor = 0.5, mode='bilinear')




class Generator(nn.Module):
    def __init__(self, x_dim, gen_in_dim):
        super().__init__()
        out_shape = (128, 4, 4)
        self.fc = nn.Sequential(
            nn.Linear(in_features=gen_in_dim[0], out_features=256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(in_features=256, out_features=np.prod(out_shape)),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            LayerReshape(shape=out_shape),
        )
        self.block1 = up_block_type2(out_shape[0], 128)   # 8x8.
        self.block2 = up_block_type2(128, 64)            # 16x16.
        self.block3 = up_block_type2(64, 64)            # 32x32.
        self.block4 = up_block_type1(64, 16)            # 64x64.
        self.block5 = up_block_type1(16, 16)             # 128x128.
        self.block6 = up_block_type1(16, 4)              # 256x256.
        self.block7 = up_block_type1(4, 4)               # 512x512.
        self.out_block = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.block1(x) + F.interpolate(x, scale_factor = 2, mode='bilinear')
        x = self.block2(x)
        x = self.block3(x) + F.interpolate(x, scale_factor = 2, mode='bilinear')
        x = self.block4(x)
        x = self.block5(x) + F.interpolate(x, scale_factor = 2, mode='bilinear')
        x = self.block6(x)
        x = self.block7(x) + F.interpolate(x, scale_factor = 2, mode='bilinear')
        x = self.out_block(x)
        return F.sigmoid(x)






def load_generator(x_dim, z_dim, y_dim=None):
    z_dim = [z_dim] if isinstance(z_dim, int) else z_dim
    y_dim = tuple([y_dim]) if isinstance(y_dim, int) else y_dim
    if len(z_dim) == 3:
        assert z_dim[1] % 2 == 0, "z_dim[1] must be divisible by 2. Given: {}.".format(z_dim[1])
        assert x_dim[1] % 2 == 0, "`x_dim[1]` must be divisible by 2. Given: {}.".format(x_dim[1])
        assert x_dim[1] % z_dim[1] == 0, "`x_dim[1]` must be divisible by `z_dim[1]`. Given: {} and {}.".format(x_dim[1], z_dim[1])
        assert (x_dim[1] / z_dim[1]) % 2 == 0, "`x_dim[1]/z_dim[1]` must be divisible by 2. Given: {} and {}.".format(x_dim[1], z_dim[1])
        assert z_dim[1] == z_dim[2], "`z_dim[1]` must be equal to `z_dim[2]`. Given: {} and {}.".format(z_dim[1], z_dim[2])
    gen_in_dim = get_input_dim(dim1=z_dim, dim2=y_dim) if y_dim is not None else z_dim
    return Generator(x_dim=x_dim, gen_in_dim=gen_in_dim)






class Adversary(nn.Module):
    def __init__(self, adv_in_dim, last_layer_activation):
        super().__init__()
        self.block1 = down_block_type2(adv_in_dim[0], 4)
        self.block2 = down_block_type2(4, 4)
        self.block3 = down_block_type2(4, 16)
        self.block4 = down_block_type1(16, 16)
        self.block5 = down_block_type1(16, 64)
        self.block6 = down_block_type1(64, 64)
        self.block7 = down_block_type1(64, 128)
        self.out_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=np.prod((128, 4, 4)), out_features=128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(in_features=128, out_features=1)
        )
        self.out_act = last_layer_activation()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x) + F.interpolate(x, scale_factor = 0.5, mode='bilinear')
        x = self.block3(x)
        x = self.block4(x) + F.interpolate(x, scale_factor = 0.5, mode='bilinear')
        x = self.block5(x)
        x = self.block6(x) + F.interpolate(x, scale_factor = 0.5, mode='bilinear')
        x = self.block7(x)
        x = self.out_block(x)
        return self.out_act(x)



'''
class Adversary(nn.Module):
    def __init__(self, adv_in_dim, last_layer_activation, dr = 0.2):
        super().__init__()
        self.hidden_part = nn.Sequential(
            nn.Conv2d(in_channels=adv_in_dim[0], out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
        )
        while True:
            current_output = self.hidden_part(torch.randn(size=(2, *adv_in_dim))).shape
            if np.prod(current_output) > 10000:
                self.hidden_part.add_module(str(len(self.hidden_part) + 1 ), nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1))
                self.hidden_part.add_module(str(len(self.hidden_part) + 1 ), nn.ReLU())
                self.hidden_part.add_module(str(len(self.hidden_part) + 1 ), nn.Dropout(dr))
                self.hidden_part.add_module(str(len(self.hidden_part) + 1 ), nn.MaxPool2d(kernel_size=4, stride=2, padding=1))
                self.hidden_part.add_module(str(len(self.hidden_part) + 1 ), nn.BatchNorm2d(num_features=256))
            else:
                self.hidden_part.add_module(str(len(self.hidden_part) + 1 ), nn.Flatten())
                current_output = self.hidden_part(torch.randn(size=(2, *adv_in_dim))).shape
                break
        self.linear_part = nn.Sequential(
            nn.Linear(in_features=current_output[1], out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1),
        )
        self.output = last_layer_activation()

    def forward(self, x):
        x = self.hidden_part(x)
        x = self.linear_part(x)
        return self.output(x)
'''



def load_adversary(x_dim, y_dim=None, adv_type="Critic"):
    possible_types = ["Discriminator", "Critic"]
    if adv_type == "Critic":
        last_layer_activation = nn.Identity
    elif adv_type == "Discriminator":
        last_layer_activation = nn.Sigmoid
    else:
        raise ValueError("'adv_type' must be one of: {}.".format(possible_types))
    adv_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim) if y_dim is not None else x_dim
    return Adversary(adv_in_dim=adv_in_dim, last_layer_activation=last_layer_activation)




#===============================================================================
