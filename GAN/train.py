
import torch
import numpy as np
import vegans.utils as utils
import vegans.utils.loading as loading
from vegans.GAN import ConditionalVanillaGAN, ConditionalWassersteinGANGP
import matplotlib.pyplot as plt

from common import *

DUMB_TEST = False
TRAIN = True
TEST = True

num_examples = 8
class_map = ["full_thickness", "partial_thickness", "no_burn", "superficial_thickness"]

# Data preparation (Load your own data or example MNIST)
#loader = loading.MNISTLoader()
#X_train, y_train, X_test, y_test = loader.load()

bs = 8
data_dir = r"../../../data/frames/"
aug = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
    ])
train_ds_0 = BurnDS_B(os.path.join(data_dir, "train_d0"), aug_transform_b = aug, t = 0)
train_ds_3 = BurnDS_B(os.path.join(data_dir, "train_d3"), aug_transform_b = aug, t = 3)
train_ds_7 = BurnDS_B(os.path.join(data_dir, "train_d7"), aug_transform_b = aug, t = 7)
train_ds = ConcatDataset([train_ds_0, train_ds_3, train_ds_7])
test_ds_0 = BurnDS_B(os.path.join(data_dir, "test_d0"), t = 0)
test_ds_3 = BurnDS_B(os.path.join(data_dir, "test_d3"), t = 3)
test_ds_7 = BurnDS_B(os.path.join(data_dir, "test_d7"), t = 7)
test_ds = ConcatDataset([test_ds_0, test_ds_3, test_ds_7])
print("Train dataset size (full):  %d." % len(train_ds))
print("Test dataset size (full):  %d." % len(test_ds))
train_dl = DataLoader(train_ds, batch_size = bs, shuffle = True, drop_last = True)
test_dl = DataLoader(test_ds, batch_size = bs, shuffle = False, drop_last = False)



x_dim = [1, 512, 512]#list(train_ds_0[0][0].size()) # [3, 224, 420]
y_dim = 4
z_dim = 64


if TRAIN:
    # Define your own architectures here. You can use a Sequential model or an object
    # inheriting from torch.nn.Module. Here, a default model for mnist is loaded.
    generator = load_generator(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)
    discriminator = load_adversary(x_dim=x_dim, y_dim=y_dim)
    gan = ConditionalWassersteinGANGP(
        generator=generator, adversary=discriminator,
        z_dim=z_dim, x_dim=x_dim, y_dim=y_dim,
        folder="./", # optional
        optim={"Generator": torch.optim.RMSprop, "Adversary": torch.optim.Adam}, # optional
        optim_kwargs={"Generator": {"lr": 0.001}, "Adversary": {"lr": 0.001}}, # optional
        fixed_noise_size=32, # optional
        device=None, # optional
        ngpu=0 # optional
    )
    gan.summary() # optional, shows architecture
    # Training
    gan.fit(
        train_dl, None, test_dl, None,
        epochs=60, # optional
        batch_size=bs, # optional
        steps={"Generator": 1, "Adversary": 2}, # optional, train generator once and discriminator twice on every mini-batch
        print_every="0.1e", # optional, prints progress 10 times per epoch
        # (might also be integer input indicating number of mini-batches)
        save_model_every=None, # optional
        save_images_every="1e", # optional
        save_losses_every="0.5e", # optional, save losses in internal losses dictionary used to generate
        # plots during and after training
        enable_tensorboard=False # optional, if true all progress is additionally saved in tensorboard subdirectory
    )
    # Vizualise results
    images, losses = gan.get_training_results()
    utils.plot_images(images, labels=np.argmax(gan.fixed_labels.cpu().numpy(), axis=1), show=False)
    plt.savefig("training.png")
    plt.clf()
    utils.plot_losses(losses, show=False)
    plt.savefig("losses.png")
    plt.clf()
    # Save model
    gan.save("./gan.pt")



if TEST:
    gan = torch.load("./gan.pt")
    gan.summary()
    # Generate specific label, for example "2"
    for e in range(num_examples):
        for i in range(4):
            label = torch.tensor([[0, 0, 0, 0]]).to("cuda:0")
            label[0][i] = 1
            image = gan(y=label)
            utils.plot_images(image, labels=[class_map[i]], show=False)
            plt.savefig("example%d_class%d.png" % (e, i))
            plt.clf()
    print("Examples generated.")




if DUMB_TEST:
    train_dl = DataLoader(train_ds, batch_size = 1, shuffle = True, drop_last = True)
    for e in range(25):
        for b_mode, label in train_dl:
            _, i = label[0].topk(1)
            label = i.item()
            print(b_mode.size())
            utils.plot_images(b_mode.numpy(), labels=[class_map[label]], show=False)
            plt.savefig("real_example%d_class%d.png" % (e, label))
            plt.clf()
            break


#===============================================================================
