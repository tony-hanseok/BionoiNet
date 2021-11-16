"""
Train an autoencoder for bionoi datasets.
User can specify the types (dense-style or cnn-style) and options (denoising, sparse, contractive).
"""
import torch
from torch import mean, nn
import torch.optim as optim
from torch.utils import data
import torchvision.transforms as transforms
import argparse
import os.path
import os
from os import listdir
from os.path import isfile, join, isdir
import matplotlib.pyplot as plt
from utils import UnsuperviseDataset, train
from utils import ConvAutoencoder_conv1x1
from helper import imshow, list_plot
from dataset_statistics import dataset_statistics


def get_args():
    parser = argparse.ArgumentParser("python")

    parser.add_argument(
        "-seed",
        default=123,
        type=int,
        required=False,
        help="seed for random number generation",
    )

    parser.add_argument(
        "-epoch",
        default=100,
        type=int,
        required=False,
        help="number of epochs to train",
    )

    parser.add_argument(
        "-feature_size",
        default=256,
        type=int,
        required=False,
        help="size of output feature of the autoencoder",
    )

    parser.add_argument(
        "-data_dir",
        default="/mnt/aitrics_ext/ext01/tony/bionoi_output/residue_type/homology_reduced_mols/",
        required=False,
        help="directory of training images",
    )

    parser.add_argument(
        "-model_file",
        default="./bionoi_autoencoder/trained_model/bionoi_autoencoder_conv2.pt",
        required=False,
        help="file to save the model",
    )

    parser.add_argument(
        "-batch_size",
        default=256,
        type=int,
        required=False,
        help="the batch size, normally 2^n.",
    )

    parser.add_argument(
        "-normalize", default=True, required=False, help="whether to normalize dataset"
    )

    parser.add_argument(
        "-num_data",
        type=int,
        default=50000,
        required=False,
        help="the batch size, normally 2^n.",
    )

    parser.add_argument(
        "-num_workers",
        default=20,
        required=False,
        help="the number of workers for dataloader",
    )

    return parser.parse_args()


def get_transform(data_dir, num_data):
    statistics = dataset_statistics(data_dir, 128, num_data)
    data_mean = statistics[0].tolist()
    data_std = statistics[1].tolist()

    print("normalizing data:")
    print("mean:", data_mean)
    print("std:", data_std)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (data_mean[0], data_mean[1], data_mean[2]),
                (data_std[0], data_std[1], data_std[2]),
            ),
        ]
    )
    return transform, data_mean, data_std


def get_dataloader(data_dir, transform, batch_size, num_workers):
    img_list = []
    for item in listdir(data_dir):
        if isfile(join(data_dir, item)):
            img_list.append(item)
        elif isdir(join(data_dir, item)):
            update_data_dir = join(data_dir, item)
            for f in listdir(update_data_dir):
                if isfile(join(update_data_dir, f)):
                    img_list.append(item + "/" + f)
                elif isdir(join(update_data_dir, f)):
                    deeper_data_dir = join(update_data_dir, f)
                    for y in listdir(deeper_data_dir):
                        if isfile(join(deeper_data_dir, y)):
                            img_list.append(item + "/" + f + "/" + y)

    dataset = UnsuperviseDataset(data_dir, img_list, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return dataloader


def get_model():
    model = ConvAutoencoder_conv1x1()
    # if there are multiple GPUs, split the batch to different GPUs
    if torch.cuda.device_count() > 1:
        print("Using " + str(torch.cuda.device_count()) + " GPUs...")
        model = nn.DataParallel(model)
    return model


def get_optimizers(model):
    # loss function
    criterion = nn.MSELoss()

    # optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-10,
        weight_decay=0.0001,
        amsgrad=False,
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 20], gamma=0.5
    )
    return criterion, optimizer, scheduler


if __name__ == "__main__":
    args = get_args()
    seed = args.seed
    num_epochs = args.epoch
    data_dir = args.data_dir
    model_file = args.model_file
    batch_size = args.batch_size
    normalize = args.normalize
    feature_size = args.feature_size
    num_data = args.num_data
    num_workers = args.num_workers

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Current device: " + str(device))

    # define a transform with mean and std, another transform without them.
    # statistics obtained from dataset_statistics.py
    transform, data_mean, data_std = get_transform(data_dir=data_dir, num_data=num_data)

    # forming list of images. images may be upto 3 directories deep
    # any deeper and the data_dir must be changed or another layer added to the code
    dataloader = get_dataloader(
        data_dir=data_dir,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # get some random training images to show
    # dataiter = iter(dataloader)
    # images, filename = dataiter.next()

    # print(images.shape)
    # imshow(torchvision.utils.make_grid(images))
    # images = images.view(batch_size,-1)
    # images = torch.reshape(images,(images.size(0),3,256,256))
    # imshow(torchvision.utils.make_grid(images))

    # image_shape = images.shape
    # print('shape of input:', image_shape)

    # instantiate model
    model = get_model()
    print(str(model))  # print model info

    # print the paramters to train
    print("paramters to train:")
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print(str(name))

    # get criterion, optimizer, scheduler
    criterion, optimizer, lr_scheduler = get_optimizers(model=model)

    # begin training
    trained_model, train_loss_history = train(
        device=device,
        num_epochs=num_epochs,
        dataloader=dataloader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    # save the model
    base, fname = os.path.split(model_file)
    if not os.path.isdir(base):
        os.makedirs(base)

    torch.save(trained_model.state_dict(), os.path.join(base, f"{fname}.pt"))

    with open(os.path.join(base, "stats.txt"), "w") as f:
        f.write(f"mean: {data_mean[0]}\t{data_mean[1]}\t{data_mean[2]}\n")
        f.write(f"std: {data_std[0]}\t{data_std[1]}\t{data_std[2]}")

    with open(os.path.join(base, "losses.txt"), "w") as f:
        f.write(f"epoch\tloss\n")
        for i, loss in enumerate(train_loss_history):
            f.write(f"{i+1}\t{loss}\n")

    # plot the loss function vs epoch
    list_plot(train_loss_history)
    base = os.path.splitext(model_file)[-1]
    plt.savefig(os.path.join(base, "loss.png"))
    # plt.show()
