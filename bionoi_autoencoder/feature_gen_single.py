import torch
import argparse
import pickle
import numpy as np
import torchvision.transforms as transforms
from torch import nn
import os
from os import listdir
from os.path import isfile, join, isdir
from utils import UnsuperviseDataset, ConvAutoencoder_conv1x1
from dataset_statistics import dataset_statistics
from collections import OrderedDict


def get_args():
    parser = argparse.ArgumentParser("python")
    parser.add_argument(
        "-input_fname",
        default="/home/tony/sandbox/pdb/bionoi/output/3D2R.mol2_XOY-_r0_OO.png",
        required=False,
        help="directory of training images",
    )
    parser.add_argument(
        "-output_fname",
        default="/home/tony/sandbox/pdb/bionoi/ae_feature/pdk4.pkl",
        required=False,
        help="directory of generated features",
    )
    parser.add_argument(
        "-model_file",
        default="./bionoi_autoencoder/trained_model/conv1x1-4M-batch512.pt",
        required=False,
        help="directory of trained model",
    )
    parser.add_argument(
        "-gpu_to_cpu",
        default=False,
        required=False,
        help="whether to reconstruct image using model made with gpu on a cpu",
    )

    return parser.parse_args()


def load_model(gpu_to_cpu):
    model = ConvAutoencoder_conv1x1()
    if gpu_to_cpu == True:
        # original saved file with DataParallel
        print("GPU to CPU true...")
        state_dict = torch.load(model_file, map_location="cpu")
        # create new OrderedDict that does not contain `module.`

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)

    else:
        # if there are multiple GPUs, split the batch to different GPUs
        if torch.cuda.device_count() > 1:
            print("Using " + str(torch.cuda.device_count()) + " GPUs...")
            model = nn.DataParallel(model)
        model.load_state_dict(torch.load(model_file))
    return model


def get_preprocessed(input_fname):
    data_mean = [0.6027, 0.4405, 0.6577]
    data_std = [0.2341, 0.1866, 0.2513]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (data_mean[0], data_mean[1], data_mean[2]),
                (data_std[0], data_std[1], data_std[2]),
            ),
        ]
    )
    data_dir, fname = os.path.split(input_fname)
    dataset = UnsuperviseDataset(
        data_dir=f"{data_dir}/", filename_list=[fname], transform=transform
    )
    image, file_name = dataset[0]
    return image, file_name


def feature_vec_gen(device, model, image, output_fname):
    """
    Generate feature vectors for a single image
    """
    image = image.to(device)
    if device.type == "cuda":
        feature_vec = model.module.encode_vec(image.unsqueeze(0))  # add one dimension
    else:
        feature_vec = model.encode_vec(
            image.unsqueeze(0)
        )  # choose this line if running on cpu only machine

    with open(output_fname, "wb") as f:
        pickle.dump(feature_vec, f)


if __name__ == "__main__":
    args = get_args()
    input_fname = args.input_fname
    output_fname = args.output_fname
    model_file = args.model_file
    gpu_to_cpu = args.gpu_to_cpu

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Current device: " + str(device))

    model = load_model(gpu_to_cpu=gpu_to_cpu)
    image, file_name = get_preprocessed(input_fname=input_fname)

    # calulate the input size (flattened)
    print("name of input:", file_name)
    image_shape = image.shape
    print("shape of input:", image_shape)

    # generate features for images in data_dir
    model = model.to(device)
    model.eval()  # don't cache the intermediate values
    print("generating feature vectors...")
    feature_vec_gen(device=device, model=model, image=image, output_fname=output_fname)
