""" Some experiments using the intermedate features from a ResNet network"""
import h5py
import numpy as np
import torch
from torch import nn
from torch.utils import data
from torchvision import models, transforms

from dermosxai import datasets


def _get_intermediate_features(resnet, input_):
    """ Receives a ResNet50 network and a valid input and extracts intermediate features.

    Extracts features right after each conv or fc layer (plus one right before the last 
    fc) so 51 in total.
    
    Arguments:
        resnet (nn.Module): A ResNet50 from torchvision.models.
        input_ (torch.Tensor): Input to the Resnet (N x C x H x W)
    
    Returns:
        x (torch.Tensor): Output of the resnet (same as resnet(input_))
        features (list): A list with 51 feature representations. Each element is a 
            (N x num_features).
            
    Note:
        To make sense of it, check the original Resnet50 forward. I essentially just 
        copied everything here.
    """
    features = []

    x = resnet.conv1(input_)
    features.append(x.mean(dim=(-1, -2)).cpu().numpy())
    x = resnet.bn1(x)
    x = resnet.relu(x)
    x = resnet.maxpool(x)

    for layer in [resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]:
        for bottleneck in layer:
            identity = x

            out = bottleneck.conv1(x)
            features.append(out.mean(dim=(-1, -2)).cpu().numpy())
            out = bottleneck.bn1(out)
            out = bottleneck.relu(out)

            out = bottleneck.conv2(out)
            features.append(out.mean(dim=(-1, -2)).cpu().numpy())
            out = bottleneck.bn2(out)
            out = bottleneck.relu(out)

            out = bottleneck.conv3(out)
            features.append(out.mean(dim=(-1, -2)).cpu().numpy())
            out = bottleneck.bn3(out)

            if bottleneck.downsample is not None:
                identity = bottleneck.downsample(x)

            out += identity
            out = bottleneck.relu(out)

            #return out
            x = out

    x = resnet.avgpool(x)
    x = torch.flatten(x, 1)
    features.append(x.cpu().numpy())
    x = resnet.fc(x)
    features.append(x.cpu().numpy())

    return x, features

def create_IAD_dsets():
    """Extracts the ResNet intermediate representations and creates the h5 dsets."""
    # Load data
    train_dset = datasets.IAD('train')
    val_dset = datasets.IAD('val')
    test_dset = datasets.IAD('test')

    # Add transforms
    img_mean, img_std = train_dset.img_mean, train_dset.img_std
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(img_mean / 255, img_std / 255)])
    train_dset.transform = transform
    val_dset.transform = transform
    test_dset.transform = transform

    # Create dloaders
    batch_size = 256
    train_dloader = data.DataLoader(train_dset, batch_size=batch_size, num_workers=4)
    val_dloader = data.DataLoader(val_dset, batch_size=batch_size, num_workers=4)
    test_dloader = data.DataLoader(test_dset, batch_size=batch_size, num_workers=4)

    # Load resnet
    resnet = models.resnet50(pretrained=True)
    resnet.cuda()
    resnet.eval()

    # Get features
    for dloader, name in [(train_dloader, 'train'), (val_dloader, 'val'), (test_dloader, 'test')]:
        all_feats = []
        with torch.no_grad():
            for images, _ in dloader:
                images = images.cuda()
                _, feats = _get_intermediate_features(resnet, images)
                all_feats.append(feats)
        all_feats = [np.concatenate(fs) for fs in zip(*all_feats)]

        with h5py.File(f'/src/dermosxai/data/IAD/resnet/{name}_features.h5', 'w') as f:
            for idx, arr in enumerate(all_feats):
                f.create_dataset(str(idx), data=arr)

def create_HAM_dsets():
    """Extracts the ResNet intermediate representations and creates the h5 dsets."""
    # Load data
    train_dset = datasets.HAM10000('train')
    val_dset = datasets.HAM10000('val')
    test_dset = datasets.HAM10000('test')

    # Add transforms
    img_mean, img_std = train_dset.img_mean, train_dset.img_std
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(img_mean / 255, img_std / 255)])
    train_dset.transform = transform
    val_dset.transform = transform
    test_dset.transform = transform

    # Create dloaders
    batch_size = 256
    train_dloader = data.DataLoader(train_dset, batch_size=batch_size, num_workers=4)
    val_dloader = data.DataLoader(val_dset, batch_size=batch_size, num_workers=4)
    test_dloader = data.DataLoader(test_dset, batch_size=batch_size, num_workers=4)

    # Load resnet
    resnet = models.resnet50(pretrained=True)
    resnet.cuda()
    resnet.eval()

    # Get features
    for dloader, name in [(train_dloader, 'train'), (val_dloader, 'val'), (test_dloader, 'test')]:
        all_feats = []
        with torch.no_grad():
            for images, _ in dloader:
                images = images.cuda()
                _, feats = _get_intermediate_features(resnet, images)
                all_feats.append(feats)
        all_feats = [np.concatenate(fs) for fs in zip(*all_feats)]

        with h5py.File(f'/src/dermosxai/data/HAM10000/resnet/{name}_features.h5', 'w') as f:
            for idx, arr in enumerate(all_feats):
                f.create_dataset(str(idx), data=arr)


def create_DDSM_dsets():
    """Extracts the ResNet intermediate representations and creates the h5 dsets."""
    # Load data
    train_dset = datasets.DDSM('train')
    val_dset = datasets.DDSM('val')
    test_dset = datasets.DDSM('test')

    # Add transforms
    img_mean, img_std = train_dset.img_mean, train_dset.img_std
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(img_mean, img_std),
        transforms.Lambda(lambda x: x.expand(3, -1, -1))])
    train_dset.transform = transform
    val_dset.transform = transform
    test_dset.transform = transform

    # Create dloaders
    batch_size = 256
    train_dloader = data.DataLoader(train_dset, batch_size=batch_size, num_workers=4)
    val_dloader = data.DataLoader(val_dset, batch_size=batch_size, num_workers=4)
    test_dloader = data.DataLoader(test_dset, batch_size=batch_size, num_workers=4)

    # Load resnet
    resnet = models.resnet50(pretrained=True)
    resnet.cuda()
    resnet.eval()

    # Get features
    for dloader, name in [(train_dloader, 'train'), (val_dloader, 'val'),
                          (test_dloader, 'test')]:
        all_feats = []
        with torch.no_grad():
            for images, _ in dloader:
                images = images.cuda()
                _, feats = _get_intermediate_features(resnet, images)
                all_feats.append(feats)
        all_feats = [np.concatenate(fs) for fs in zip(*all_feats)]

        with h5py.File(f'/src/dermosxai/data/DDSM/resnet/{name}_features.h5',
                       'w') as f:
            for idx, arr in enumerate(all_feats):
                f.create_dataset(str(idx), data=arr)
