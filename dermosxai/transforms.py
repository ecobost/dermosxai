""" Some extra transforms not provided in torchvision and default transforms for our 
dsets."""
import torch
from torchvision import transforms


class RandomRotate90():
    def __call__(self, x):
        return torch.rot90(x, k=torch.randint(4, []), dims=(-1, -2))

class Expandto3Channels():
    def __call__(self, x):
        return x.expand(3, -1, -1)


def get_DDSM_transforms(img_mean, img_std, make_rgb=True):
    """ Get the train and test transformation for DDSM datasets.
    
    Standard transformations: RandomHorizontalFlip, RandomVerticalFlip, Random90Rotation,
        Normalization and ToTensor.
    
    Arguments:
        img_mean, img_std: Mean and std values used for the normalization. Usually taken
            per-channel from the training set.
        make_rgb (bool): Whether to copy the inputimage so it has 3 channels.
    """
    extra_transforms = [Expandto3Channels()] if make_rgb else []
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomRotate90(),
        transforms.Normalize(img_mean, img_std), *extra_transforms])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(img_mean, img_std), *extra_transforms])

    return train_transform, test_transform


def get_IAD_transforms(img_mean, img_std, use_full_augmentations=False):
    """ Get the train and test transformation for IAD datasets.
    
    Standard transformations: RandomHorizontalFlip, RandomVerticalFlip, Normalization and
        ToTensor.
    
    
    Arguments:
        img_mean, img_std: Mean and std values used for the normalization. Usually taken
            per-channel from the training set.
        use_full_augmentations (bool): Whether to use some affine transformations and 
            color jittering.
    """
    if use_full_augmentations:
        extra_transforms = [
            transforms.RandomAffine(degrees=90, shear=20, scale=(1, 1.4),
                                    translate=(0.03, 0.03), fill=list(img_mean / 255)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)]
    else:
        extra_transforms = []
    train_transform = transforms.Compose([
        transforms.ToTensor(),  # changes range of images to [0, 1]
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        *extra_transforms,
        transforms.Normalize(img_mean / 255, img_std / 255)])

    test_transform = transforms.Compose([
        transforms.ToTensor(),  # changes range of images to [0, 1]
        transforms.Normalize(img_mean / 255, img_std / 255)])

    return train_transform, test_transform