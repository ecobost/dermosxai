""" Pytorch datasets. """
from torch.utils import data as pt_data
import numpy as np

from dermosxai import data
from dermosxai import utils

class IAD(pt_data.Dataset):
    """ Interactive Atlas of Dermoscopy (IAD) dataset.
    
    Images of dermoscopic lessions and their diagnosis (and optionally image-derived 
    attributes of the lesion). It has 986 lesions with >=1 RGB images of the lesion 
    (cropped and resized to 150 x 200 pixels). Each patient is diagnosed with one of four
    labels: 0: Nevus, 1: Melanoma, 2: Keratosis or 3: Basal cell carcinoma.
    Dataset was already randomized so to split it it into train/val/test sets we pick the 
    first 80% of patients as train, other 10% as val and rest as test set.
    
    Arguments:
        split(string): Whether to use the "train", "val" or "test" split.
        transform (callable): Transform operation: receives an RGB image and returns the 
            transformed image.
        one_image_per_lesion (bool): Whether only one image should be selected for each 
            lesion (there are usually diff images of the same lesion); first image in the 
            study is selected. If False, unravel all images into a single array.
        return_attributes (bool): Whether attributes should be returned along with images
            and labels
    
    Returns:
        image (np.array): A 3-d np.uint8 array (150 x 200 x 3).
        label (int64): A single digit in [0, 3]
        (optionally) attrs (np.array): Array with attribute variables. Name of the 
            returned variables can be accessed as dset.attribute_names
    """
    def __init__(self, split='train', transform=None, one_image_per_lesion=True,
                 return_attributes=False):
        # Load data
        images, labels = data.get_IAD()

        # Split data
        train_slice, val_slice, test_slice = utils.split_data(len(images))
        if split == 'train':
            split_slice = train_slice
        elif split == 'val':
            split_slice = val_slice
        elif split == 'test':
            split_slice = test_slice
        else:
            raise ValueError('split has to be one of train, val or test.')
        self.images = images[split_slice]
        self.labels = labels[split_slice]

        # Deal with each lesion having a diff number of images.
        if one_image_per_lesion: # pick the first image for each lesion
            self.images = np.stack([lesion[0] for lesion in self.images])
        else:
            num_images_per_lesion = [len(lesion) for lesion in self.images]
            self.images = np.stack([im for lesion in self.images for im in lesion])
            self.labels = np.repeat(self.labels, num_images_per_lesion)

        # Get attributes (if needed)
        self.return_attributes = return_attributes
        if return_attributes:
            # Get attributes
            attributes, self.attribute_names = data.get_IAD_attributes()

            # Split attributes
            self.attributes = attributes[split_slice]

            # Deal with each lesion having a diff number of images.
            if not one_image_per_lesion:
                num_images_per_lesion = [len(lesion) for lesion in images[split_slice]]
                self.attributes = np.repeat(self.attributes, num_images_per_lesion,
                                            axis=0)

        # Save transform
        self.transform = transform

    @property
    def img_mean(self):
        """ Channel-wise mean. """
        return self.images.mean(axis=(0, 1, 2))

    @property
    def img_std(self):
        """ Channel-wise std. """
        return self.images.std(axis=(1, 2)).mean(0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        example = (self.images[i], self.labels[i])
        if self.transform is not None:
            example = (self.transform(example[0]), example[1])
        if self.return_attributes:
            example = (*example, self.attributes[i])

        return example


class HAM10000(pt_data.Dataset):
    """ HAM10000 dataset (Tschandl et al., 2018).
    
    Images of dermoscopic lesions and their diagnosis. It has 7470 lesions with >=1 RGB 
    images (resized to 150 x 200 pixels). Each patient is diagnosed with one of four 
    labels: 0: Nevus, 1: Melanoma, 2: Keratosis or 3: Basal cell carcinoma.
    Dataset was already randomized so to split it it into train/val/test sets we pick the 
    first 80% of patients as train, other 10% as val and rest as test set.
    
    Arguments:
        split(string): Whether to return the "train", "val" or "test" split.
        transform (callable): Image-wise transform operation: receives a (hxwx3) uint8 RGB
            image and returns the transformed image.
        one_image_per_lesion (bool): Whether only one image should be selected for each 
            lesion (there are usually > 1 images of the same lesion); first image in the 
            study is selected. If False, unravel all images into a single array.
    
    Returns:
        image (np.array): A 3-d np.uint8 array (150 x 200 x 3).
        label (int64): A single digit in [0, 3]
    """
    def __init__(self, split='train', transform=None, one_image_per_lesion=True):
        # Load data
        images, labels = data.get_HAM10000()

        # Split data
        train_slice, val_slice, test_slice = utils.split_data(len(images))
        if split == 'train':
            split_slice = train_slice
        elif split == 'val':
            split_slice = val_slice
        elif split == 'test':
            split_slice = test_slice
        else:
            raise ValueError('split has to be one of train, val or test.')
        self.images = images[split_slice]
        self.labels = labels[split_slice]

        # Deal with each lesion having a diff number of images.
        if one_image_per_lesion:  # pick the first image for each lesion
            self.images = np.stack([lesion[0] for lesion in self.images])
        else:
            num_images_per_lesion = [len(lesion) for lesion in self.images]
            self.images = np.stack([im for lesion in self.images for im in lesion])
            self.labels = np.repeat(self.labels, num_images_per_lesion)

        # Save transform
        self.transform = transform

    @property
    def img_mean(self):
        """ Channel-wise mean. """
        return self.images.mean(axis=(0, 1, 2))

    @property
    def img_std(self):
        """ Channel-wise std. """
        return self.images.std(axis=(1, 2)).mean(0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        example = (self.images[i], self.labels[i])
        if self.transform is not None:
            example = (self.transform(example[0]), example[1])

        return example

    
class StackDataset(pt_data.Dataset):
    """ Stacks datasets along the batch axis. 
    
    Arguments:
        datasets: Datasets to stack.
    """
    def __init__(self, *datasets):
        self.datasets = datasets
        self.dset_lengths = [len(dset) for dset in datasets]
    
    def __len__(self):
        sum(self.dset_lengths)
    
    def __getitem__(self, i):
        pass
        #TODO: Find which dset to sample from and return that
        

class ConcatDataset(pt_data.Dataset):
    """ Concatenate datasets.
    
    Arguments:
        datasets: Datasets to concatenate. They should all have the same length.
        
    Returns:
        (ex1, ex2, ..., exn): A tuple with as many entries as there are datasets.
    """
    def __init__(self, *datasets):
        if any([len(dset) != len(datasets[0]) for dset in datasets]):
            raise ValueError('All datasets must have the same size.')
        
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])
    
    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)
        