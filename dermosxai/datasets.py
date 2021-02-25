""" Pytorch datasets. """
from torch.utils import data as pt_data
import numpy as np

from dermosxai import data

class IAD(pt_data.Dataset):
    """ Interactive Atlas of Dermoscopy (IAD) dataset.
    
    Images of dermoscopic lessions and their diagnosis (and optionally image-derived 
    attributes of the lesion). It has 986 patients with >=1 RGB images of the lesion 
    (at 150 x 200 pixels). Each patient is diagnosed with one of four labels: 0: Nevus, 
    1: Melanoma, 2: Keratosis or 3: Basal cell carcinoma.
    Dataset was already randomized so to split it it into train/val/test sets we pick the 
    first 80% of patients as train, other 10% as val and rest as test set.
    
    Arguments:
        split(string): Whether to use the "train", "val" or "test" split.
        normalize (bool): Whether to normalize images using channel-wise training mean and 
            std. If return_attributes(see below) is True, it also normalizes any attribute
            that is not a binary variable. Values used for this normalization can be 
            accessed as train_img_mean, train_img_std and train_attr_mean and 
            train_attr_std.
        transform (callable): Transform operation: receives an RGB image and returns the 
            transformed image.
        one_image_per_lesion (bool): Whether only one image should be selected for each 
            patient (they are usually diff images of the same lesion); first image in the 
            study is selected. If False, unravel all images into a single array.
        return_attributes (bool): Whether attributes should be returned along with images
            and labels
    
    Returns:
        image (np.array): A 3-d np.float32 array (150 x 200 x 3).
        label (int64): A single digit in [0, 3]
        (optionally) attrs (np.array): Array with attribute variables. Name of the 
            returned variables can be accessed as dset.attribute_names; whether each 
            attribute is a binary dummy variable can be accessed as 
            dset.is_attribute_binary
    """
    def __init__(self, split='train', normalize=True, transform=None,
                 one_image_per_lesion=True, return_attributes=False):
        # Load data
        images, labels = data.get_IAD()

        # Set up training splits
        train_slice = slice(int(round(0.8 * len(images))))  # first 80%
        val_slice = slice(train_slice.stop, int(round(0.9 * len(images))))  # 80-90%
        test_slice = slice(val_slice.stop, None)  # 90%-100%

        # Compute image mean and standard deviation in the training set (used for normalization)
        if one_image_per_lesion:
            train_images = np.stack([lesion[0] for lesion in images[train_slice]])
        else:
            train_images = np.stack([im for lesion in images[train_slice] for im in lesion])
        self.train_img_mean = train_images.mean(axis=(0, 1, 2))  # this could overflow a float32
        self.train_img_std = train_images.std(axis=(1, 2)).mean(0)

        # Split data
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

        # Normalize images
        if normalize:
            self.images = (self.images - self.train_img_mean) / self.train_img_std

        # Make sure images are np.float32
        self.images = self.images.astype(np.float32)

        # Get attributes (if needed)
        self.return_attributes = return_attributes
        if return_attributes:
            # Get attributes
            attributes, self.attribute_names, self.is_attribute_binary = data.get_IAD_attributes()

            # Compute attribute training stats
            train_attributes = attributes[train_slice]
            if not one_image_per_lesion:
                num_images_per_lesion = [len(lesion) for lesion in images[train_slice]]
                train_attributes = np.repeat(train_attributes, num_images_per_lesion,
                                             axis=0)
            self.train_attr_mean = train_attributes.mean(axis=0)
            self.train_attr_std = train_attributes.std(axis=0)

            # Split attributes
            self.attributes = attributes[split_slice]
            if not one_image_per_lesion:
                num_images_per_lesion = [len(lesion) for lesion in images[split_slice]]
                self.attributes = np.repeat(self.attributes, num_images_per_lesion,
                                            axis=0)

            # Normalize
            if normalize:
                norm_attributes = ((self.attributes - self.train_attr_mean) /
                                   self.train_attr_std)[:, ~self.is_attribute_binary]
                self.attributes[:, ~self.is_attribute_binary] = norm_attributes
        
        # Save transform
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        example = (self.images[i], self.labels[i])
        if self.transform is not None:
            example = (self.transform(example[0]), example[1])
        if self.return_attributes:
            example = (*example, self.attributes[i])

        return example


# class HAM1000()
#  def __init__():
#  	# Make sure I use the HAM1000 official valisatio nd test set.

#  	# maybe sample the test set from the training set as we don't have access to the original test set.
