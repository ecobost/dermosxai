""" Pytorch datasets. """
import numpy as np
from torch.utils import data as pt_data

from dermosxai import data, utils


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
        transform (callable): Transform operation: receives an uint8 RGB image and returns
            the transformed image.
        one_image_per_lesion (bool): Whether only one image should be selected for each 
            lesion (there are usually diff images of the same lesion); first image in the 
            study is selected. If False, unravel all images into a single array.
        return_attributes (bool): Whether attributes should be returned along with images
            and labels
        split_proportion (tuple): Proportion of examples in the training set and 
            validation set (test set has all remaining examples).
    
    Returns:
        image (np.array): A 3-d np.uint8 array (150 x 200 x 3).
        label (int64): A single digit in [0, 3]
        (optionally) attrs (np.array): Array with attribute variables. Name of the 
            returned variables can be accessed as dset.attribute_names
    """
    def __init__(self, split='train', transform=None, one_image_per_lesion=True,
                 return_attributes=False, split_proportion=(0.8, 0.1)):
        # Load data
        images, labels = data.get_IAD()

        # Split data
        train_slice, val_slice, test_slice = utils.split_data(len(images),
                                                              split_proportion)
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

        # Get attributes (if needed)
        self.return_attributes = return_attributes
        if return_attributes:
            # Get attributes
            attributes, self.attribute_names = data.get_IAD_attributes()
            attributes = attributes.astype(np.int64)

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
        split_proportion (tuple): Proportion of examples in the training set and 
            validation set (test set has all remaining examples).
    
    Returns:
        image (np.array): A 3-d np.uint8 array (150 x 200 x 3).
        label (int64): A single digit in [0, 3]
    """
    def __init__(self, split='train', transform=None, one_image_per_lesion=True,
                 split_proportion=(0.8, 0.1)):
        # Load data
        images, labels = data.get_HAM10000()

        # Split data
        train_slice, val_slice, test_slice = utils.split_data(len(images),
                                                              split_proportion)
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


class DDSM(pt_data.Dataset):
    """ CBIS-DDSM dataset.
    
    Images of breast cancer lesions, their diagnosis and (optionally) image-derived 
    attributes: mass shape and mass margin. It has 1326 lesions with grayscale X-ray 
    images cropped and resized to 128 x 128 pixels. Each lesion is diagnosed as benign
    or malignant. Test split is provided (22%); we use 10% of the remaining training 
    images as validation (patient stratified sampling).
    
    Arguments:
        split(string): Whether to use the "train", "val" or "test" split.
        transform (callable): Transform operation: receives a grayscale float image and 
            returns the transformed image.
        return_attributes (bool): Whether attributes should be returned along with images
            and labels.
    
    Returns:
        image (np.array): A 2-d np.float32 array (128 x 128).
        label (int64): Whether mass was benign or malignant.
        (optionally) attrs (np.array): Array with attributes (shape and margin) as 
            categorical variables (0-n).
    """
    def __init__(self, split='train', transform=None, return_attributes=False):
        # Load data
        images, labels, attributes = data.get_DDSM()

        # Split data
        if split not in ['train', 'val', 'test']:
            raise ValueError(f'split can only be train/val/test, not {split}')
        self.images = images[split]
        self.labels = labels[split].astype(np.int64)

        # Get attributes (if needed)
        self.return_attributes = return_attributes
        if return_attributes:
            self.attributes = attributes[split].astype(np.int64)
            #self.attribute_names = None

        # Save transform
        self.transform = transform

    @property
    def img_mean(self):
        return self.images.astype(np.float64).mean()

    @property
    def img_std(self):
        return self.images.astype(np.float64).std(axis=(1, 2)).mean()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        example = (self.images[i], self.labels[i])
        if self.transform is not None:
            example = (self.transform(example[0]), example[1])
        if self.return_attributes:
            example = (*example, self.attributes[i])

        return example
