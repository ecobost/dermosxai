""" Interface to our imaging data. Loads and returns data in a standard format."""
import h5py
from os import path
import numpy as np
import pandas as pd


# Set some important directories
base_dir = '/src/dermosxai/data'
IAD_dir = path.join(base_dir, 'IAD')
HAM_dir = path.join(base_dir, 'HAM10000')

# Map dataset diagnoses to label ids
HAM_diagnosis2id = {
    'nv': 0,  # Melanocytic nevi
    'mel': 1,  # Melanoma
    'bkl': 2,  # Benign keratosis
    'bcc': 3,  # Basal cell carcinoma
    'akiec': -1,  # Actinic keratosis
    'df': -1,  # Dermatofibroma
    'vasc': -1,  # Vascular skin lesions
}

IAD_diagnosis2id = {
    'clark_nevus': 0, 'melanoma_less_than_0.76mm': 1, 'reed/spitz_nevus': 0,
    'melanoma_in_situ': 1, 'melanoma_0.76_to_1.5mm': 1, 'seborrheic_keratosis': 2,
    'basal_cell_carcinoma': 3, 'dermal_nevus': 0, 'vascular_lesion': -1,
    'melanoma_more_than_1.5mm': 1, 'blue_nevus': 0, 'lentigo': 0, 'dermatofibroma': -1,
    'congenital_nevus': -1, 'melanosis': -1, 'combined_nevus': 0, 'miscellaneous': -1,
    'recurrent_nevus': -1, 'melanoma_metastasis': -1, 'melanoma': 1}

# for informational purposes only
id2diagnosis = {
    0: 'Nevus', 1: 'Melanoma', 2: 'Keratosis', 3: 'Basal cell carcinoma', -1: 'ignored'}


def get_IAD():
    """ Return images and labels for each lesion in the IAD dataset. 
    
    Returns:
        images (list): A list (of size num_lesions) with the images as uint8 4-d arrays 
            (num_images_per_lesion x height x widht x 3). This needs to be a list because
            number of images differs per lesion.
        labels (np.array): Array of labels for each patient. 0: Nevus, 1: Melanoma, 
            2: Keratosis, 3: Basal cell carcinoma.
    """
    # Load images
    images = []
    with h5py.File(path.join(IAD_dir, 'images.h5'), 'r') as f:
        for i in range(len(f)):
            im = f[str(i)][()]  # load images
            images.append(im)

    # Load labels
    attrs = pd.read_csv(path.join(IAD_dir, 'lesion_info.csv'))
    diagnosis = attrs['diagnosis']
    labels = np.array([IAD_diagnosis2id[d] for d in diagnosis])

    # Drop labels with -1
    images = [im for im, l in zip(images, labels) if l != -1]
    labels = labels[labels != -1]

    return images, labels


def get_HAM10000():
    """ Return images and labels for each lesion in the HAM10000 dataset. 
    
    Returns:
        images (list): A list (of size num_lesions) with the images as uint8 4-d arrays 
            (num_images_per_lesion x height x widht x 3). This needs to be a list because
            number of images differs per lesion.
        labels (np.array): Array of labels for each patient. 0: Nevus, 1: Melanoma, 
            2: Keratosis, 3: Basal cell carcinoma.
    """
    # Load images
    images = []
    with h5py.File(path.join(HAM_dir, 'images.h5'), 'r') as f:
        for i in range(len(f)):
            im = f[str(i)][()]  # load images
            images.append(im)

    # Load labels
    attrs = pd.read_csv(path.join(HAM_dir, 'lesion_info.csv'))
    diagnosis = attrs['dx']
    labels = np.array([HAM_diagnosis2id[d] for d in diagnosis])

    # Drop labels with -1
    images = [im for im, l in zip(images, labels) if l != -1]
    labels = labels[labels != -1]

    return images, labels


def get_IAD_attributes(attribute_names=[
    'global_pattern', 'dots_globules', 'elevation', 'gray_blue_areas',
    'hypopigmentations', 'pigment_network', 'pigmentation', 'regression_c', 'streaks',
    'vascular_pattern']):
    """ Return the image-derived attributes from the IAD dataset. 
    
    Patients are ordered in the same way as the images (and labels) returned by get_IAD().
    Each attribute is encoded as an integer in the [0, n) range (where n is the number of 
    possible values for that attribue).
    
    With the default attributes, we ignore attributes {'management', 'certainty', 
    'other_criteria', 'location', 'sex', 'age', 'diameter'} because they are not 
    image-derived. We don't use 'note' either because it is text and only defined for ~200
    out of the 700 patients.
    
    Arguments:
        attribute_names (list of strings): Names of the attributes to read from the csv.
        
    Returns:
        attributes (np.array): An int8 array (num_patients x num_attributes) with all of 
            the attributes.
        value_names (list of lists of str): For each attribute, a list with the names of 
            each encoded value as `attribute_name#attribute_value` so we can map back the
            encoded number to the attribute value.
    """
    # Load attributes
    attrs = pd.read_csv(path.join(IAD_dir, 'lesion_info.csv'))
    attributes = attrs[attribute_names]

    # Drop attributes with label = -1 (ignored diagnoses)
    labels = np.array([IAD_diagnosis2id[d] for d in attrs['diagnosis']])
    attributes = attributes[labels != -1]

    # Encode categorical variables as integers
    if not all(attributes.dtypes == 'object'):
        # TODO: Rewrite this to also deal with non-categorical attributes? (if needed)
        # Return a list with the type of each attr. e.g. categorical vs int vs booolean
        raise ValueError('This function only supports categorical values')
    attributes = attributes.astype('category')
    value_names = []
    for attr_name in attribute_names:
        value_names.append([f'{attr_name}#{v}' for v in attributes[attr_name].cat.categories])
        attributes[attr_name] = attributes[attr_name].cat.codes

    # Transform to numpy
    attributes = attributes.to_numpy()

    return attributes, value_names