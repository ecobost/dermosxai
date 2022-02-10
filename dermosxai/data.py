""" Interface to our imaging data. Loads and returns data in a standard format."""
from os import path

import h5py
import numpy as np
import pandas as pd

# Set some important directories
base_dir = '/src/dermosxai/data'
IAD_dir = path.join(base_dir, 'IAD')
HAM_dir = path.join(base_dir, 'HAM10000')
DDSM_dir = path.join(base_dir, 'DDSM')

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


#########################################################################################
# CBIS-DDSM

# Maps 'mass shape' values to a shape id
shape2id = {
    'architectural_distortion': -1,
    'asymmetric_breast_tissue': -1,
    'focal_asymmetric_density': -1,
    'irregular': 1,
    'irregular/architectural_distortion': -1,
    'irregular/asymmetric_breast_tissue': -1,
    'irregular/focal_asymmetric_density': -1,
    'lobulated': 2,
    'lobulated/architectural_distortion': -1,
    'lobulated/irregular': -1,
    'lobulated/lymph_node': -1,
    'lobulated/oval': -1,
    'lymph_node': -1,
    'oval': 0,
    'oval/lobulated': -1,
    'oval/lymph_node': -1,
    'round': 0,
    'round/irregular/architectural_distortion': -1,
    'round/lobulated': -1,
    'round/oval': 0, }

id2shape = {0: 'Oval', 1: 'Irregular', 2: 'Lobulated', -1: 'ignored'}

# Maps 'mass margins' values to an id
margins2id = {
    'circumscribed': 0,
    'circumscribed/ill_defined': -1,
    'circumscribed/microlobulated': -1,
    'circumscribed/microlobulated/ill_defined': -1,
    'circumscribed/obscured': -1,
    'circumscribed/obscured/ill_defined': -1,
    'circumscribed/spiculated': -1,
    'ill_defined': 1,
    'ill_defined/spiculated': -1,
    'microlobulated': 4,
    'microlobulated/ill_defined': -1,
    'microlobulated/ill_defined/spiculated': -1,
    'microlobulated/spiculated': -1,
    'obscured': 3,
    'obscured/circumscribed': -1,
    'obscured/ill_defined': -1,
    'obscured/ill_defined/spiculated': -1,
    'obscured/spiculated': -1,
    'spiculated': 2,
    }

id2margins = {
    0: 'Circumscribed', 1: 'Ill-defined', 2: 'Spiculated', 3: 'Obscured',
    4: 'Microlobulated', -1: 'ignored'}


# Maps pathology to diagnosis
DDSM_diagnosis2id = {
    'benign': False,
    'benign_without_callback': False,
    'malignant': True,
}


def get_DDSM():
    """ Load and return images, labels and attributes for the DDSM dataset.
    
    Returns:
        images (dict): Dictionary with 'train', 'val' and 'test' keys. Each maps to a 
            num_examples x height x width np.float32 array with the images.
        labels (dict): Dictionary with 'train', 'val' and 'test' keys. Each maps to a 
            num_examples boolean array with the labels for each image. 0: Benign, 
            1: Malignant.
        attributes (dict): Dictionary with 'train', 'val' and 'test' keys. Each maps to a 
            num_examples x 2 int8 array with the shape and margin of the mass. See 
            id2shape and id2margins to map these values to their names.
    """
    # Load images
    with h5py.File(path.join(DDSM_dir, 'images.h5'), 'r') as f:
        images = np.array(f['images'], dtype=np.float32)

    # Load csv
    df = pd.read_csv(path.join(DDSM_dir, 'image_info.csv'))

    # Encode attributes (and labels) as integers
    df['mass_shape'] = [shape2id[s] for s in df['mass_shape']]
    df['mass_margins'] = [margins2id[m] for m in df['mass_margins']]
    df['pathology'] = [DDSM_diagnosis2id[p] for p in df['pathology']]

    # Restrict data to examples with valid mass_shape and mass_margins
    images = images[(df['mass_shape'] != -1) & (df['mass_margins'] != -1)]
    df = df[(df['mass_shape'] != -1) & (df['mass_margins'] != -1)]

    # Split datasets
    splits = ['train', 'val', 'test']
    images = {s: images[df['split'] == s] for s in splits}
    labels = {s: np.array(df['pathology'][df['split'] == s], dtype=bool) for s in splits}
    attributes = {
        s: np.array(df[['mass_shape', 'mass_margins']][df['split'] == s], dtype=np.int8)
        for s in splits}

    return images, labels, attributes

# here just for reference
# DDSM_value_names = [[f'Shape: {id2shape[i]}' for i in range(3)],
#                     [f'Margins: {id2margins[i]}' for i in range(5)]]
