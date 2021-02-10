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
    'akiem': -1,  # Actinic keratosis
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
    """ Return images and labels for each patient in the IAD dataset. 
    
    Returns:
        images (list): A list (of size num_patients) with the images as 4-d arrays 
            (num_images_per_patient x height x widht x 3). This needs to be a list because
            number of images differs per patient.
        labels (np.array): Array of labels for each patient. 0: Nevus, 1: Melanoma, 
            2: Keratosis, 3: Basal cell carcinoma.
    """
    # Load images
    images = []
    with h5py.File(path.join(IAD_dir, 'IAD_images.h5'), 'r') as f:
        for i in range(len(f)):
            im = f[str(i)][()]  # load images
            images.append(im)

    # Load labels
    attrs = pd.read_csv(path.join(IAD_dir, 'IAD_metadata.csv'))
    diagnosis = attrs['diagnosis']
    labels = np.array([IAD_diagnosis2id[d] for d in diagnosis])

    # Drop labels with -1
    images = images[labels != -1]
    labels = labels[labels != -1]

    return images, labels


def get_HAM10000():
    """ Return images and labels for each patient in the HAM10000 dataset. 
    
    Returns:
        images (np.array): A np.uint8 array (num_patients x height x widht x 3) with the 
            images.
        labels (np.array): Array of labels for each patient. 0: Nevus, 1: Melanoma, 
            2: Keratosis, 3: Basal cell carcinoma.
    """
    #TODO:
    pass


def get_IAD_attributes(attribute_names=[
    'global_pattern', 'diameter', 'dots_globules', 'elevation', 'gray_blue_areas',
    'hypopigmentations', 'pigment_network', 'pigmentation', 'regression_c', 'streaks',
    'vascular_pattern'], make_dummy=[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], ref_values=[
        'unspecific', 'absent', 'flat', 'absent', 'absent', 'absent', 'absent', 'absent',
        'absent', 'absent']):
    """ Return the image-derived attributes from the IAD dataset. Same order as the images 
    (and labels) returned by get_IAD()
    
    Arguments:
        attribute_names (list of strings): Names of the attributes to read from the csv.
        make_dummy (sequence of bools): Whether to represent each of the returned 
            attributes as a set of dummy variables (used for categorical variables).
        ref_values (list of strings): Which attribute value to use as the reference value 
            when dummy coding the categorical variables (see notes for details)
        
    Returns:
        attributes (np.array): An array (num_patients x num_attributes) with all of the 
            attributes.
        attribute_names (list of str): Names of the returned attributes as 
            `attribute_name#attribute_value` so we can map back to each value.
        is_binary (boolean array): Whether the attribute is binary.
        
    Note:
        We represent categorical variables using a one-hot encoding method where a 
        variable with k values is represented as k-1 binary variables representing the 
        presence or absence of each of k-1 variables and a reference level of all zeros 
        represents the remaining attribute.
        
        Some attributes like pigment_network have values [absent, typical, atypical] and 
        could have been coded as ordinal rather than nominal categorical variables. I 
        chose not to because it is not fully clear that atypical > typical (and if so, by 
        ho much) so I could encode them as continuous variables or that atypical is a 
        superset of typical so they could be encoded as 00 (absent), 10 (typical) and 
        11 (atypical).
        
        With the default attributes, we ignore attributes {'management', 'certainty', 
        'other_criteria', 'location', 'sex', 'age'} because they are not image-derived. 
        We don't use 'note' either because it is text and only defined for ~200 out of the 
        700 patients.
    """
    # Basic checks
    if len(attribute_names) != len(make_dummy):
        raise ValueError('attributes and make_dummy should have the same length.')
    if len(ref_values) != sum(make_dummy):
        raise ValueError('Need to provide a ref_value for all categorical variables.')

    # Load attributes
    attrs = pd.read_csv(path.join(IAD_dir, 'IAD_metadata.csv'))
    attributes = attrs[attribute_names]

    # Code attributes as dummy variables
    nominal_attributes = [a for a, d in zip(attribute_names, make_dummy) if d]
    for column_name, ref_value in zip(nominal_attributes, ref_values):
        attributes = pd.get_dummies(attributes, columns=[column_name], prefix_sep='#')
        attributes = attributes.drop(f'{column_name}#{ref_value}', axis='columns')

    # Transform to numpy
    attribute_names = attributes.columns.to_list()
    is_binary = np.array([('#' in n) for n in attribute_names])
    attributes = attributes.to_numpy()

    # Drop aatributes with label = -1 (ignored diagnoses)
    labels = np.array([IAD_diagnosis2id[d] for d in attrs['diagnosis']])
    attributes = attributes[labels != -1]

    return attributes, attribute_names, is_binary