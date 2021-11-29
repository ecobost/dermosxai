""" Some basic utility functions. """
import numpy as np
import time
from sklearn import metrics


def crop_black_edges(im, threshold=10):
    """ Crops black edges around an image by transforming it into grayscale (if needed), 
    thresholding it and then dropping any rows/cols with no value above the threshold.

    Arguments:
        im (np.array): Image to be processed (2-d or 3-d).
        threshold (float): Threshold to use when binarizing the image.

    Returns:
        An image (np.array) with the edges cropped out.
    """
    # Make grayscale
    if im.ndim == 2:
        grayscale = im
    if im.ndim == 3:
        grayscale = im.mean(-1)
    else:
        raise ValueError('Only works for 2-d or 3-d arrays.')

    # Threshold
    binary = grayscale > threshold

    # Find bounding box
    nonzero_cols = np.nonzero(binary.sum(axis=0))[0]
    first_col, last_col = nonzero_cols[0], nonzero_cols[-1] + 1
    nonzero_rows = np.nonzero(binary.sum(axis=-1))[0]
    first_row, last_row = nonzero_rows[0], nonzero_rows[-1] + 1

    # Crop image
    cropped_im = im[first_row:last_row, first_col:last_col]

    return cropped_im


def crop_to_ratio(im, desired_ratio=4 / 3):
    """ Crop (either) the rows or columns of an image to match (as best as possible) the 
    desired ratio.
    
    Arguments:
        im (np.array): Image to be processed.
        desired_ratio (float): The desired ratio of the output image expressed as 
            width/height so 3:2 (= 3/2) or 16:9 (  = 16/9).
        
    Returns:
        An image (np.array) with the desired ratio.
    """
    height = im.shape[0]
    width = im.shape[1]
    if width / height < desired_ratio:  # Crop rows
        desired_height = int(round(width / desired_ratio))
        to_crop = height - desired_height
        top_crop = to_crop // 2
        bottom_crop = to_crop - top_crop
        cropped_image = im[top_crop:height - bottom_crop, :]
    else:  # Crop columns
        desired_width = int(round(height * desired_ratio))
        to_crop = width - desired_width
        left_crop = to_crop // 2
        right_crop = to_crop - left_crop
        cropped_image = im[:, left_crop:width - right_crop]

    return cropped_image


def tprint(*messages):
    """ Prints a message (with a timestamp next to it).
    
    Arguments:
        message (string): Arguments to be print()'d.
    """
    formatted_time = '[{}]'.format(time.ctime())
    print(formatted_time, *messages, flush=True)


def create_grid(ims, num_rows=3, num_cols=4, row_gap=5, col_gap=5, bg_value=1):
    """ Creates a grid of images from individual images.
    
    Arguments:
        ims (sequence of arrays): Images. Should all have the same size.
        num_rows, num_cols (int): Size of the grid.
        row_gap, col_gap (int): The space in pixels between each image.
        bg_value (float): The color of the background.
    
    Returns:
        grid (np.array): A big grid image with the desired rows and columns.
    """
    # Create empty array
    height, width, num_channels = ims[0].shape
    grid = np.full((num_rows * (height + row_gap) - row_gap, num_cols *
                    (width + col_gap) - col_gap, num_channels), fill_value=bg_value,
                   dtype=ims[0].dtype)

    # Fill the grid
    for i, im in enumerate(ims[:num_rows * num_cols]):
        row = i // num_cols
        col = i % num_cols

        grid[row * (height + row_gap):row * (height + row_gap) + height,
             col * (width + col_gap):col * (width + col_gap) + width] = im

    return grid


def to_npy(torch_im, img_mean=0, img_std=1):
    """ Transform a (batch of) torch.Float tensor image(s) into a numpy array.
    
    Arguments:
        torch_im (torch.FloatTensor): A 3-d or 4-d tensor with the image(s) in the 
            (N x ... x) C x H x W format.
        mean, std (int or np.array): Mean and standard deviation (broadcastable to 
            H x W x C) that will be used to unnormalize the image.
    
    Returns:
        A (n x h x w x c) np.float array with the (possibly unnormalized) images in the 
            [0, 1] range.
    """
    if torch_im.ndim < 3:
        raise ValueError('Only works for images with at least one channel (>= 3-d).')

    im = np.moveaxis(torch_im.detach().cpu().numpy(), -3, -1)
    im = (im * img_std + img_mean)
    return im


def binarize_categorical(categorical, num_categories=None):
    """ One-hot encode a categorical variable with values [0, n) as n binary variables.
    
    Arguments:
        categorical (np.array): A 1-d int array with the categorical variables (0-n).
        num_categories (int): Number of categories expected in the categorical variable.
            If not provided, it is deduced from the categorical variable.
    
    Returns
        binary (np.array): A 2-d boolean array (num_variables x n) with the encoded 
            variables.
    """
    if num_categories is None:
        num_categories = categorical.max() + 1
    return np.eye(num_categories, dtype=bool)[categorical]


def compute_metrics(probs, targets):
    """ Computes a set of classification metrics.
    
    In the multi-class case, F1, AUC and AP are computed per class and averaged; for the 
    binary case, the F1, AUC and AP of the positive class is returned.
    
    Arguments:
        probs (np.array): A num_samples x num_classes array with the predicted 
            probabilities per class. Should sum up to 1 per example.
        targets (np.array): An num_samples array (int or bool) with the correct target 
            classes (in [0, n) range).
            
    Returns
        accuracy (float): Classification accuracy.
        kappa (float): Cohen's kappa score.
        mcc (float): Matthews' correlation coefficient.
        f1 (array): F1 score.
        auc (array): Area under the ROC curve.
        ap (array): Average precision per class/Area under the PR curve.
    
    Note: 
        Cohen's kappa measures how much better (or worse) the classifier does than one 
        that predicts classes at random but with the same proportions. And normalizes this 
        value by the maximum possible difference: (acc - acc_random) / (1-acc_random) to 
        keep numbers in [-1, 1] range.
                
        Matthew's correlation coefficient (for binary cases) essentially computes the 
        correlation between the vector of predictions and the ground-truth vector.
     
        According to [1], MCC and Kappa return the same number if the confusion matrix is 
        symmetric but kappa gives counter-intuitive results when the distribution of the 
        off-diagonals has high entropy. According to [2], MCC is more informative and 
        truthful than accuracy and F1 and kappa has too many problems to even be 
        considered.
        
        [2] also says average precision (the area under the precision-recall curve) should 
        be preferred to AUC for imbalanced datasets.
        
        Conclusion: Use MCC or average precision. Or accuracy if not afraid of unbalance.
        
        [1] Delgado R, Tibau X-A (2019) Why Cohen’s Kappa should be avoided as performance
        measure in classification. PLoS ONE 14(9): e0222916.
        
        [2] Chicco, D., Jurman, G. The advantages of the Matthews correlation coefficient 
        (MCC) over F1 score and accuracy in binary classification evaluation. BMC Genomics 
        21, 6 (2020).
    """
    # Sanity check
    if probs.ndim != 2 or probs.shape[-1] < 2:
        # in case they send only the probs for the positive class (a num_examples vector)
        raise ValueError('Expects probabilities to be num_examples x num_classes matrix.')

    # Compute metrics
    pred_labels = np.argmax(probs, -1)
    accuracy = metrics.accuracy_score(targets, pred_labels)
    kappa = metrics.cohen_kappa_score(targets, pred_labels)
    mcc = metrics.matthews_corrcoef(targets, pred_labels)

    # Compute metrics that are averaged across all classes
    num_classes = probs.shape[-1]
    binarized_targets = binarize_categorical(targets.astype(int),
                                             num_categories=num_classes)
    f1 = metrics.f1_score(targets, pred_labels,
                          average='binary' if num_classes == 2 else 'macro')
    if all(binarized_targets.sum(0) > 0):  # all classes have at least one example
        if num_classes == 2:
            auc = metrics.roc_auc_score(targets, probs[:, 1])
            ap = metrics.average_precision_score(targets, probs[:, 1])
        else:
            auc = metrics.roc_auc_score(targets, probs, multi_class='ovr')
            ap = metrics.average_precision_score(binarized_targets, probs)  #*
            #* unsure about this one, sklearn says it is multi-label not multiclass (?)
    else:
        # equivalent to all examples having the same label for the binary case.
        ap, auc = float('nan'), float('nan')  # AUC and PRAUC are undefined

    return accuracy, kappa, mcc, f1, auc, ap


def off_diagonal(m):
    """Returns off-diagonal elements of a square matrix. """
    n = m.shape[0]
    return m.flatten()[1:].view(n - 1, n + 1)[:, :-1].flatten()  #.reshape(n, n-1)


def split_data(num_examples, split=[0.8, 0.1]):
    """ Create the slices to split num_examples into train/val/test.
    
    Arguments:
        num_examples: Total number of examples
        split: A tuple. Proportion of examples in training and validation (Test is the 
            rest of examples).
            
    Returns:
        train_slice, val_slice, test_slice: Slice objects that can be used to indent the 
            data arrays.
    """
    if sum(split) > 1:
        raise ValueError('Splits have to be a proportion in [0, 1] and sum <= 1')

    # Create splits
    train_slice = slice(int(round(split[0] * num_examples)))  # first 80%
    val_slice = slice(train_slice.stop, int(round(sum(split) * num_examples)))  # 80-90%
    test_slice = slice(val_slice.stop, None)  # 90%-100%

    return train_slice, val_slice, test_slice