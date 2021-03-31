""" Some basic utility functions. """
import numpy as np
import time



def crop_black_edges(im, threshold=8):
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
    cropped_im = im[first_row: last_row, first_col:last_col]

    return cropped_im

def crop_to_ratio(im, desired_ratio=4/3):
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


def tprint(*messages):
    """ Prints a message (with a timestamp next to it).
    
    Arguments:
        message (string): Arguments to be print()'d.
    """
    formatted_time = '[{}]'.format(time.ctime())
    print(formatted_time, *messages, flush=True)