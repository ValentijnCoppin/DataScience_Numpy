import numpy as np
from PIL import Image


def read_image_for_processing(directory:str) -> np.array:
    """
    Returns the numpy array of an RGB formatted image

    Parameters:
    - directory: The directory + name of the image file.

    Returns:
    - 3 dimensional NumPy array formatted to RGB.
    """
    image = Image.open(directory)
    if image.mode == 'RGB':
        image = Image.fromarray(np.array(image), 'RGB')
    np_image = np.array(image)
    
    return np_image


def tile_array(array:np.array, n_horizontal: int, n_vertical: int) -> np.array:
    """
    Replicates (tiles) a 3 dimensional numpy array horizontally vertically

    Parameters:
    - array: NumPy array to be replicated.
    - n_horizontal: Number of horizontal tiles.
    - n_vertical: Number of vertical tiles.

    Returns:
    - tiled NumPy array.
    """
    
    tiled_image = np.tile(array, (n_vertical, n_horizontal, 1))
    return tiled_image


def flip_image(image, axis):
    if axis == 'y':
        flipped_image = image[::-1, :, :]
    elif axis == 'x':
        flipped_image = image[:, ::-1, :]
    elif axis == 'both':
        flipped_image = image[::-1, ::-1, :]
    else:
        raise ValueError('Invalid axis: {}'.format(axis))
    return flipped_image


def create_filpped_images(image):
    unflipped = image.copy()
    flipped_x = flip_image(image.copy(), 'x')
    flipped_y = flip_image(image.copy(), 'y')
    flipped_both = flip_image(image.copy(), 'both')
    return [unflipped, flipped_x, flipped_y, flipped_both]


def grid_with_flips(image, flip_matrix):
    list_of_concatenated_images = []
    image_list = create_filpped_images(image)
    rows, cols = len(flip_matrix), len(flip_matrix[0])
    image_matrix = [[None] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            # Replace each value with the corresponding value from the list
            flip_index = flip_matrix[i][j]
            image_matrix[i][j] = image_list[flip_index]
        list_of_concatenated_images.append(np.concatenate(image_matrix[i], axis=1))

    return np.concatenate(list_of_concatenated_images, axis=0)
    
