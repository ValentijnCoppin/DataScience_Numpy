import numpy as np
from PIL import Image
import random

def create_error_image() -> np.array:
    """
    creates a numpy array for a black image with a red cross
    Returns:
        np.ndarray: A 3-dimensional NumPy array formatted to RGB.
    """
    arr = np.zeros((20, 20, 3)).astype(np.uint8)
    for i in range(arr.shape[0]):
        arr[i, i, 0] = 255
        arr[-i, i - 1, 0] = 255
    return arr


def image_read_from_file(directory:str) -> np.array:
    """
    Returns the numpy array of an RGB formatted image

    Parameters:
        directory (str): The directory and name of the image file.

    Returns:
        np.ndarray: Numpy array representing an image.
    """

    image = Image.open(directory)
    # if image.mode == 'RGB':
    #     image = Image.fromarray(np.array(image), 'RGB')
    image = Image.fromarray(np.array(image), image.mode)
    np_image = np.array(image)
    
    return np_image


def image_flip(image: np.array, axis:str) -> np.array:
    """
    Returns an image flipped over a defined axis.

    Parameters:
        image (np.ndarray): Numpy array representing an image.
        axis (str): Specifies the axis to flip the image over ('x', 'y', or 'both').

    Returns:
        np.ndarray: Numpy array representing the flipped image.
    """

    flipped_image = image.copy()
    axis = axis.lower()
    if axis == 'y':
        flipped_image = flipped_image[::-1, :, :]
    elif axis == 'x':
        flipped_image = flipped_image[:, ::-1, :]
    elif axis == 'both' or axis == 'b' or axis == 'xy':
        flipped_image = flipped_image[::-1, ::-1, :]
    else:
        print("Please give a correct axis argument ['x', 'y', 'both']")
        return error_image
    return flipped_image


def image_change_hue(image:np.array, color:str, saturating:bool = False) -> np.array:
    """
    Returns an image with updated hue.

    The hue is altered by setting RGB layers to 0.

    Parameters:
        image (np.ndarray): Numpy array representing an image.
        color (str): Name of the color to alter.
        saturating (bool, optional): If True, changes hue by saturating RGB layers. 
            If False, changes hue by zeroing RGB layers. Defaults to False.

    Returns:
        np.ndarray: Numpy array representing the altered image.
    """

    if saturating:
        color_dict = {"cyan":(1, 2), "magenta":(0, 2), "yellow":(0, 1), "red":0, "green":1, "blue":2}
        fill_values = 255
    else:
        color_dict = {"red":(1, 2), "green":(0, 2), "blue":(0, 1), "cyan":0, "magenta":1, "yellow":2}
        fill_values = 0
    
    try:
        color_lower = color.lower()
        if color_lower not in color_dict.keys():
            color_lower = update_color_key_name(color_lower)
    except KeyError as e:
        print(f"{e}")
        return error_image
    try:
        resulting_image = image.copy()

        layers = color_dict[color_lower]
        resulting_image[:,:,layers] = np.full_like(image[:,:,layers], fill_values)

        return resulting_image
    except Exception as e:
        print(f"{e}")
        return error_image


def update_color_key_name(color_key:str) -> str:
    """
    Updates the color key name to the full color name.

    Parameters:
        color_key (str): Short form of the color key.

    Returns:
        str: Full name of the color key.
    """

    color_mapping = {
        'r': 'red',
        'g': 'green',
        'b': 'blue',
        'c': 'cyan',
        'm': 'magenta',
        'y': 'yellow'
    }
    for key, value in color_mapping.items():
        if color_key.startswith(key):
            print(f"Color given = {color_key}, assuming you meant {value}")
            return value
    raise KeyError(f"The color index {color_key} is unknown")


def create_list_of_filpped_images(image:np.array) -> list:
    """
    Returns a list of images flipped over each possible axis.

    The order of flipping: [no flip, flipped over x-axis, flipped over y-axis, flipped over both x and y axis].

    Parameters:
        image (np.ndarray): Numpy array representing an image.

    Returns:
        list: List of Numpy arrays representing flipped images.
    """

    unflipped = image.copy()
    flipped_x = image_flip(unflipped, 'x')
    flipped_y = image_flip(unflipped, 'y')
    flipped_both = image_flip(unflipped, 'both')
    return [unflipped, flipped_x, flipped_y, flipped_both]


def create_dict_of_colored_images(image:np.array, saturating:bool = False) -> dict:
    """
    Returns a dictionary of images colored in red (r), green (g), blue (b), cyan (c), magenta (m), yellow (y).

    Parameters:
        image (np.ndarray): Numpy array representing an image.
        saturating (bool, optional): If True, changes hue by saturating RGB layers. 
            If False, changes hue by zeroing RGB layers. Defaults to False.

    Returns:
        dict: Dictionary of Numpy arrays representing colored images.
    """

    original = image.copy()
    red_colored = image_change_hue(original, 'red', saturating)
    green_colored = image_change_hue(original, 'green', saturating)
    blue_colored = image_change_hue(original, 'blue', saturating)
    cyan_colored = image_change_hue(original, 'cyan', saturating)
    yellow_colored = image_change_hue(original, 'yellow', saturating)
    magenta_colored = image_change_hue(original, 'magenta', saturating)
    return {"o": original, "r": red_colored, "g": green_colored, "b": blue_colored, "c": cyan_colored, "y": yellow_colored, "m": magenta_colored}

   
def create_all_possible_alterations(image:np.array) -> list[dict]:
    """
    Returns a list of images flipped over each possible axis.

    The order of flipping: [no flip, flipped over x-axis, flipped over y-axis, flipped over both x and y axis].

    Parameters:
        image (np.ndarray): Numpy array representing an image.

    Returns:
        list: List of dictionaries of Numpy arrays representing altered images.
    """
   
    list_of_flipped_images = create_list_of_filpped_images(image.copy())

    unflipped_dict = create_dict_of_colored_images(list_of_flipped_images[0])
    flipped_x_dict = create_dict_of_colored_images(list_of_flipped_images[1])
    flipped_y_dict = create_dict_of_colored_images(list_of_flipped_images[2])
    flipped_both_dict = create_dict_of_colored_images(list_of_flipped_images[3])
    return [unflipped_dict, flipped_x_dict, flipped_y_dict, flipped_both_dict]


def image_matrix_with_alterations(image: np.array, *, flip_matrix: np.array=None, color_matrix: np.array=None):
    """
    Returns a composite image with various alterations based on flip and color matrices.

    Parameters:
        image (np.ndarray): Numpy array representing an image.
        flip_matrix (np.ndarray, optional): Matrix indicating flipping directions.
        color_matrix (np.ndarray, optional): Matrix indicating colors for each segment.

    Returns:
        np.ndarray: Numpy array representing the composite image.
    """
    
    try:
        flip_matrix, color_matrix = create_flip_or_color_matrix(flip_matrix, color_matrix)
    except MemoryError as e:
        print(f"{e}")
        return error_image

    list_of_concatenated_images = []
    list_of_possible_alterations = create_all_possible_alterations(image)
    rows, cols = len(flip_matrix), len(flip_matrix[0])
    image_matrix = [[None] * cols for _ in range(rows)]
    try:
        for i in range(rows):
            for j in range(cols):
                # Replace each value with the corresponding value from the list
                flip_index = flip_matrix[i][j]
                color_index = color_matrix[i][j]
                image_matrix[i][j] = list_of_possible_alterations[flip_index][color_index]
            list_of_concatenated_images.append(np.concatenate(image_matrix[i], axis=1))
    except KeyError:
        print(f"key {color_matrix[i][j]} is not found as a valid key")
        return error_image
    except Exception as e:
        print(f"{e}")
        return error_image

    return np.concatenate(list_of_concatenated_images, axis=0)

 
def enlarge_image(image: np.array, factor: int) -> np.array:
    """
    Enlarges an image by a given factor.

    Parameters:
        image (np.ndarray): Numpy array representing an image.
        factor (int): Integer factor for enlarging.

    Returns:
        np.ndarray: Numpy array representing the enlarged image.
    """

    factor = int(factor)
    return np.kron(image, np.ones((factor, factor, 1), dtype=image.dtype))


def create_color_matrix_from_border_list(color_list: list) -> np.array:
    """
    Creates a square Numpy array with the border based on the color list.

    Parameters:
        color_list (list): List of colors.

    Returns:
        np.ndarray: Numpy array representing the border based on the color list.
    """

    if len(color_list) % 4 != 0:
        print("The length of the color list should be divisible by 4")
        return error_image
    side_length = (len(color_list) // 4) + 1
    grid = np.tile('o', (side_length, side_length))

    top_border = color_list[:side_length]
    right_border = color_list[side_length:2*side_length - 2]
    bottom_border = color_list[2*side_length - 2: 3*side_length - 2][::-1]
    left_border = color_list[3*side_length - 2: 4*side_length - 4][::-1]
    
    grid[0, :] = top_border
    grid[1:-1, -1] = right_border
    grid[-1, :] = bottom_border
    grid[1:-1, 0] = left_border
    return grid


def create_colorful_big_one(image: np.array, scramble_center: bool=False, *, flip_matrix: np.array=None, color_matrix: np.array=None) -> np.array:
    """
    Creates a composite image with various alterations, including a colored border and a central image.

    Parameters:
        image (np.ndarray): Numpy array representing an image.
        scramble_center (bool, optional): If True, the central image will be scrambled. Defaults to False.
        flip_matrix (np.ndarray, optional): Matrix indicating flipping directions.
        color_matrix (np.ndarray, optional): Matrix indicating colors for each segment.

    Returns:
        np.ndarray: Numpy array representing the composite image.
    """
    
    try:
        flip_matrix, color_matrix = create_flip_or_color_matrix(flip_matrix, color_matrix)
    except MemoryError as e:
        print(f"{e}")
        return error_image

    center_size = flip_matrix.shape[0] - 2

    picture_grid = image_matrix_with_alterations(image, flip_matrix=flip_matrix, color_matrix=color_matrix)
    center_image = enlarge_image(image, center_size)
    if scramble_center:
        random_list = random.sample(range(center_size**2), center_size**2)
        segmented_list = segment_image(center_image, center_size**2)
        center_image = reshape_image(segmented_list, random_list)

    start_y = image.shape[0]
    end_y = start_y + (image.shape[0] * int(center_size))
    start_x = image.shape[1]
    end_x = start_x + (image.shape[1] * int(center_size))
    if center_image.shape == picture_grid[start_y:end_y, start_x:end_x, :].shape:
        picture_grid[start_y:end_y, start_x:end_x, :] = center_image
    else:
        print("Error: Dimensions mismatch")
        return error_image
    return picture_grid


def create_flip_or_color_matrix(flip_matrix: np.array = None, color_matrix: np.array = None) -> tuple[np.array, np.array]:
    """
    Creates a flip or color matrix if it is missing and the other one exists.

    Parameters:
        flip_matrix (np.ndarray, optional): Matrix indicating flipping directions.
        color_matrix (np.ndarray, optional): Matrix indicating colors for each segment.

    Returns:
        tuple: Tuple of Numpy arrays representing flip and color matrices.
    """

    if flip_matrix is None:
        flip_matrix = np.full_like(color_matrix, 0, dtype='int')
    elif color_matrix is None:
        color_matrix = np.full_like(flip_matrix, 'o', dtype='U1')

    flip_matrix = np.array(flip_matrix)
    color_matrix = np.array(color_matrix)

    if flip_matrix.shape != color_matrix.shape:
        raise MemoryError("flip_matrix and color_matrix are not of the same shape")
    
    return flip_matrix, color_matrix


def segment_image(image:np.array, number_of_segments: int) -> list:
    """
    Segments an image into a specified number of segments.

    Parameters:
        image (np.ndarray): Numpy array representing an image.
        number_of_segments (int): Number of segments to divide the image into.

    Returns:
        list: List of Numpy arrays representing image segments.
    """

    side_length = int(number_of_segments**0.5)
    segments = []   
    height, width, layer = image.shape

    for i in range(side_length):
        for j in range(side_length):
            segment = image[j * height//side_length: (j + 1) * height//side_length, 
                            i * width//side_length: (i + 1) * width//side_length,
                              :]
            segments.append(segment)
    return segments


def reshape_image(image_list: list, new_arrangment: list) -> np.array:
    """
    Reshapes a list of image segments based on a new arrangement.

    Parameters:
        image_list (list): List of Numpy arrays representing image segments.
        new_arrangement (list): New arrangement of segments.

    Returns:
        np.ndarray: Numpy array representing the reshaped image.
    """

    image_rows = []
    number_of_segments = len(image_list)
    side_length = int(number_of_segments**0.5)
    rearranged_list = [image_list[i] for i in new_arrangment]
    for i in range(side_length):
        image_row = rearranged_list[i * side_length : (i+1) * side_length]
        image_row = np.concatenate(image_row, axis=0)
        image_rows.append(image_row)
    return np.concatenate(image_rows, axis=1)


error_image = create_error_image()