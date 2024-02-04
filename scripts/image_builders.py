"""
Module Name: image_builders
Description: This module contains a collection of functions to create various
big images that contain multiple smaller (manipulated) images.
Author: Raf Mesotten
"""

# Importing necessary packages
import numpy as np
import scripts.image_manipulations as img


def tile(image, rows, columns):
    """
    Tile the input image horizontally (rows) and vertically (columns).
    """
    tiled_image = np.tile(image, (rows, columns, 1))
    return tiled_image


def grid_with_flips(image, matrix):
    """
    Create a grid of images with specified flips based on the input matrix. The
    matrix should be a 2D matrix specifying flip operations for each element.
    Each element in the matrix can be 1 (horizontal flip), 2 (vertical flip),
    or 3 (both flips). In all other cases the original image will be returned.
    """
    # Creating a list with flipped images
    images_list = []

    for row in matrix:
        row_images = []
        for element in row:
            if element == 1:
                row_images.append(img.h_flip(image))
            elif element == 2:
                row_images.append(img.v_flip(image))
            elif element == 3:
                row_images.append(img.h_flip(img.v_flip(image)))
            else:
                row_images.append(image)
        images_list.append(row_images)

    # Creating 1 big image out of the list with flipped images
    result_images_rows = []  # Empty list to store rows of images

    for row_images in images_list:
        row_concatenated_images = np.concatenate(row_images, axis=1)
        result_images_rows.append(row_concatenated_images)

    result_image = np.concatenate(result_images_rows, axis=0)

    # Returning the result
    return result_image



def create_colorful_big_one(image, matrix):
    """
    Create a colorful composite image based on a given image and a list of
    color codes.

    The function takes an image and a matrix of color codes ('r' for red, 'g'
    for green, 'b' for blue) as input. The color codes are mapped clockwise on
    the edges of a square matrix, and the resulting matrix is used to generate
    a composite image with colored segments. In case the input matrix doesn't
    contain a multiple of 4 elements (required to create a square output
    image), extra (unmodified) images are added.
    """
    if len(matrix) % 4 != 0:
        # Calculate the number of 'o' elements needed to make len(matrix) % 4 == 0
        num_o_elements = 4 - len(matrix) % 4
        # Append 'o' elements to the matrix ('o' stands for 'original image')
        matrix.extend(['o'] * num_o_elements)

    size = 1 + round(len(matrix)/4)

    # Define a [size x size] matrix
    colored_matrix = np.zeros((size, size), dtype='str')

    # Map the colors clockwise on the edges of the matrix
    colored_matrix[0, :] = matrix[:size]  # Top row
    colored_matrix[:, -1] = matrix[size-1:2*(size-1)+1]  # Right column
    colored_matrix[-1, ::-1] = matrix[2*(size-1):3*(size-1)+1]  # Bottom row (reversed)
    colored_matrix[::-1, 0][:-1] = matrix[3*(size-1):]  # Left column (reversed), except the last element

    # Creating a list with colored images
    images_list = []

    for row in colored_matrix:
        row_images = []
        for element in row:
            if element == "r":
                row_images.append(img.color(image, "r"))
            elif element == "g":
                row_images.append(img.color(image, "g"))
            elif element == "b":
                row_images.append(img.color(image, "b"))
            else:
                row_images.append(image)
        images_list.append(row_images)

    # Creating 1 big image out of the list with colored images
    result_images_rows = []  # Empty list to store rows of images

    for row_images in images_list:
        row_concatenated_images = np.concatenate(row_images, axis=1)
        result_images_rows.append(row_concatenated_images)

    result_image = np.concatenate(result_images_rows, axis=0)

    # Putting the big one on top of it
    d = np.shape(image)
    result_image[d[0]:(size-1) * d[0], d[0]:(size-1) * d[0], :] = img.enlarge(image, size-2)

    # Returning the result
    return result_image
