"""
Module Name: image_builders
Description: This module contains a collection of functions to create various
big images that contain multiple smaller (manipulated) images.
Author: Raf Mesotten
"""

# Importing necessary packages
import numpy as np
import sys
sys.path.append('../scripts')  # Adjust the path accordingly
import image_manipulations as img


def tile(image, rows, columns):
    """
    Tile the input image horizontally (rows) and vertically (columns).
    """
    tiled_image = np.tile(image, (rows, columns, 1))
    return tiled_image


def grid_with_flips(image, matrix):
    """
    Create a grid of images (based on the input image) with specified
    manipulations (based on the input matrix). Each element in the matrix can
    be a code (str or int) representing a sequence of manipulations that are
    available in the 'multiple_manipulations' method. See the docstring of
    'multiple_manipulations' for details on the code format.
    """
    result_images_rows = []

    for row in matrix:
        row_images = [img.multiple_manipulations(image, element) for element in row]
        row_concatenated_images = np.concatenate(row_images, axis=1)
        result_images_rows.append(row_concatenated_images)

    result_image = np.concatenate(result_images_rows, axis=0)

    return result_image


def create_colorful_big_one(image, matrix):
    """
    Create a colorful image based on a given image and a list of color codes.

    The function takes an image and a matrix of color codes ('r' for red, 'g'
    for green, 'b' for blue) as input. The color codes are mapped clockwise on
    the edges of a square matrix, and the resulting matrix is used to generate
    a composite image with colored segments. In case the input matrix doesn't
    contain a multiple of 4 elements (required to create a square output
    image), extra (unmodified) images are added. The input matrix should
    contain at least 5 elements. The big image will be in the center, and its
    size automatically adjusts.

    Examples:
        matrix_1 = ['r', 'g', 'b', 'r', 'g', 'b', 'r', 'g', 'b', 'r', 'g', 'b']
        matrix_2 = ['r', 'g', 'b', 'r', 'g', 'b', 'r', 'g', 'b']
    """
    if len(matrix) % 4 != 0:
        # Calculate the number of 'o' elements needed to make len(matrix) % 4 == 0
        num_o_elements = 4 - len(matrix) % 4
        # Append 'o' elements to the matrix ('o' stands for 'original image')
        matrix.extend(['o'] * num_o_elements)

    size = 1 + round(len(matrix)/4)

    # Define a [size x size] matrix
    manipulation_matrix = np.zeros((size, size), dtype='str')

    # Map the colors clockwise on the edges of the matrix
    manipulation_matrix[0, :] = matrix[:size]  # Top row
    manipulation_matrix[:, -1] = matrix[size - 1:2 * (size - 1) + 1]  # Right column
    manipulation_matrix[-1, ::-1] = matrix[2 * (size - 1):3 * (size - 1) + 1]  # Bottom row (reversed)
    manipulation_matrix[::-1, 0][:-1] = matrix[3 * (size - 1):]  # Left column (reversed), except the last element

    # Creating 1 big image out of the list with colored images
    result_images_rows = []  # Empty list to store rows of images

    for row in manipulation_matrix:
        row_images = [img.color(image, code) for code in row]
        row_concatenated_images = np.concatenate(row_images, axis=1)
        result_images_rows.append(row_concatenated_images)

    result_image = np.concatenate(result_images_rows, axis=0)

    # Putting the big one on top of it
    d = np.shape(image)
    result_image[d[0]:(size-1) * d[0], d[0]:(size-1) * d[0], :] = img.enlarge(image, size-2)

    # Returning the result
    return result_image
