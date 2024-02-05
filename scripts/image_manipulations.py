"""
Module Name: image_manipulations
Description: This module contains a collection of manipulations of a single image.
Author: Raf Mesotten
"""


# Importing necessary packages
import numpy as np


def h_flip(image):
    """
    Flip the input image horizontally, reversing the order of pixels within
    each row.
    """
    h_flipped_image = np.flip(image, axis=1)
    return h_flipped_image


def v_flip(image):
    """
    Flip the input image vertically, reversing the order of pixels within
    each column.
    """
    v_flipped_image = np.flip(image, axis=0)
    return v_flipped_image


def color(image, color):
    """
    Color the input image red ('r'), green ('g') or blue ('b'). Return the
    original image in case of an unknown color.
    """
    if color == 'r':  # red
        red_image = image.copy()
        red_image[:, :, [1, 2]] = 0
        return red_image

    elif color == 'g':  # green
        green_image = image.copy()
        green_image[:, :, [0, 2]] = 0
        return green_image

    elif color == 'b':  # blue
        blue_image = image.copy()
        blue_image[:, :, [0, 1]] = 0
        return blue_image

    else:
        return image  # In case you've entered an invalid color


def color_quantization(image, factor):
    """
    Apply color quantization to an image. The function processes each color
    channel independently and rounds the pixel values to the nearest multiple
    of the specified factor. The shape of the returned array is the same as the
    shape of the input image.
    """
    quantized_image = image.copy()
    for color in range(3):
        quantized_image[:, :, color] = (quantized_image[:, :, color] // factor) * (factor)

    return quantized_image


def enlarge(image, factor):
    """
    Enlarge the given image by repeating its pixels along the X and Y axes.
    The parameter 'factor' is the factor by which to enlarge the image. If the
    provided factor is not within the valid range (1 to 30, for safety reasons),
    a message is printed, and the original image is returned without modification.
    """
    if not (1 <= factor <= 30):
        print("Pick an integer number between 1 and 30.")
        return image

    else:
        enlarged_x = np.repeat(image, factor, axis=0)
        enlarged_image = np.repeat(enlarged_x, factor, axis=1)
        return enlarged_image


def pixelize(image, step):
    """
    Pixelize an image by reducing its resolution with a factor 'step'.
    Adjust the size of the pixelized image to match the original dimensions after enlarging.
    """
    image_pixelized = image[::step, ::step, :]  # Pixelize the image
    image_pixelized = enlarge(image_pixelized, step)  # Enlarge the pixelized image

    # Adjust the size of the pixelized image to match the original dimensions
    image_pixelized = image_pixelized[:image.shape[0], :image.shape[1], :]

    return image_pixelized


def multiple_manipulations(image, code):
    """
    Apply a sequence of image manipulations on the parameter "image", specified
    by the parameter "code".

    The code is allowed to be a string of characters. Each character in the
    code corresponds to a specific manipulation:
    '1' = Horizontal flip, '2' = Vertical flip, '3' = Horizontal and vertical flips,
    'r' = Color the image red, 'g' = Color the image green, 'b' = Color the image blue,
    'p': Pixelize the image with a step of 10, 'q': Apply color quantization with a factor of 54.

    Examples:
        code = [[0, 1, 2, 3], [2, 3, 0, 1], [1, 0, 3, 2], [3, 2, 1, 0]]
        code = [["0p", 1, "2b", 3], [2, "r3", 0, "q1"], ["g1", 0, "qp3", 2]]
    """
    new_image = image.copy()
    code_str = str(code)

    for letter in code_str:
        if letter == '1':
            new_image = h_flip(new_image)
        if letter == '2':
            new_image = v_flip(new_image)
        if letter == '3':
            new_image = h_flip(v_flip(new_image))
        if letter == 'r':
            new_image = color(new_image, 'r')
        if letter == 'g':
            new_image = color(new_image, 'g')
        if letter == 'b':
            new_image = color(new_image, 'b')
        if letter == 'p':
            new_image = pixelize(new_image, 10)
        if letter == 'q':
            new_image = color_quantization(new_image, 54)

    return new_image
