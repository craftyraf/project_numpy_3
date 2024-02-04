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


def tile(image, rows, columns):
    """
    Tile the input image horizontally (rows) and vertically (columns).
    """
    tiled_image = np.tile(image, (rows, columns, 1))
    return tiled_image


def color(image, color):
    """
    Color the input image red ('r'), green ('g') or blue ('b'). Return the
    original image in case of an unknown color.
    """
    if color == 'r':
        red_image = image.copy()
        red_image[:, :, [1, 2]] = 0
        return red_image

    elif color == 'g':
        green_image = image.copy()
        green_image[:, :, [0, 2]] = 0
        return green_image

    elif color == 'b':
        blue_image = image.copy()
        blue_image[:, :, [0, 1]] = 0
        return blue_image

    else:
        return image  # In case you've entered an invalid color


def enlarge(image, factor):
    """
    Enlarge the given image by repeating its pixels along X- and Y- axes.
    """
    if not (1 <= factor <= 20):
        print("Pick an integer between 1 and 20.")
        return image

    else:
        enlarged_x = np.repeat(image, factor, axis=0)
        enlarged_image = np.repeat(enlarged_x, factor, axis=1)
        return enlarged_image


def pixelize(image, step):
    """
    Pixelize an image by reducing its resolution with a factor 'step'.
    """
    image_pixelized = image[::step, ::step, :]
    return image_pixelized
