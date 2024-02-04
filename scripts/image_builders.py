import numpy as np
import scripts.image_manipulations as img

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
