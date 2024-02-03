# CODE VAN DE MEEST ADVANCED VERSIE, vb 3 aparte blokjes in jupyter notebook,
# die 3 aparte afbeeldingen
# Alle soorten inputs moet ie mee omkunnen.

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

print(os.getcwd())  # output: C:\Users\XXX\PycharmProjects\project_numpy_3
loc_input_img = r"C:\Users\CraftyRaf\PycharmProjects\project_numpy_3\data\input\carcassonne.jpg"
#loc_input_img = os.path.join('..', 'data', 'input', 'carcassonne.jpg')
image = Image.open(loc_input_img)
np_image = np.array(image)
plt.imshow(np_image)
plt.show()

print(np.shape(np_image))
rep_x = np.repeat(np_image, 3, axis=0)
print(np.shape(rep_x))
rep_y = np.repeat(rep_x, 3, axis=1)
print(np.shape(rep_y))


def tile_image(image, rows, columns):
    """
    Tile a 3D NumPy array along specified axes.
    """
    tiled_image = np.tile(image, (rows, columns, 1))
    return tiled_image


def h_flip_image(image):
    """
    Flip the input image horizontally, reversing the order of pixels within
    each row.
    """
    h_flipped_image = np.flip(image, axis=1)
    return h_flipped_image


def v_flip_image(image):
    """
    Flip the input image vertically, reversing the order of pixels within
    each column.
    """
    v_flipped_image = np.flip(image, axis=0)
    return v_flipped_image


def color_image(image, color):
    """
    Color the input image red ('r'), green ('g') or blue ('b').
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


def pixelize_image(image, step):
    np_image_pixelized = image[::step, ::step, :]
    return np_image_pixelized


def enlarge_image(image, factor):
    """
    Enlarges the given image by repeating its pixels along X- and Y- axes.
    """
    if not (1 <= factor <= 20):
        print("Pick an integer between 1 and 20.")
        return image

    else:
        enlarged_x = np.repeat(image, factor, axis=0)
        enlarged_image = np.repeat(enlarged_x, factor, axis=1)
        return enlarged_image


def multiple_actions(image, code):
    result = image.copy()

    function_mapping = {
        'h': h_flip_image,
        'v': v_flip_image,
        'r': lambda result: color_image(result, 'r'),  # Using lambda to pass the color argument
        'g': lambda result: color_image(result, 'g'),
        'b': lambda result: color_image(result, 'b'),
        'p': lambda result: pixelize_image(result, 10),
    }

    for function_name in code:
        if function_name in function_mapping:
            result = function_mapping[function_name](result)

    return result

test = multiple_actions(np_image, 'hvr')
plt.imshow(test)
plt.show()

test = multiple_actions(np_image, 'rvh')
plt.imshow(test)
plt.show()

############## STEP 4 - MANIPULATIONS

# Manipulation 1
result_manipulation_1 = tile_image(np_image, 3, 8)
plt.imshow(result_manipulation_1)
plt.show()

# Manipulation 2
row_1 = tile_image(np_image, 1, 6)
row_2 = tile_image(h_flip_image(np_image), 1, 6)
row_3 = tile_image(v_flip_image(np_image), 1, 6)
row_4 = tile_image(h_flip_image(v_flip_image(np_image)), 1, 6)
result_manipulation_2 = np.vstack([row_1, row_2, row_3, row_4])
plt.imshow(result_manipulation_2)
plt.show()

# Manipulation 3
d = np.shape(np_image)  # d[0] = 174 pixels in case of the example image
row_1 = tile_image(color_image(np_image, 'b'), 1, 4)
row_2 = tile_image(color_image(np_image, 'r'), 1, 4)
row_3 = tile_image(color_image(np_image, 'r'), 1, 4)
row_4 = tile_image(color_image(np_image, 'g'), 1, 4)
result_manipulation_3 = np.vstack([row_1, row_2, row_3, row_4])
print(np.shape(result_manipulation_3))
result_manipulation_3[d[0]:3*d[0], d[0]:3*d[0], :] = enlarge_image(np_image, 2)
plt.imshow(result_manipulation_3)
plt.show()


############## STEP 6 - YOUR OWN MANIPULATIONS
# Pixelize
plt.imshow(pixelize_image(np_image, step=10))
plt.show()

def create_colorful_big_one(colors):
    if len(colors) % 4 != 0:
        print("The amount of values in your list should be 8 or greater, and "
              "a multiple of 4.")
    else:
        grid_length = (len(colors) / 4) + 1
        row_1 = tile_image()



create_colorful_big_one(['b','b','b','b','r','r','g','g','g','g','r','r'])


# Manipulation 2b (respecting the rules of the carcassonne game)

# Creating a 2x2 square image using 4 tiles with different orientations
top_left = np_image
bottom_left = v_flip_image(np_image)
top_right = h_flip_image(np_image)
bottom_right = h_flip_image(v_flip_image(np_image))

left = np.vstack([top_left, bottom_left])
right = np.vstack([top_right, bottom_right])
little_square = np.hstack([left, right])

# Tiling the 2x2 image
result = tile_image(little_square, 3, 8)
plt.imshow(result)
plt.show()


np_image = np.array(image)
def grid_with_flips(image, matrix):
    rows, columns = np.shape(matrix)
    reshaped_matrix = matrix.reshape(1, -1)

    images_list = []

    for element in reshaped_matrix[0]:
        if element == 1:
            images_list.append(h_flip_image(image))
        elif element == 2:
            images_list.append(v_flip_image(image))
        elif element == 3:
            images_list.append(h_flip_image(v_flip_image(image)))
        else:
            images_list.append(image)

    # Concatenating the images horizontally (axis = 1)
    one_long_image = np.concatenate(images_list, axis=1)
    final_result = one_long_image.reshape(
        rows * np_image.shape[0],
        np_image.shape[1] / rows,
        np_image.shape[2])

    return final_result


 #   concatenated_images = np.concatenate(images_list, axis=1)
 #   result = concatenated_images.reshape(3, 3)
 #   return result

matrix = np.array([[1, 2, 3], [0, 1, 2], [3, 0, 1]])
result_image = grid_with_flips(np_image, matrix)
plt.imshow(result_image)
plt.show()

print(np.shape(result_image))







[[1 for i in range(7)] for j in range(3)]
[[j for i in range(7)] for j in range(4)]











import matplotlib.pyplot as plt
import numpy as np

# Assuming you have defined h_flip_image and v_flip_image functions

np_image = np.array(image)

def grid_with_flips(image, matrix):
    rows, columns = np.shape(matrix)
    print(rows)

    reshaped_matrix = matrix.reshape(1, -1)

    images_list = []

    for element in reshaped_matrix[0]:
        if element == 1:
            images_list.append(h_flip_image(image))
        elif element == 2:
            images_list.append(v_flip_image(image))
        elif element == 3:
            images_list.append(h_flip_image(v_flip_image(image)))
        else:
            images_list.append(image)

    # Concatenate the images horizontally (axis=1)
    one_long_image = np.concatenate(images_list, axis=1)
    print(np.shape(one_long_image))
    plt.imshow(one_long_image)
    plt.show()

    sub_arrays = np.split(one_long_image, rows)
    for sub_array in sub_arrays:
        plt.imshow(sub_array)
        plt.show()

    result_image = one_long_image[:, :].reshape(rows*175, -1, 3)
    print(np.shape(result_image))
    plt.imshow(result_image)
    plt.show()


matrix = np.array([[1, 2, 3], [0, 1, 2], [3, 0, 1]])
result_image = grid_with_flips(np_image, matrix)





enlarged_image = enlarge_image(np_image, 10)
plt.imshow(enlarged_image)
plt.show()



import numpy as np

def multiple_actions(image, codes_list):
    results = []

    function_mapping = {
        'h': h_flip_image,
        'v': v_flip_image,
        'r': lambda img: color_image(img, 'r'),
        'g': lambda img: color_image(img, 'g'),
        'b': lambda img: color_image(img, 'b'),
        'p': lambda img: pixelize_image(img, 10),
    }

    for codes in codes_list:
        result = image.copy()

        transformations = []
        for function_name in codes:
            if function_name in function_mapping:
                result = function_mapping[function_name](result)
                transformations.append(result.copy())  # Append each transformation individually

        results.extend(transformations)

    return results

# Example usage with an array of strings
codes_list = [['rvh', 'h', 'b'], ['g', 'vhv', 'hg']]
image = np_image

result_list = multiple_actions(image, codes_list)

# Concatenate the individual transformations horizontally into a single image
concatenated_image = np.concatenate(result_list, axis=1)

# Display the concatenated image
plt.imshow(concatenated_image)
plt.show()