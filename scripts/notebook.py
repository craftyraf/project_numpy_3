from PIL import Image
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import os
import scripts.image_manipulations as img
import scripts.image_builders as bld

loc_input_img = os.path.join('..', 'project_numpy_3', 'data', 'input', 'carcassonne.jpg')
output_path = os.path.join('..', 'project_numpy_3', 'data', 'output')
# absolute_path = os.path.abspath(loc_input_img)
# print(absolute_path)  # Check the absolute path in case something goes wrong
matplotlib.use('TkAgg', force=True)  # Images don't load without this...

image = Image.open(loc_input_img)
np_image = np.array(image)

plt.imshow(np_image)
# plt.savefig(os.path.join(output_path, 'original.jpg'))
plt.show()



def multiple_manipulations(image, code):
    result = image.copy()

    function_mapping = {
        'h': img.h_flip,
        'v': img.v_flip,
        'r': lambda result: img.color(result, 'r'),  # Using lambda to pass the color argument
        'g': lambda result: img.color(result, 'g'),
        'b': lambda result: img.color(result, 'b'),
        'p': lambda result: img.pixelize(result, 10),  # Using lambda to pass the pixelize argument
    }

    for function_name in code:
        if function_name in function_mapping:
            result = function_mapping[function_name](result)

    return result

test = multiple_manipulations(np_image, 'hvr')
plt.imshow(test)
plt.show()

test = multiple_manipulations(np_image, 'rvh')
plt.imshow(test)
plt.show()













import numpy as np

def multiple_actions(image, codes_list):
    results = []

    function_mapping = {
        'h': img.h_flip,
        'v': img.v_flip,
        'r': lambda image_: img.color(image_, 'r'),
        'g': lambda image_: img.color(image_, 'g'),
        'b': lambda image_: img.color(image_, 'b'),
        'p': lambda image_: img.pixelize(image_, 10),
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








########################################
### STEP 4 - "Do my image manipulations"
########################################


## Manipulation 1
result_manipulation_1 = bld.tile(np_image, 3, 8)
plt.imshow(result_manipulation_1)
# plt.savefig(os.path.join(output_path, 'STEP_4__manipulation_1.jpg'))
plt.show()


## Manipulation 2
row_1 = bld.tile(np_image, 1, 6)
row_2 = bld.tile(img.h_flip(np_image), 1, 6)
row_3 = bld.tile(img.v_flip(np_image), 1, 6)
row_4 = bld.tile(img.h_flip(img.v_flip(np_image)), 1, 6)
result_manipulation_2 = np.vstack([row_1, row_2, row_3, row_4])
plt.imshow(result_manipulation_2)
# plt.savefig(os.path.join(output_path, 'STEP_4__manipulation_2.jpg'))
plt.show()


## Manipulation 2b (extra, respecting the rules of the Carcassonne game)

# Creating a 2x2 square image using 4 tiles with different orientations
top_left = np_image
bottom_left = img.v_flip(np_image)
top_right = img.h_flip(np_image)
bottom_right = img.h_flip(img.v_flip(np_image))

left = np.vstack([top_left, bottom_left])
right = np.vstack([top_right, bottom_right])
little_square = np.hstack([left, right])

# Tiling the 2x2 image
result = bld.tile(little_square, 3, 8)
plt.imshow(result)
# plt.savefig(os.path.join(output_path, 'STEP_4__manipulation_2b.jpg'))
plt.show()


## Manipulation 3

# Creating the background first
row_1 = bld.tile(img.color(np_image, 'b'), 1, 4)  # 1 row with 4 blue
row_2 = bld.tile(img.color(np_image, 'r'), 1, 4)  # 1 row with 4 red
row_3 = bld.tile(img.color(np_image, 'r'), 1, 4)  # 1 row with 4 red
row_4 = bld.tile(img.color(np_image, 'g'), 1, 4)  # 1 row with 4 green
result_manipulation_3 = np.vstack([row_1, row_2, row_3, row_4])

# Putting the big one on top of it
d = np.shape(np_image)
result_manipulation_3[d[0]:3*d[0], d[0]:3*d[0], :] = img.enlarge(np_image, 2)
plt.imshow(result_manipulation_3)
# plt.savefig(os.path.join(output_path, 'STEP_4__manipulation_3.jpg'))
plt.show()


#############################################
### STEP 5 - "Generalize these manipulations"
#############################################


## Manipulation 1
input_matrix_1 = [[1 for i in range(7)] for j in range(3)]
result_1 = bld.grid_with_flips(np_image, input_matrix_1)
plt.imshow(result_1)
# plt.savefig(os.path.join(output_path, 'STEP_5__manipulation_1.jpg'))
plt.show()


## Manipulation 2a
input_matrix_2a = [[j for i in range(7)] for j in range(4)]
result_2a = bld.grid_with_flips(np_image, input_matrix_2a)
plt.imshow(result_2a)
# plt.savefig(os.path.join(output_path, 'STEP_5__manipulation_2a.jpg'))
plt.show()


## Manipulation 2b - respecting the rules of the Carcassonne game
input_matrix_2b = [[0, 1, 2, 3], [2, 3, 0, 1], [1, 0, 3, 2], [3, 2, 1, 0]]
result_2b = bld.grid_with_flips(np_image, input_matrix_2b)
plt.imshow(result_2b)
# plt.savefig(os.path.join(output_path, 'STEP_5__manipulation_2b.jpg'))
plt.show()


## Manipulation 3a - big one
input_matrix_3a = ['b', 'b', 'b', 'b', 'r', 'r', 'g', 'g', 'g', 'g', 'r', 'r']
big_one = bld.create_colorful_big_one(np_image, input_matrix_3a)
plt.imshow(big_one)
# plt.savefig(os.path.join(output_path, 'STEP_5__manipulation_3a.jpg'))
plt.show()


## Manipulation 3b - bigger one
input_matrix_3b = ['r', 'g', 'b', 'r', 'g', 'b', 'r', 'g', 'b', 'r', 'g', 'b', 'r', 'g', 'b', 'r', 'g', 'b', 'r', 'g']
big_one = bld.create_colorful_big_one(np_image, input_matrix_3b)
plt.imshow(big_one)
# plt.savefig(os.path.join(output_path, 'STEP_5__manipulation_3b.jpg'))
plt.show()


## Manipulation 3c - big one, but with len(matrix) % 4 != 0
input_matrix_3c = ['r', 'g', 'b', 'r', 'g', 'b', 'r', 'g', 'b']
big_one = bld.create_colorful_big_one(np_image, input_matrix_3c)
plt.imshow(big_one)
# plt.savefig(os.path.join(output_path, 'STEP_5__manipulation_3c.jpg'))
plt.show()


#####################################
### STEP 6 - "Your own manipulations"
#####################################

## Pixelized image
pixelized_image = img.pixelize(np_image, step=10)  # step 10 = 10 pixels form 1 big pixel
plt.imshow(pixelized_image)
# plt.savefig(os.path.join(output_path, 'STEP_6__manipulation_1.jpg'))
plt.show()

## Quantized colors
quantized_image = img.color_quantization(np_image, 54)  # factor 54 = manually chosen optimum
plt.imshow(quantized_image)
# plt.savefig(os.path.join(output_path, 'STEP_6__manipulation_2.jpg'))
plt.show()
