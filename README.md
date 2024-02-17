# Project name: Image manipulation in Numpy
- Name student: Raf Mesotten
- Date project: Feb. 2024
- Project folder: project_numpy_3
- Github: https://github.com/craftyraf/project_numpy_3

### Set up your virtual environment
- Import 'environment.yml' (in the same folder as 'README.md'), e.g. with Anaconda Navigator -> Environments -> Import
- Start -> Anaconda Powershell Prompt -> typ: "conda activate environment" (in case 'environment' is the name of the environment you just imported)
- Next, still in Anaconda Powershell Prompt, typ: "jupyter notebook", in order to open jupyter notebook. In case the notebook is in a different directory, e.g. D:\Documents, typ: "jupyter notebook --notebook-dir D:\Documents".

### Run the notebook 'Input_to_output.ipynb' to test the project
- Navigate to the project folder, and the subfolder 'notebooks'. You can find this notebook in the the folder 'notebooks'.
- This notebook turns the input image (in the folder data\input) into output images (in the folder data\output)
- This notebook uses scripts from the folder 'scripts'.
- I've already designed some functions to be robust to incorrect input, such as showing the original image when an unknown color is selected and handling unusual list lengths in the 'create_colorful_big_one' function. I realize that there is room for optimization in this regard, but in my opinion that is outside the scope of the exercise.

### The scripts that are used by the notebook:
- image_manipulations.py is a script with functions that manipulate the input image (e.g. change color, flip horizontally, pixelize an image, quantize colors,...)
- image_builders.py is a script with functions that turns the input image (or manipulated images) into a bigger image (e.g. grid_with_flips, create_colorful_big_one)

### Problems with running the notebook? To get an idea of what I did, you can take a look at:
- 'Input_to_output.pdf' in the folder 'notebooks'
- Separate images in the folder 'data\output'
- The scripts in the folder 'scripts'. I provided each function with a clear docstring.
- In case something isn't clear: contact me via email or by telephone.

### You can ignore the folder '.idea'
- I used Pycharm to program, and Pycharm created this folder.

I'm happy to receive your feedback!
Raf
