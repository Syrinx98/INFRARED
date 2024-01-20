import numpy as np
from matplotlib import pyplot as plt

from utils import *

# Put here the given data for x and y
# read the polyfit_and_plot function description in "utils.py" for more info
data = [
    {"x": 33.5, "y": 53.58},
    {"x": 33.4, "y": 52.87},
    {"x": 33.2, "y": 53.30},
    {"x": 33.0, "y": 54.91},
    {"x": 32.8, "y": 51.59},
    {"x": 32.6, "y": 65.16},
    {"x": 33.7, "y": 63.02},
    {"x": 33.9, "y": 53.08}
]

if __name__ == '__main__':
    # Use cases of the functions in "utils.py"
    # polyfit_and_plot(data)

    # General use case of auto_correlate_the_folder, note some parameters are optional
    # auto_correlate_the_folder("INPUT_FOLDER/", "OUTPUT_FOLDER/")

    # General use case of auto_correlate_the_folder, utilising the optional parameters
    # auto_correlate_the_folder("INPUT_FOLDER/", "OUTPUT_FOLDER/", "png", True, 0.02)

    # HD27214
    auto_correlate_the_folder("HD27214_first_5_images/", "OUTPUT_FOLDER/", "png", True, 0.02)

    # HD39315
    auto_correlate_the_folder("HD39315_first_5_images/", "OUTPUT_FOLDER/", "png", True, 0.02)

    # HD41116
    auto_correlate_the_folder("HD41116_first_5_images/", "OUTPUT_FOLDER/", "png", True, 0.02)

    # HD50522
    auto_correlate_the_folder("HD50522_first_5_images/", "OUTPUT_FOLDER/")

    # HD77327
    auto_correlate_the_folder("HD77327_first_5_images/", "OUTPUT_FOLDER/")
