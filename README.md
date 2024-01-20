# LAB1 OPTICAL AND INFRARED, HEHE

## Description

Simple software with auto correlation function and polynomial fitting, the functions you will use at Asiago telescope.

## Installation

Install requirements.txt with pip, these are all the libraries needed to run the software.

To do that, open the terminal inside "INFRARED" folder with administrator privileges and type:

```
pip install -r requirements.txt
```

## Features

1. Function parameters checks and errors handling
2. Detailed log in the terminal for educational purposes
3. Optional input and output folders
4. All input format for images are supported
5. Optional output format (.tif, .fit, .fits, .png)
6. Automatic output image cutting given a luminosity threshold

## Usage

With the terminal, go inside "INFRARED" folder and type:

```
python
```

Then, type:

```
from utils import *
```

Now you can use the functions inside the software.
For example, with this function call you can try the software with the images inside "HD27214_first_5_images" folder,
the output
images will be saved inside "OUTPUT_FOLDER" folder, the output format will be .png, the output images will be cut
with a luminosity threshold of 0.02 times the value of the maximum luminosity of the image and the log will be printed
in the terminal:

```
auto_correlate_the_folder("HD27214_first_5_images/", "OUTPUT_FOLDER/", "png", True, 0.02)
```

The most general use case is:

```
auto_correlate_the_folder("INPUT_FOLDER/", "OUTPUT_FOLDER/")
```

## Authors

**Alberto Pesce**

**Bianca Pigato**











