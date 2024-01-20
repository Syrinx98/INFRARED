import datetime

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import scipy as sp

from PIL import Image

"""
This is a simple parabola definition function
:param x: the x value
:param a: the a coefficient
:param b: the b coefficient
:param c: the c coefficient

:return: the value of the parabola in x
"""


def parabola_definition(x, a, b, c):
    return a * (x ** 2) + b * x + c


"""
This function will do the polynomial interpolation with numpy.polyfit
and will plot the results, the return it's a function representing the parabola

:param data: the data to be fitted, it must be a list of dictionaries with x and y values as keys
"""


def polyfit_and_plot(data: list):
    # data must be a list of tuples to store the data given by the observatory
    # x values: position of the secondary mirror
    # y values: size in pixels of the FWHM

    # check for data structure to be correct, we need a list of dictionaries
    if len(data) == 0:
        print("No data given")
        return

    # check for data structure to be correct, we need a list of dictionaries
    if "x" not in data[0].keys() or "y" not in data[0].keys():
        print("Data structure is not correct")
        return

    # check for data structure to be correct, we need a list of dictionaries with only two keys
    if len(data[0].keys()) != 2:
        print("Data structure is not correct")
        return

    # now we want to build a parabola that fits the data
    # we will use the numpy.polyfit function, that will return the coefficients of the polynomial
    # with these we can find the minimum value that will be the best position for the secondary mirror

    # a parabola is defined as y = ax^2 + bx + c, so the degree of the polynomial is 2
    deg = 2

    # let's build the polynomial, we need x_data and y_data stored as separated lists, we should take it from data
    x_data = []
    y_data = []

    for d in data:
        x_data.append(d["x"])
        y_data.append(d["y"])

    """
    technical note:
        we cannot have size conflicts between the two lists, since they are taken by a list of dictionaries that has 
        been built by us. More formally, we could define a class that has x and y as attributes, and then build a 
        list of objects of that class, in this case it's not necessary since i did some manual checks before.
    """
    poly = np.polyfit(x_data, y_data, deg)

    print("Fitting done! The coefficients of the polynomial are: ")
    a = poly[0]
    b = poly[1]
    c = poly[2]
    print("a: " + str(a) + " b: " + str(b) + " c: " + str(c))

    # find the minimum value of the parabola with the formula -b/2a
    minimum = -b / (2 * a)

    print("The minimum value of the parabola is: " + str(minimum))

    # now that we have the coefficients of the polynomial, we can plot it
    # let's plot the parabola but prettier, we just take more values for x and y, let's say 100
    x_data_pretty_parabola = np.arange(0, 100, 1)
    y_data_pretty_parabola = []

    # calculate the y values for the parabola
    for x in x_data_pretty_parabola:
        y_data_pretty_parabola.append(parabola_definition(x, a, b, c))

    # plot the results
    plt.plot(x_data_pretty_parabola, y_data_pretty_parabola, 'o', label='hehe, boy')
    plt.legend()

    # write the value of the minimum below the x-axis
    plt.text(minimum, 0, str(minimum), fontsize=12)

    # plot x and y value of the minimum in the graph
    plt.plot(minimum, np.polyval(poly, minimum), 'ro', label='minimum')
    plt.show()


"""
This function computes the auto correlation of a matrix
:param mat: the matrix to be auto correlated
:return: the auto correlated matrix

We give as input the .fits file in matrix form, then we compute the auto correlation of the matrix and return it.
"""


def auto_correlate(fits_image_matrix):
    # Performing a 2D Fast Fourier Transform (FFT)
    f1 = sp.fftpack.fft2(fits_image_matrix)
    # Shifting the zero-frequency component to the center
    f1shift = sp.fftpack.fftshift(f1)
    # Computing the squared magnitude of the frequency spectrum
    f1abs = np.abs(f1shift) ** 2
    # Performing an inverse FFT
    f2 = sp.fftpack.ifft2(f1abs)
    # Shifting the zero-frequency component back to the center
    f2shift = sp.fftpack.fftshift(f2)
    # Calculating the magnitude of the inverse transform
    f2abs = np.abs(f2shift)
    return f2abs


"""
This function will find the illuminated area to cut
:param: image_matrix: the matrix of the image
:param: threshold: the threshold for the image
:return: the coordinates of the area to cut
"""


def find_illuminated_area(image_matrix, threshold):
    # Trova le coordinate dei pixel con valore superiore alla soglia
    illuminated_pixels = np.where(image_matrix > threshold)

    # Calcola le coordinate minime e massime dell'area illuminata
    min_x, max_x = np.min(illuminated_pixels[1]), np.max(illuminated_pixels[1])
    min_y, max_y = np.min(illuminated_pixels[0]), np.max(illuminated_pixels[0])

    return min_x, max_x, min_y, max_y


"""
This function will take a folder, apply the auto correlation to all the images and do an average.
Then it will filter the resulting average with a gaussian filter, the result of this operation
will be subtracted to the average

:param: input_folder: the folder where all the images are stored
:param: output_folder: the folder where the output will be stored
:param: output_format: the format of the output, it can be png, fits, fit, tif
:param: cut_image: if the resulting image must be cut or not
:param: cut_image_threshold: the threshold for the cut

"""


def auto_correlate_the_folder(input_folder: str, output_folder: str, output_format: str = "png",
                              cut_image: bool = False, cut_image_threshold: float = 0):
    # close all current plots
    plt.close('all')
    print("")
    print("=======================================")

    # check if input folder exists
    if not os.path.exists(input_folder):
        print("Input folder does not exist")
        return

    # check if output folder exists
    if not os.path.exists(output_folder):
        print("Output folder does not exist")
        return

    # check if input folder is empty
    input_files = os.listdir(input_folder)
    if len(input_files) == 0:
        print("Input folder is empty")
        return

    if output_format not in ["png", "fits", "fit", "tif"]:
        print("Invalid output format. Accepted formats are 'png', 'fits', 'fit', 'tif'")
        return

    # controlla che il formato di input folder e output folder sia corretto
    if input_folder[-1] != "/":
        input_folder = input_folder + "/"
    if output_folder[-1] != "/":
        output_folder = output_folder + "/"

    # Brief review of the inserted parameters
    print("Executing auto_correlate_the_folder with the selected parameters:")
    print("input folder: the input will be taken from                -> {}".format(input_folder))
    print("output folder: the output will be stored in               -> {}".format(output_folder))
    print("output format: the output will have the following format  -> {}".format(output_format))
    if cut_image:
        print("cut image: all the pixels having a value lower than {} times "
              "the max pixel value of the image will be cut".format(cut_image_threshold))

    # cancella i plt presenti
    plt.close()

    # Generate the name for the output file, it is unique since it is based on the current date and time
    now = datetime.datetime.now()
    name = input_folder.replace("/", "") + "_AUTOCORRELATE_" + now.strftime("%d-%m-%Y_%H-%M-%S")

    # Reading the first image
    ima = Image.open(input_folder + input_files[0])
    ima = np.array(ima)

    # Initializing a zero matrix of the same size as the image
    base_matrix = np.zeros((ima.shape[0], ima.shape[1]))

    print("")
    print("START PROCESSING")
    # Processing each image
    for a, b in enumerate(input_files[:]):
        # Reading the image
        image = Image.open(input_folder + b)
        image = np.array(image)
        # Converting the image to a matrix
        image_matrix = np.asmatrix(image[:, :])
        # Calculating the auto correlation of the image
        auto_correlated_image = auto_correlate(image_matrix)
        # Summing the auto correlation results to base_matrix
        base_matrix = base_matrix + auto_correlated_image
        print("Processed image {} of {}".format(a + 1, len(input_files)))
    print("END PROCESSING")
    print("")

    # using astropy.io.fits to apply a gaussian filter to the image
    base_matrix_fitted = sp.ndimage.filters.gaussian_filter(base_matrix, sigma=0.8)

    # subtract base_matrix and base_matrix_fitted
    base_matrix = base_matrix - base_matrix_fitted

    if cut_image:
        # Finding the maximum value of base_matrix
        max_base_matrix = np.max(base_matrix)

        # find the coordinates in the matrix where to cut the image,
        # the coordinates are given only by the illuminated area
        # the threshold is calculated with the max and the min value
        threshold = 0.02 * max_base_matrix
        min_x, max_x, min_y, max_y = find_illuminated_area(base_matrix, threshold=threshold)

        print("Cutting resulting image in these coordinates: ")
        print("min_x: {}".format(min_x))
        print("max_x: {}".format(max_x))
        print("min_y: {}".format(min_y))
        print("max_y: {}".format(max_y))
        # cut the image
        base_matrix = base_matrix[min_x: max_x, min_y: max_y]

    # Saving the resulting image as the output_format
    if output_format == "png":
        plt.imshow(base_matrix, cmap='gray')
        plt.colorbar()
        plt.savefig(output_folder + name + ".png")
    elif output_format == "fit":
        hdu = fits.PrimaryHDU(base_matrix)
        hdu.writeto(output_folder + name + ".fit", overwrite=True)
    elif output_format == "fits":
        hdu = fits.PrimaryHDU(base_matrix)
        hdu.writeto(output_folder + name + ".fits", overwrite=True)
    elif output_format == "tif":
        plt.imshow(base_matrix, cmap='gray')
        plt.colorbar()
        plt.savefig(output_folder + name + ".tif")
