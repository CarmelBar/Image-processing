# **************** Imports *******************
import numpy as np
from skimage import color
import scipy.signal as signal
import scipy.misc as misc
import scipy.stats as st


# **************** Constants *******************
VALID_REPRESENTATION = [1, 2]
NORMALIZE_VAL = 255
DERIVATIVE_MATRIX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

'''
The function reads an image as the requested representation (grey scale or RGB)
'''


def read_image(filename, representation):
    if representation not in VALID_REPRESENTATION:
        return
    else:
        image = misc.imread(filename).astype(np.float64)
        image = np.divide(image, NORMALIZE_VAL)

        if representation == VALID_REPRESENTATION[0]:
            image = color.rgb2grey(image)
            return image
        return image

'''
The function creates a 1 dimensional gaussian kernel in the given size
'''


def create_1d_gaussian_kernal(kernel_size):
    base = np.array([1,1])
    res = base
    for i in range(0, kernel_size - 2):
        res = signal.convolve(res, base)
    return np.array([res / np.sum(res)])



'''
The function blurs an image using a convolution with gaussian kernel g in image space
'''


def blur(im, kernel_size):
    gaussian_kernel = create_1d_gaussian_kernal(kernel_size)
    image_conv_kernel = signal.convolve2d(im, gaussian_kernel, 'same')
    return image_conv_kernel, gaussian_kernel


'''
The function builds gaussian pyramid and returns array in length of max levels of reduced images
'''


def build_gaussian_pyramid(im, max_levels, filter_size):
    cur_image = im
    ret_array = [im]
    filter = None
    for i in range(1, max_levels):
        blurred_image, filter = blur(cur_image, filter_size)
        cur_image = blurred_image[::2,::2]
        ret_array.append(cur_image)
    return ret_array, filter

'''
The function creates a gaussian kernel in the given size
'''


def create_gaussian_kernal(kernel_size):
    interval = 7 / kernel_size
    x = np.linspace(-3-interval/2., 3+interval/2., kernel_size+1)
    kern_1d = np.diff(st.norm.cdf(x))
    kernel_2_d = np.sqrt(np.outer(kern_1d, kern_1d))
    kernel = kernel_2_d/kernel_2_d.sum()
    return kernel

'''
The function blurs an image using a convolution with gaussian kernel g in image space
'''


def blur_spatial(im, kernel_size):
    gaussian_kernel = create_gaussian_kernal(kernel_size)
    image_conv_kernel = signal.convolve2d(im, gaussian_kernel, 'same')
    return image_conv_kernel