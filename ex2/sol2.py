# **************** Imports *******************
import numpy as np
import cmath
from skimage import color
from scipy.fftpack import ifftshift, fftshift
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
The function converts a 1D discrete signal to its Fourier representation
'''


def DFT(signal):
    def dft_algo(n):
            return lambda i, j: cmath.e ** (-((2 * cmath.pi * np.complex(0, 1) * i * j) / n))

    img_shape = signal.shape
    signal.reshape(-1)
    dft_algo_res = dft_algo(img_shape[0])
    matrix = np.fromfunction(dft_algo_res, (img_shape[0], img_shape[0]))

    ret = np.dot(matrix, signal).reshape(img_shape)
    return ret

'''
The function converts a 1D signal in its Fourier representation to a discrete signal
'''


def IDFT(fourier_signal):
    def idft_algo(n):
            return lambda i, j: (cmath.e ** ((2 * cmath.pi * np.complex(0, 1) * i * j) / n))

    img_shape = fourier_signal.shape
    fourier_signal.reshape(-1)
    dft_algo_res = idft_algo(img_shape[0])
    matrix = np.fromfunction(dft_algo_res, (img_shape[0], img_shape[0]))

    return (np.dot(matrix, fourier_signal).reshape(img_shape)) / img_shape[0]


'''
The function converts a 1D discrete signal to its Fourier representation
'''


def DFT2(image):
    def dft_algo(n):
            return lambda i, j: cmath.e ** (-((2 * cmath.pi * np.complex(0, 1) * i * j) / n))

    img_shape = image.shape

    dft_algo_res = dft_algo(img_shape[0])
    dft_algo_res2 = dft_algo(img_shape[1])

    matrix = np.fromfunction(dft_algo_res, (img_shape[0], img_shape[0]))
    matrix2 = np.fromfunction(dft_algo_res2, (img_shape[1], img_shape[1]))

    res = np.dot(np.array(matrix), np.dot(image, np.array(matrix2)))
    return res


'''
The function converts a 2D signal in its Fourier representation to a discrete signal
'''


def IDFT2(fourier_image):
    def idft_algo(n):
            return lambda i, j: (cmath.e ** ((2 * cmath.pi * np.complex(0, 1) * i * j) / n))

    img_shape = fourier_image.shape
    dft_algo_res = idft_algo(img_shape[0])
    matrix = np.fromfunction(dft_algo_res, (img_shape[0], img_shape[0]))

    dft_algo_res2 = idft_algo(img_shape[1])
    matrix2 = np.fromfunction(dft_algo_res2, (img_shape[1], img_shape[1]))
    return np.dot(matrix, np.dot(fourier_image, matrix2))


'''
The function computes the magnitude of image derivatives
'''


def conv_der(im):
    # derivative vertically
    vertical = signal.convolve2d(im, DERIVATIVE_MATRIX, 'same')
    # derivative horizontally
    derivative_mat_trans = np.transpose(DERIVATIVE_MATRIX)
    horizon = signal.convolve2d(im, derivative_mat_trans, 'same')
    magnitude = np.sqrt((np.abs(horizon)**2) + (np.abs(vertical)**2))
    return magnitude

'''
The function computes the magnitude of image derivatives using Fourier transform
'''


def fourier_der(im):
    rows, cols = im.shape
    fourier_range_rows = range(np.floor(-rows/2).astype(np.int32)-1, np.floor(rows/2).astype(np.int32)-1)
    rows_shift = np.fft.fftshift(fourier_range_rows)
    fourier_range_cols = range(np.floor(-cols/2).astype(np.int32)-1, np.floor(cols/2).astype(np.int32)-1)
    cols_shift = np.fft.fftshift(fourier_range_cols)

    zero_rows_size = np.zeros((rows, rows))
    zero_cols_size = np.zeros((cols, cols))

    np.fill_diagonal(zero_rows_size, rows_shift)
    np.fill_diagonal(zero_cols_size, cols_shift)

    dft = DFT2(im)

    x_derivative = np.dot(zero_rows_size, dft)
    idft_x_der = np.abs(IDFT2(x_derivative) * ((2 * cmath.pi * np.complex(0, 1)) / (rows * cols)))

    y_derivative = np.dot(dft, zero_cols_size)
    idft_y_der = abs(IDFT2(y_derivative))

    return np.sqrt((np.abs(idft_x_der) ** 2) + (np.abs(idft_y_der) ** 2))

'''
The function blurs an image using a convolution with gaussian kernel g in image space
'''


def blur_spatial(im, kernel_size):
    gaussian_kernel = create_gaussian_kernal(kernel_size)
    image_conv_kernel = signal.convolve2d(im, gaussian_kernel, 'same')

    return image_conv_kernel

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
The function blurs an image, using an operation between the fourier image F with a gaussian kernel in the given size,
in fourier space G.
'''


def blur_fourier(im, kernel_size):
    # 1) create 2D gaussian kernel in kernel size -g
    gaussian_kernel = create_gaussian_kernal(kernel_size)

    # 2) pad gaussian kernel with zeros
    padded_gaussian = np.zeros(im.shape)
    r_size = im.shape[0]
    r_size = int(np.floor(r_size / 2))

    c_size = im.shape[1]
    c_size = int(np.floor(c_size / 2))

    gaussian_center = int(np.floor(kernel_size/2) + 1)

    padded_gaussian[r_size-gaussian_center:r_size+gaussian_center-1,
                    c_size-gaussian_center:c_size+gaussian_center-1] = np.fft.ifftshift(gaussian_kernel)

    # 2) transform g to fourier representation- G
    padded_gaussian = ifftshift(DFT2(fftshift(padded_gaussian)))

    # 3) transform image f to fourier representation- F
    image_fourier = fftshift(DFT2(im))

    # 4) multiply point wise R = F*G
    gaussian = image_fourier * padded_gaussian

    # 5) preform inverse transform on R
    image_blurred = np.real(IDFT2(fftshift(gaussian)))
    return image_blurred