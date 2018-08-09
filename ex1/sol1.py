import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc
from skimage import color


''' constants '''
VALID_REPRESENTATION = [1, 2]
NORMALIZE_VAL = 255
TRANSPOSE_MATRIX = [[0.299,      0.587,        0.114],
                    [0.59590059, -0.27455667, -0.32134392],
                    [0.21153661, -0.52273617, 0.31119955]]

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
The function shows the input image as the requested representation
'''


def imdisplay(filename, representation):
    image = read_image(filename, representation)
    fig, ax = plt.subplots()
    ax.imshow(image)


'''
The function transforms the input image from RGB representation to YIQ by multiplying in the transpose matrix
that converts the image from RGB to YIQ
'''


def rgb2yiq(imRGB):
    try:
        if imRGB.ndim is not 3:  # verify the there are 3 dimensions and the image is RGB
            return False
    except:
        raise ValueError("The argument you entered is not numpy object")

    yiq_from_rgb = np.array(TRANSPOSE_MATRIX)
    multiply_matrices = np.dot(imRGB, yiq_from_rgb)
    return multiply_matrices


'''
The function transforms the input image from YIQ representation to RGB by multiplying in the transpose matrix
that converts the image from YIQ to RGB
'''


def yiq2rgb(imYIQ):
    try:
        if imYIQ.ndim is not 3:  # verify the there are 3 dimensions and the image is YIQ
            return False
    except:
        raise ValueError("please enter a numpy object as an input")

    rgb_from_yiq = np.linalg.inv(np.array(TRANSPOSE_MATRIX))
    multiply_matrices = np.dot(imYIQ, rgb_from_yiq)
    return multiply_matrices


'''
Task 3.4
The function preforms histogram equalization on a given image.
The function calculates the original histogram of the image, equalizes the image, and calculates the new histogram/
'''


def histogram_equalize(im_orig):
    is_rgb = True if im_orig.ndim == 3 else False
    im_orig_as_gray = color.rgb2grey(im_orig)

    hist_orig, bin_edges = np.histogram(np.multiply(im_orig_as_gray, 255), bins=256, range=(0, 255))
    cdf = hist_orig.cumsum()
    cdf = np.floor((cdf / cdf[-1]) * 255)
    to_map = np.array(range(0, 256))

    image_equalized = np.multiply(im_orig_as_gray.copy(), 255)
    image_equalized = np.interp(image_equalized, to_map, cdf)

    image_equalized = image_equalized.reshape(im_orig_as_gray.shape)
    new_hist, bin_edges = np.histogram(image_equalized, bins=256, range=(0, 255))
    new_hist = np.divide(new_hist, 255)

    if is_rgb:
        image_as_YIQ = rgb2yiq(im_orig.copy())  # convert the image to YIQ
        image_as_YIQ = image_as_YIQ.reshape(-1)

        image_as_YIQ[::3] = image_equalized.reshape(-1)  # change the Y according to the histogram calculated
        image_as_YIQ = image_as_YIQ.reshape(im_orig.shape)
        image_equalized = yiq2rgb(image_as_YIQ)  # convert back to RGB

    return [image_equalized, hist_orig, new_hist]

'''
Task 3.5
The function preforms quantization to a given image, and return the image after quantization, and the minimized error
'''


def quantize(im_orig, n_quant, n_iter):
    # convert image to grey scale and find histogram
    im_orig_grey = color.rgb2grey(im_orig)

    histogram, array = np.histogram(im_orig_grey,bins=256,range=(0,1))

    # initialize z and error
    z = find_initial_z(histogram, im_orig_grey, n_quant)
    error = np.empty(n_iter)

    # iterate over the iters and calculate q, z, error for each time.
    for i in range(0, n_iter):
        q = calculate_q(histogram, z)
        z = calculate_z(z, q)
        error[i] = calculate_error(q, z, histogram)

        if i is not 0:
            # check if the difference between the iteration did not improve the error, and if so- return
            if error[i-1] < error[i]:
                error[i:] = None
                return output_final_image(z, q, im_orig_grey), error

    return output_final_image(z, q, im_orig_grey), error

'''
The function puts the computed q values int the suitable image indexes, according to z computation
'''

def output_final_image(z, q, input_im):
    q = np.floor(q).astype(int)
    lut = np.empty((256,))

    for i in range(q.shape[0]):
        lut[int(z[i]):int(z[i+1]) + 1] = q[i]

    input_im *= 255
    input_im = np.floor(input_im).astype(int)

    output_image = np.array(lut[input_im])
    return output_image


'''
The function creates the initial z values to start the algorithm with
'''


def find_initial_z(histogram, im_orig, n_quant):
    convern_cumsum_to = n_quant / im_orig.size
    cumsum_from_zero_to_quant = np.cumsum(histogram) * convern_cumsum_to
    binary_count = np.bincount(cumsum_from_zero_to_quant.astype(np.uint8))
    return np.array(np.cumsum(binary_count))


'''
The function calculates the q array, with the given z and original histogram
'''


def calculate_q(histogram, z):
    def calc_qi_upper(value):
        return histogram[value] * value
    f = np.vectorize(calc_qi_upper)

    def calc_qi_lower(value):
        return histogram[value]
    g = np.vectorize(calc_qi_lower)

    q = np.empty(z.shape[0]-1)
    for i in range(z.shape[0]-1):
        # calculate q[i] with the formula given in the tirgul
        upper = 0
        lower = 0
        # create array of all the values from z[i] to z[i+1]
        arr = np.array(range(z[i], z[i+1]))
        upper += sum(f(arr))
        lower += sum(g(arr))

        # verify there is no division in zero
        if np.equal(lower, 0):
            lower = 1
        q[i] = (upper / lower)

    return q


'''
The function calculates the z array, with the given q
'''


def calculate_z(z, q):
    def calc_zi(q1, qi1):
        return (q1 + qi1) / 2
    f = np.vectorize(calc_zi)

    for i in range(1, q.shape[0]-1):
        z[i+1] = f(q[i], q[i+1])
    z[0] = 0
    z[q.shape[0]] = 255

    return z


'''
The function calculates the error of the quantization by going over the z values and calculate for each index in
the middle it's error
'''


def calculate_error(q, z, histogram):
    # def calc_error(i, j):
    #     (q[i] - arr[j])**2 * histogram[]

    sum_error = 0
    for i in range(0, z.shape[0]-1):
        arr = np.array(range(z[i], z[i+1] + 1))
        for j in range(0, arr.size):  # todo
            sum_error += ((q[i] - arr[j])**2) * histogram[arr[j]]
    return sum_error
