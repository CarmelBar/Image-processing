# **************** Imports *******************
import numpy as np
from skimage import color
import scipy.signal as signal
import scipy.misc as misc
import matplotlib.pyplot as plt
import os


# **************** Constants *******************
VALID_REPRESENTATION = [1, 2]
NORMALIZE_VAL = 255

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


def expand_image(im, kernel):
    zero_padded_large_image = np.zeros((im.shape[0]*2, im.shape[1]*2))
    zero_padded_large_image[::2, ::2] = im[:, :]
    out = 4 * signal.convolve2d(zero_padded_large_image, kernel, 'same')
    return out

'''
The function builds laplacian pyramid and returns array in length of max levels of reduced images
'''


def build_laplacian_pyramid(im, max_levels, filter_size):
    gaussian_array, kernel = build_gaussian_pyramid(im, max_levels, filter_size)
    laplacian_array = list()
    k = len(gaussian_array)
    for i in range(0, k-1):
        gaussian_image = gaussian_array[i]
        expended_gaussian = expand_image(gaussian_array[i+1], kernel)
        if expended_gaussian.shape[0] > gaussian_image.shape[0]:
            expended_gaussian = np.delete(expended_gaussian, (-1), axis=0)
        if expended_gaussian.shape[1] > gaussian_image.shape[1]:
            expended_gaussian = np.delete(expended_gaussian, (-1), axis=1)
        laplacian_array.append(np.subtract(gaussian_image, expended_gaussian))
    laplacian_array.append(gaussian_array[-1])
    return laplacian_array, kernel

'''
'''


def laplacian_to_image(lpyr, filter_vec, coeff):
    output = np.zeros((lpyr[0].shape[0], lpyr[0].shape[1]), dtype=np.float64)
    for i in range(len(lpyr)-1, 0, -1):
        laplacian = expand_image(lpyr[i], filter_vec)
        laplacian_b = lpyr[i-1]
        if laplacian.shape[0] > laplacian_b.shape[0]:
            laplacian = np.delete(laplacian, -1, axis=0)
        if laplacian.shape[1] > laplacian_b.shape[1]:
            laplacian = np.delete(laplacian, (-1), axis=1)
        tmp = laplacian + laplacian_b
        lpyr.pop()
        lpyr.pop()
        lpyr *= coeff[i]
        lpyr.append(tmp)
        output = tmp
    return output


def render_pyramid(pyr, levels):
    black_image_rows = pyr[0].shape[0]
    black_image_cols = 0
    for i in range(0, levels):
        black_image_cols += pyr[i].shape[1]

    black_image = np.zeros((black_image_rows, black_image_cols))

    rows_count = 0
    cols_count = 0
    for level in range(0, levels):
        black_image[0:pyr[level].shape[0]:1, cols_count:cols_count+pyr[level].shape[1]:1] += pyr[level]
        rows_count += pyr[level].shape[0]
        cols_count += pyr[level].shape[1]
    return black_image


def display_pyramid(pyr, levels):
    pyramid_image = render_pyramid(pyr, levels)
    plt.imshow(pyramid_image, cmap='gray')
    plt.show()
    return


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    assert im1.shape == im2.shape
    # build Laplacian pyramids L1 and L2
    laplacian_pyr_L1, filter = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    laplacian_pyr_L2, filter = build_gaussian_pyramid(im2, max_levels, filter_size_im)

    # consteuct gaussian pyramid Gm (first convert to float64)
    mask_G_m = np.array(mask, dtype=np.float64)
    gaussian_pyr_Gm, filter2 = build_gaussian_pyramid(mask_G_m, max_levels, filter_size_mask)

    # construct laplacian pyramid L-out of the blending image for each level k:
    l_out = []
    for k in range(0, max_levels):
        l_out.append(np.multiply(gaussian_pyr_Gm[k], laplacian_pyr_L1[k]) + np.multiply((1-gaussian_pyr_Gm[k]),
                                                                                  laplacian_pyr_L2[k]))

    # Reconstruct the resulting blended image
    vector = np.ones(max_levels, dtype=np.int)
    image = laplacian_to_image(l_out, filter, vector)
    return image


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def blending_example1():
    jump = read_image(relpath("externals/jump.jpg"), 2)
    fire = read_image(relpath("externals/fire.jpg"), 2)
    mask_fire_jump = read_image(relpath("externals/jump_fire_mask.jpg"), 1)
    mask_fire_jump = mask_fire_jump.astype(bool)

    jump_over_fire_blue = pyramid_blending(jump[:,:,2], fire[:,:,2], mask_fire_jump, 3, 3, 3)
    jump_over_fire_green = pyramid_blending(jump[:,:,1], fire[:,:,1], mask_fire_jump, 3, 3, 3)
    jump_over_fire_red = pyramid_blending(jump[:,:,0], fire[:,:,0], mask_fire_jump, 3, 3, 3)

    jump_over_fire = np.zeros((fire.shape[0], fire.shape[1], 3))
    jump_over_fire[..., 0] = jump_over_fire_red
    jump_over_fire[..., 1] = jump_over_fire_green
    jump_over_fire[..., 2] = jump_over_fire_blue

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(jump)
    axarr[0,1].imshow(fire)
    axarr[1,0].imshow(mask_fire_jump, cmap='gray')
    axarr[1,1].imshow(jump_over_fire)

    plt.show()
    return fire, jump, mask_fire_jump, jump_over_fire


def blending_example2():
    throw = read_image(relpath("externals/throw.jpg"), 2)
    iphone = read_image(relpath("externals/iphone.jpg"), 2)
    mask_throw_iphone = read_image(relpath("externals/iphone_throw_mask.jpg"), 1)
    mask_throw_iphone = mask_throw_iphone.astype(bool)

    jump_over_fire_blue = pyramid_blending(throw[:,:,2], iphone[:,:,2], mask_throw_iphone, 3, 3, 3)
    jump_over_fire_green = pyramid_blending(throw[:,:,1], iphone[:,:,1], mask_throw_iphone, 3, 3, 3)
    jump_over_fire_red = pyramid_blending(throw[:,:,0], iphone[:,:,0], mask_throw_iphone, 3, 3, 3)

    throw_iphone = np.zeros((throw.shape[0], throw.shape[1], 3))
    throw_iphone[..., 0] = jump_over_fire_red
    throw_iphone[..., 1] = jump_over_fire_green
    throw_iphone[..., 2] = jump_over_fire_blue

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(throw)
    axarr[0,1].imshow(iphone)
    axarr[1,0].imshow(mask_throw_iphone, cmap='gray')
    axarr[1,1].imshow(throw_iphone)

    plt.show()


    return throw, iphone, mask_throw_iphone, throw_iphone
