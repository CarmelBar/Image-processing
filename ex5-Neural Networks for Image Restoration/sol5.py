
# **************** Imports *******************

import datetime
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
import scipy.misc as misc

from sol5_utils import *
import numpy as np
import random
from skimage import color

from keras.layers import Convolution2D, merge, Input, Activation
from keras.models import Model
from keras.optimizers import Adam

# **************** Constants *******************

VALID_REPRESENTATION = [1, 2]
NORMALIZE_VAL = 255
DERIVATIVE_MATRIX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
PATCH_SIZE_DENOISE = (24, 24)
PATCH_SIZE_DEBLUR = (16, 16)
NUM_OF_CHANNELS_DENOISE = 48
NUM_OF_CHANNELS_DEBLUR = 32
MIN_SIGMA = 0.0
MAX_SIGMA = 0.2
IMAGES_IN_BATCH = 100
SAMPLES_PER_EPOCH = 10000
NUM_OF_EPOCHS = 5
SAMPLES = 1000

# **************** Program *******************

def read_image(filename, representation = 1):
    if representation not in VALID_REPRESENTATION:
        return
    else:
        image = misc.imread(filename).astype(np.float64)
        image = np.divide(image, NORMALIZE_VAL)

        if representation == VALID_REPRESENTATION[0]:
            image = color.rgb2grey(image)
            return image
        return image


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    images_cache = {}
    # create arrays in the given size
    source_batch = np.zeros((batch_size, 1, crop_size[0], crop_size[1]), dtype=np.float64)
    target_batch = np.zeros((batch_size, 1, crop_size[1], crop_size[1]), dtype=np.float64)

    while True:

        # randomly picking from the filenames
        filenames = np.asarray(filenames)
        images_choice = np.random.choice(filenames, batch_size)

        #  and insert to source_batch and target_batch
        for i in range(batch_size):
            # read the images, or get the image from the dictionary
            if images_choice[i] not in images_cache:
                images_cache[images_choice[i]]= read_image(images_choice[i])
            image = images_cache[images_choice[i]]

            # crop the image randomly
            start_crop_first = random.randint(0, image.shape[0]-crop_size[0])
            start_crop_second = random.randint(0, image.shape[1]-crop_size[1])
            image_cropped = image[start_crop_first: start_crop_first+crop_size[0], start_crop_second: start_crop_second+crop_size[1]]

            source_batch[i] = np.subtract(image_cropped, 0.5)
            curropted = corruption_func(image_cropped)
            target_batch[i] = np.subtract(curropted, 0.5)

        yield source_batch, target_batch


def resblock(input_tensor, num_channels):
    a = Convolution2D(num_channels, 3, 3, border_mode='same')(input_tensor)
    b = Activation('relu')(a)
    c = Convolution2D(num_channels, 3, 3, border_mode='same')(b)
    d = merge([input_tensor, c], mode='sum')
    return d


def build_nn_model(height, width, num_channels, num_res_blocks):
    input = Input(shape=(1, height, width))
    input_convolution = Convolution2D(num_channels, 3, 3, border_mode='same')(input)
    relu = Activation('relu')(input_convolution)
    b = relu
    for i in range(0, num_res_blocks):
        b = resblock(b, num_channels)
    merged = merge([b, relu])
    c = Convolution2D(1, 3, 3, border_mode='same')(merged)
    model = Model(input = input, output = c)
    return model


def train_model(model, images, corruption_func, batch_size, samples_per_epoch, num_epochs, num_valid_samples):
    # divide images to training and validation set
    precent_to_train = int(round(0.8*len(images)))
    image_train = images[0:precent_to_train]
    image_valid = images[precent_to_train:-1]

    # generate from each set a data set with given batch size and corruption function
    train_set = load_dataset(image_train, batch_size, corruption_func, model.input_shape[-2:])
    valid_set = load_dataset(image_valid, batch_size, corruption_func, model.input_shape[-2:])

    adam = Adam(beta_2=0.999)
    model.compile(loss='mean_squared_error', optimizer=adam)
    model.fit_generator(train_set,
                        samples_per_epoch=samples_per_epoch, nb_epoch=num_epochs,
                        validation_data=valid_set, nb_val_samples=num_valid_samples)


def restore_image(corrupted_image, base_model):
    # create a new model that fits the size of the input image, and has the same weight as the
    # given base model
    image = corrupted_image
    height, width = corrupted_image.shape[0], corrupted_image.shape[1]
    image = image.reshape((1,1) + (height, width))

    a = Input(shape=(1,) + (height, width))
    b = base_model(a)
    new_model = Model(input=a, output=b)

    # use predict to restore the image
    new_model = new_model.predict(np.subtract(image, 0.5))
    new_model = new_model.reshape((height, width))
    new_model = np.add(new_model, 0.5)

    # clip the results to [0,1] range
    clipped_image = np.clip(new_model, 0, 1).astype(np.float64)
    return clipped_image


def add_gaussian_noise(image, min_sigma, max_sigma):
    # sample value of sigma in range [min_sigma, max_sigma]
    sigma = random.uniform(min_sigma, max_sigma)
    normal = np.random.normal(0, sigma, image.shape)

    # adding to every pixel of the input image a zero-mean gaussian var with standard deviation equal to sigma.
    noised_image = np.add(image, normal)

    # the value should be rounded to the nearest fraction i/255
    noised_image = np.divide(np.round(np.multiply(noised_image, 255)), 255)

    # and clipped to [0, 1]
    noised_image = np.clip(noised_image, 0, 1)
    return noised_image


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    # get the image paths list for training
    def curroption_function(image):
        return add_gaussian_noise(image, MIN_SIGMA, MAX_SIGMA)

    image_paths = images_for_denoising()
    model = build_nn_model(PATCH_SIZE_DENOISE[0], PATCH_SIZE_DENOISE[1], NUM_OF_CHANNELS_DENOISE, num_res_blocks)

    IMAGES_IN_BATCH, SAMPLES_PER_EPOCH, NUM_OF_EPOCHS, SAMPLES = 100, 10000, 5, 1000
    if quick_mode:
        IMAGES_IN_BATCH, SAMPLES_PER_EPOCH, NUM_OF_EPOCHS, SAMPLES = 10, 30, 2, 30


    train_model(model, image_paths, curroption_function, IMAGES_IN_BATCH, SAMPLES_PER_EPOCH, NUM_OF_EPOCHS, SAMPLES)
    return model


def add_motion_blur(image, kernel_size, angle):
    kernel = motion_blur_kernel(kernel_size, angle)
    blurred_image = convolve(image, kernel, mode='mirror') # todo- mirror
    return blurred_image


def random_motion_blur(image, list_of_kernel_sizes):
    angle = np.random.randint(0, np.pi)
    kernel = np.random.choice(list_of_kernel_sizes, 1)
    blurred_image = add_motion_blur(image, kernel[0], angle)
    return blurred_image


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    # get the image paths list for training
    image_paths = images_for_deblurring()

    def curroption_function(image):
        return random_motion_blur(image, [7])

    model = build_nn_model(PATCH_SIZE_DEBLUR[0], PATCH_SIZE_DEBLUR[1], NUM_OF_CHANNELS_DEBLUR, num_res_blocks)

    IMAGES_IN_BATCH, SAMPLES_PER_EPOCH, NUM_OF_EPOCHS, SAMPLES = 100, 10000, 10, 1000
    if quick_mode:
        IMAGES_IN_BATCH, SAMPLES_PER_EPOCH, NUM_OF_EPOCHS, SAMPLES = 10, 30, 2, 30

    train_model(model, image_paths, curroption_function, IMAGES_IN_BATCH, SAMPLES_PER_EPOCH, NUM_OF_EPOCHS, SAMPLES)

    return model