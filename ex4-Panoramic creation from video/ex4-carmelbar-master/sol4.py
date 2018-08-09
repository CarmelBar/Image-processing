
# **************** Imports *******************
from random import sample
from shutil import *

import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from scipy.ndimage import label, center_of_mass
from scipy.ndimage.filters import convolve
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure

from sol4_utils import *
import os as os


# ***************** Program *******************
def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    # get the I_x and I_y derivatives of the image using the filters [1,0,-1], [1,0,-1]^T
    derivative_filter = np.array([[1, 0, -1]])
    i_x_derivative = convolve(im, derivative_filter)
    i_y_derivative = convolve(im, derivative_filter.transpose())

    # create M matrix for each pixel
    height, width = im.shape
    M = np.ndarray((height, width, 2, 2))

    # blur the images I_x^2, I_y^2, I_x*I_y
    M[:, :, 0, 0] = blur_spatial(i_x_derivative ** 2, 3)
    M[:, :, 1, 1] = blur_spatial(i_y_derivative ** 2, 3)
    calc = blur_spatial(i_x_derivative * i_y_derivative, 3)
    M[:, :, 0, 1] = calc
    M[:, :, 1, 0] = calc

    # calculate determinant, trace, r for each pixel
    det = np.linalg.det(M)
    trace = M[:, :, 0, 0] + M[:, :, 1, 1]
    r = det - 0.04 * (trace ** 2)

    # take only the points which are max
    x, y = np.where(non_maximum_suppression(r))

    # create corner list to return
    corner_list = np.ndarray((x.shape[0], 2))
    corner_list[:, 1] = x
    corner_list[:, 0] = y

    return corner_list


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    # prepare variables
    pos = pos /4
    size = pos.shape[0]
    sample_size = (desc_rad * 2) + 1

    # make grid in size sample_size**2 for all pos
    grid = np.meshgrid(np.arange(- desc_rad,  desc_rad + 1).astype(np.float64),
                           np.arange( - desc_rad,  desc_rad + 1).astype(np.float64))
    grid_row = np.tile(grid[0], (size, 1)).reshape(size, sample_size, sample_size)
    grid_row += pos[:, 1].reshape(size, 1, 1)
    grid_col = np.tile(grid[1], (size, 1)).reshape(size, sample_size, sample_size)
    grid_col += pos[:, 0].reshape(size, 1, 1)
    grid = np.array([grid_row, grid_col])

    # take the grid out from the image
    descriptor = ndi.map_coordinates(im, grid, order=1, prefilter='false')

    # create normalize array
    mean = np.mean(descriptor, axis=(1, 2))
    descriptor -= mean.reshape(size, 1, 1)
    normalaized = np.linalg.norm(descriptor, axis=(1, 2), keepdims=True)

    #verify there is no division in zero
    zeros_in_normalized = np.where(normalaized == 0)
    descriptor[zeros_in_normalized] = 0.0
    normalaized[zeros_in_normalized] += 1

    return descriptor / normalaized


def find_features(pyr):
    """
  Detects and extracts feature points from a pyramid.
  :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
  :return: A list containing:
              1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                 These coordinates are provided at the pyramid level pyr[0].
              2) A feature descriptor array with shape (N,K,K)
  """
    harris_points = spread_out_corners(pyr[0], 7, 7, 3)
    descriptors = sample_descriptor(pyr[2], harris_points.copy(), 3)
    return harris_points, descriptors


def match_feature(feature, desc, min_score):
    radius = desc.shape[1]
    desc = desc.reshape((desc.shape[0], radius * radius))
    feature = feature.flatten()

    product = desc[:, :].dot(feature)
    if np.max(product) >= min_score:
        max_index = np.argmax(product)
        remaining_product = np.concatenate((product[0:max_index], product[max_index + 1:]))
        second_max_index = np.argmax(remaining_product)
        if product[second_max_index] >= min_score:
            if second_max_index > max_index - 1:
                second_max_index += 1
            return [max_index, second_max_index]
        return [max_index, -1]
    return [-1, -1]


def match_features(desc1, desc2, min_score):
    """
  Return indices of matching descriptors.
  :param desc1: A feature descriptor array with shape (N1,K,K).
  :param desc2: A feature descriptor array with shape (N2,K,K).
  :param min_score: Minimal match score.
  :return: A list containing:
              1) An array with shape (M,) and dtype int of matching indices in desc1.
              2) An array with shape (M,) and dtype int of matching indices in desc2.
  """
    N1 = desc1.shape[0]
    N2 = desc2.shape[0]

    matches_part_1 = np.array([-1, -1] * N1).reshape(N1, 2)
    matches_part_2 = list()

    for i in range(0, N1):
        matches_part_1[i] = match_feature(desc1[i], desc2, min_score)
    for j in range(0, N2):
        max, second_max = match_feature(desc2[j], desc1, min_score)

        # Verify that the feature points matches all 3 conditions
        if max != -1 and j in matches_part_1[max]:
            matches_part_2.append([max, j])
        elif second_max != -1 and j in matches_part_1[second_max]:
            matches_part_2.append([second_max, j])
    matches_part_2 = np.array(matches_part_2)
    return matches_part_2[:, 0], matches_part_2[:, 1]


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    # create a 3-D array for pos by transforming [x, y] to [x, y, 1]
    ones = np.ones([pos1.shape[0],1])
    three_d_array = np.concatenate((pos1, ones), axis=1).T

    # apply the homography on the 3D array
    pos_H_multiplication = np.dot(H12, three_d_array)

    # normalize the array
    pos_H_multiplication /= pos_H_multiplication[2, :]
    return pos_H_multiplication[:2, :].T


def ransac_iteration(num_of_points, points1, points2, inlier_tol, translation_only):
    """
    The function preforms the ransac process for one iteration in order to find
    The points with the max num of inliers
    """
    # pick a random set of 2 point matches
    j = sample(range(0, num_of_points), 2)
    p_1_j = np.take(points1, j, axis=0)
    p_2_j = np.take(points2, j, axis=0)

    # compute homography H_1_2
    H_1_2 = estimate_rigid_transform(p_1_j, p_2_j, translation_only)

    # use H_1_2 to transform the set p_1 in image 1 to the transformed set p_2
    p_1_j_transformed = apply_homography(points1, H_1_2)

    # compute euclidean distance Ej = ||P 2;j - P 2;j||**2
    E_j = np.linalg.norm(p_1_j_transformed - points2, axis=1) ** 2
    inliers = E_j < inlier_tol
    return inliers


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
              1) A 3x3 normalized homography matrix.
              2) An Array with shape (S,) where S is the number of inliers,
                  containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    num_of_points = points1.shape[0]
    largest_inliers_count = 0
    H_1_2 = None
    largest_inliers_set = None
    for i in range(0, num_iter):
        inliers = ransac_iteration(num_of_points, points1, points2, inlier_tol, translation_only)
        # check if it is the largest set, and if so- switch it
        if np.sum(inliers) > largest_inliers_count:
            largest_inliers_set = np.where(inliers)[0]
            largest_inliers_count = np.sum(inliers)

    inliners_1 = points1[largest_inliers_set, :]
    inliners_2 = points2[largest_inliers_set, :]
    H_1_2 = estimate_rigid_transform(inliners_1, inliners_2, translation_only)
    return H_1_2, largest_inliers_set


def display_matches(im1, im2, pos1, pos2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    # measure image size
    image_display_height = max(im1.shape[0], im2.shape[0])
    image_display_width = im1.shape[1] + im2.shape[1]

    # display both images side by side
    new_image = np.ndarray((image_display_height, image_display_width))
    new_image[0:im1.shape[0], 0:im1.shape[1]] = im1
    new_image[0:im2.shape[0], im1.shape[1]:im1.shape[1] + im2.shape[1]] = im2

    pos2[:, 0] += im1.shape[1] #change pos2 pixels in the width section so it will match the new image

    # create red dots to mark the pos (1 and 2)
    plt.scatter(pos2[:, 0], pos2[:, 1], s=8, color='r', alpha=0.5)
    plt.scatter(pos1[:, 0], pos1[:, 1], s=8, color='r', alpha=0.5)

    # mark all the inliners in pos
    inliners_1 = np.take(pos1, inliers, axis=0)
    inliners_2 = np.take(pos2, inliers, axis=0)

    # mark all the outliners in pos
    outliers = [i for i in range(pos1.shape[0]) if i not in inliers]
    outlienrs_1 = np.take(pos1, outliers, axis=0)
    outlienrs_2 = np.take(pos2, outliers, axis=0)

    for i in range(outlienrs_1.shape[0]):
        # create blue lines from for all pos that are outliners
        plt.plot([outlienrs_1[i, 0], outlienrs_2[i, 0]], [outlienrs_1[i, 1], outlienrs_2[i, 1]], c='b', linewidth=0.5)

    for j in range(inliners_1.shape[0]):
        # create yellow lines from for all pos that are inliners
        plt.plot([inliners_1[j, 0], inliners_2[j, 0]], [inliners_1[j, 1], inliners_2[j, 1]], c='y', linewidth=0.5)

    #display image
    plt.imshow(new_image, cmap=plt.gray())
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
    matrices where H_successive[i] is a homography which transforms points
    from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
    accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
    where H2m[i] transforms points from coordinate system i to coordinate system m
    """

    h_2_m = [np.eye(3)] * (len(H_succesive) + 1)
    for i in range(m-1, -1, -1):
        h_2_m[i] = np.dot(H_succesive[i], h_2_m[i+1])
        h_2_m[i] /= h_2_m[i][2,2]
    for j in range(m + 1, (len(H_succesive) + 1)):
        h_2_m[j] = np.dot(h_2_m[j-1], np.linalg.inv(H_succesive[j-1]))
        h_2_m[j] /= h_2_m[j][2,2]
    return h_2_m


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
    and the second row is the [x,y] of the bottom right corner
    """
    x = np.array([[0, w-1, 0, w-1], [0, 0, h-1, h-1], [1, 1, 1, 1]])

    image_box_homography = homography.dot(x)
    image_box_homography[0, :] /= image_box_homography[2, :]
    image_box_homography[1, :] /= image_box_homography[2, :]
    image_box_homography = image_box_homography[:2, :]

    top_left = np.floor(np.min(image_box_homography, axis=1)).astype(np.int)
    bottom_right = np.ceil(np.max(image_box_homography, axis=1)).astype(np.int)

    return np.vstack((top_left.astype(np.int), bottom_right.astype(np.int)))


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    h, w = image.shape
    top_left_corner, bottom_right_corner = compute_bounding_box(homography, w, h)
    new_box = np.meshgrid(range(top_left_corner[0], bottom_right_corner[0] +1), range(top_left_corner[1], bottom_right_corner[1] + 1))
    box_height = new_box[0].shape[0]
    box_width = new_box[1].shape[1]

    coordinates = np.dstack((new_box[0], new_box[1]))
    coordinates = coordinates.reshape((box_height * box_width, 2))

    homography = np.linalg.inv(homography)
    coordinates_after_homography = np.array((np.fliplr(apply_homography(coordinates, homography)).T)).reshape((2, box_height, box_width))
    image_new_coords = ndi.map_coordinates(image, coordinates_after_homography, order=1, prefilter=False)
    return image_new_coords

def warp_image(image, homography):
    """
  Warps an RGB image with a given homography.
  :param image: an RGB image.
  :param homography: homograhpy.
  :return: A warped image.
  """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
        last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)



def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2,:2] = rotation
    H[:2, 2] = translation
    return H



def non_maximum_suppression(image):
    """
  Finds local maximas of an image.
  :param image: A 2D array representing an image.
  :return: A boolean array with the same shape as the input image, where True indicates local maximum.
  """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.row_stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
  Splits the image im to m by n rectangles and uses harris_corner_detector on each.
  :param im: A 2D array representing an image.
  :param m: Vertical number of rectangles.
  :param n: Horizontal number of rectangles.
  :param radius: Minimal distance of corner points from the boundary of the image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
  Generates panorama from a set of images.
  """

    def __init__(self, data_dir, file_prefix, num_images):
        """
    The naming convention for a sequence of images is file_prefixN.jpg,
    where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    """
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            misc.imsave('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()
