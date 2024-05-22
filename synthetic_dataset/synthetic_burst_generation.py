# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import random
import cv2
import numpy as np
import torch.nn.functional as F
from synthetic_dataset.data_format_utils import torch_to_numpy, numpy_to_torch
import math

def random_crop(frames, crop_sz):
    """ Extract a random crop of size crop_sz from the input frames. If the crop_sz is larger than the input image size,
    then the largest possible crop of same aspect ratio as crop_sz will be extracted from frames, and upsampled to
    crop_sz.
    """
    if not isinstance(crop_sz, (tuple, list)):
        crop_sz = (crop_sz, crop_sz)
    crop_sz = torch.tensor(crop_sz).float()

    shape = frames.shape

    # Select scale_factor. Ensure the crop fits inside the image
    max_scale_factor = torch.tensor(shape[-2:]).float() / crop_sz
    max_scale_factor = max_scale_factor.min().item()

    if max_scale_factor < 1.0:
        scale_factor = max_scale_factor
    else:
        scale_factor = 1.0

    # Extract the crop
    orig_crop_sz = (crop_sz * scale_factor).floor()

    assert orig_crop_sz[-2] <= shape[-2] and orig_crop_sz[-1] <= shape[-1], 'Bug in crop size estimation!'

    r1 = random.randint(0, shape[-2] - orig_crop_sz[-2])
    c1 = random.randint(0, shape[-1] - orig_crop_sz[-1])

    r2 = r1 + orig_crop_sz[0].int().item()
    c2 = c1 + orig_crop_sz[1].int().item()

    frames_crop = frames[r1:r2, c1:c2]

    # Resize to crop_sz
    if scale_factor < 1.0:
        frames_crop = F.interpolate(frames_crop.unsqueeze(0), size=crop_sz.int().tolist(), mode='bilinear').squeeze(0)
    return frames_crop, r1, c1

def hr2burst(image, burst_size, downsample_factor=1, burst_transformation_params=None, interpolation_type='bilinear', gray_max=255.0):
    """ Generates a synthetic LR RAW burst from the input image. The input sRGB image is first converted to linear
    sensor space using an inverse camera pipeline. A LR burst is then generated by applying random
    transformations defined by burst_transformation_params to the input image, and downsampling it by the
    downsample_factor. The generated burst is then mosaicekd and corrputed by random noise.

    args:
        image - input sRGB image
        burst_size - Number of images in the output burst
        downsample_factor - Amount of downsampling of the input sRGB image to generate the LR image
        burst_transformation_params - Parameters of the affine transformation used to generate a burst from single image
        interpolation_type - interpolation operator used when performing affine transformations and downsampling
    """

    # Generate LR burst
    image_burst, flow_vectors = single2lrburst(image, burst_size=burst_size,
                                                   downsample_factor=downsample_factor,
                                                   transformation_params=burst_transformation_params,
                                                   interpolation_type=interpolation_type)

    # Add noise
    #shot_noise_level, read_noise_level = random_noise_levels()
    image_burst = image_burst / gray_max
    #image_burst = add_noise(image_burst, shot_noise_level, read_noise_level)
    image_burst = add_gaussian_noise(image_burst)
    image_burst = image_burst * gray_max
    # Clip saturated pixels.
    image_burst = image_burst.clamp(0.0, gray_max)

    #meta_info = {'shot_noise_level': shot_noise_level, 'read_noise_level': read_noise_level}
    return image_burst, image, flow_vectors

def hr2burst_full(image, translations, thetas, shear_factors, ar_factors, scale_factors, burst_size, downsample_factor=1, interpolation_type='bilinear', gray_max=255.0):
    """ Generates a synthetic LR RAW burst from the input image. The input sRGB image is first converted to linear
    sensor space using an inverse camera pipeline. A LR burst is then generated by applying random
    transformations defined by burst_transformation_params to the input image, and downsampling it by the
    downsample_factor. The generated burst is then mosaicekd and corrputed by random noise.

    args:
        image - input sRGB image
        burst_size - Number of images in the output burst
        downsample_factor - Amount of downsampling of the input sRGB image to generate the LR image
        burst_transformation_params - Parameters of the affine transformation used to generate a burst from single image
        interpolation_type - interpolation operator used when performing affine transformations and downsampling
    """

    # Generate LR burst
    image_burst, flow_vectors = single2lrburst_full(image, translations, thetas, shear_factors, ar_factors, scale_factors,
                                                burst_size=burst_size,
                                                   downsample_factor=downsample_factor,
                                                   interpolation_type=interpolation_type)

    # Add noise
    #shot_noise_level, read_noise_level = random_noise_levels()
    image_burst = image_burst / gray_max
    #image_burst = add_noise(image_burst, shot_noise_level, read_noise_level)
    image_burst = add_gaussian_noise(image_burst)
    image_burst = image_burst * gray_max
    # Clip saturated pixels.
    image_burst = image_burst.clamp(0.0, gray_max)

    #meta_info = {'shot_noise_level': shot_noise_level, 'read_noise_level': read_noise_level}
    return image_burst, image, flow_vectors


def random_noise_levels():
    """Generates random noise levels from a log-log linear distribution."""
    log_min_shot_noise = math.log(0.0001)
    log_max_shot_noise = math.log(0.012)
    log_shot_noise = random.uniform(log_min_shot_noise, log_max_shot_noise)
    shot_noise = math.exp(log_shot_noise)

    line = lambda x: 2.18 * x + 1.20
    log_read_noise = line(log_shot_noise) + random.gauss(mu=0.0, sigma=0.26)
    read_noise = math.exp(log_read_noise)
    return shot_noise, read_noise


def add_noise(image, shot_noise=0.01, read_noise=0.0005):
    """Adds random shot (proportional to image) and read (independent) noise."""
    variance = image * shot_noise + read_noise
    noise = torch.FloatTensor(image.shape).normal_().to(image.device)*variance.sqrt()
    return image + noise

def add_gaussian_noise(image, std=3.0/255.0):
    mean = torch.zeros_like(image)
    std = std * torch.ones_like(image)
    noise = torch.normal(mean, std)
    '''noise = torch.zeros_like(image)
    num_im, h, w = image.shape
    for i in range(num_im):
        mean = torch.zeros(h, w)
        std = std * torch.ones(h, w)
        noise[i, ...] = torch.normal(mean, std)'''
    return image + noise

def get_tmat(image_shape, translation, theta, shear_values, scale_factors):
    """ Generates a transformation matrix corresponding to the input transformation parameters """
    im_h, im_w = image_shape

    t_mat = np.identity(3)

    t_mat[0, 2] = translation[0]
    t_mat[1, 2] = translation[1]
    t_rot = cv2.getRotationMatrix2D((im_w * 0.5, im_h * 0.5), theta, 1.0)
    t_rot = np.concatenate((t_rot, np.array([0.0, 0.0, 1.0]).reshape(1, 3)))

    t_shear = np.array([[1.0, shear_values[0], -shear_values[0] * 0.5 * im_w],
                        [shear_values[1], 1.0, -shear_values[1] * 0.5 * im_h],
                        [0.0, 0.0, 1.0]])

    t_scale = np.array([[scale_factors[0], 0.0, 0.0],
                        [0.0, scale_factors[1], 0.0],
                        [0.0, 0.0, 1.0]])

    t_mat = t_scale @ t_rot @ t_shear @ t_mat

    t_mat = t_mat[:2, :]

    return t_mat


def single2lrburst(image, burst_size, downsample_factor=1, transformation_params=None,
                   interpolation_type='bilinear'):
    """ Generates a burst of size burst_size from the input image by applying random transformations defined by
    transformation_params, and downsampling the resulting burst by downsample_factor.

    args:
        image - input sRGB image
        burst_size - Number of images in the output burst
        downsample_factor - Amount of downsampling of the input sRGB image to generate the LR image
        transformation_params - Parameters of the affine transformation used to generate a burst from single image
        interpolation_type - interpolation operator used when performing affine transformations and downsampling
    """

    if interpolation_type == 'bilinear':
        interpolation = cv2.INTER_LINEAR
    elif interpolation_type == 'lanczos':
        interpolation = cv2.INTER_LANCZOS4
    else:
        raise ValueError

    normalize = False
    if isinstance(image, torch.Tensor):
        if image.max() < 2.0:
            image = image * 255.0
            normalize = True
        image = torch_to_numpy(image).astype(np.uint8)

    burst = []
    sample_pos_inv_all = []

    rvs, cvs = torch.meshgrid([torch.arange(0, image.shape[0]),
                               torch.arange(0, image.shape[1])])

    sample_grid = torch.stack((cvs, rvs, torch.ones_like(cvs)), dim=-1).float()
    translations = []
    thetas = []
    shear_factors = []
    ar_factors = []
    scale_factors = []

    for i in range(burst_size):
        if i == 0:
            # For base image, do not apply any random transformations. We only translate the image to center the
            # sampling grid
            shift = (downsample_factor / 2.0) - 0.5
            translation = (shift, shift)
            theta = 0.0
            shear_factor = (0.0, 0.0)
            scale_factor = (1.0, 1.0)
            ar_factor = 1.0
        else:
            # Sample random image transformation parameters
            max_translation = transformation_params.get('max_translation', 0.0)

            if max_translation <= 0.01:
                shift = (downsample_factor / 2.0) - 0.5
                translation = (shift, shift)
            else:
                translation = (random.uniform(-max_translation, max_translation),
                               random.uniform(-max_translation, max_translation))

            max_rotation = transformation_params.get('max_rotation', 0.0)
            theta = random.uniform(-max_rotation, max_rotation)

            max_shear = transformation_params.get('max_shear', 0.0)
            shear_x = random.uniform(-max_shear, max_shear)
            shear_y = random.uniform(-max_shear, max_shear)
            shear_factor = (shear_x, shear_y)

            max_ar_factor = transformation_params.get('max_ar_factor', 0.0)
            ar_factor = np.exp(random.uniform(-max_ar_factor, max_ar_factor))

            max_scale = transformation_params.get('max_scale', 0.0)
            scale_factor = np.exp(random.uniform(-max_scale, max_scale))

            scale_factor = (scale_factor, scale_factor * ar_factor)

        translations.append(translation)
        thetas.append(theta)
        shear_factors.append(shear_factor)
        ar_factors.append(ar_factor)
        scale_factors.append(scale_factor)

        output_sz = (image.shape[1], image.shape[0])

        # Generate a affine transformation matrix corresponding to the sampled parameters
        t_mat = get_tmat((image.shape[0], image.shape[1]), translation, theta, shear_factor, scale_factor)
        t_mat_tensor = torch.from_numpy(t_mat)

        # Apply the sampled affine transformation
        image_t = cv2.warpAffine(image, t_mat, output_sz, flags=interpolation,
                                 borderMode=cv2.BORDER_CONSTANT)

        t_mat_tensor_3x3 = torch.cat((t_mat_tensor.float(), torch.tensor([0.0, 0.0, 1.0]).view(1, 3)), dim=0)
        t_mat_tensor_inverse = t_mat_tensor_3x3.inverse()[:2, :].contiguous()

        sample_pos_inv = torch.mm(sample_grid.view(-1, 3), t_mat_tensor_inverse.t().float()).view(
            *sample_grid.shape[:2], -1)

        if transformation_params.get('border_crop') is not None:
            border_crop = transformation_params.get('border_crop')

            image_t = image_t[border_crop:-border_crop, border_crop:-border_crop]
            sample_pos_inv = sample_pos_inv[border_crop:-border_crop, border_crop:-border_crop]

        # Downsample the image
        image_t = cv2.resize(image_t, None, fx=1.0 / downsample_factor, fy=1.0 / downsample_factor,
                             interpolation=interpolation)
        sample_pos_inv = cv2.resize(sample_pos_inv.numpy(), None, fx=1.0 / downsample_factor,
                                    fy=1.0 / downsample_factor,
                                    interpolation=interpolation)

        sample_pos_inv = torch.from_numpy(sample_pos_inv)

        if normalize:
            image_t = numpy_to_torch(image_t).float() / 255.0
        else:
            image_t = numpy_to_torch(image_t).float()
        burst.append(image_t)
        sample_pos_inv_all.append(sample_pos_inv / downsample_factor)

    burst_images = torch.stack(burst)
    sample_pos_inv_all = torch.stack(sample_pos_inv_all)

    # Compute the flow vectors to go from the i'th burst image to the base image
    flow_vectors = sample_pos_inv_all - sample_pos_inv_all[:1, ...]


    return burst_images, flow_vectors

def single2lrburst_full(image, translations, thetas, shear_factors, ar_factors, scale_factors, burst_size, downsample_factor=1,
                   interpolation_type='bilinear'):
    """ Generates a burst of size burst_size from the input image by applying random transformations defined by
    transformation_params, and downsampling the resulting burst by downsample_factor.

    args:
        image - input sRGB image
        burst_size - Number of images in the output burst
        downsample_factor - Amount of downsampling of the input sRGB image to generate the LR image
        transformation_params - Parameters of the affine transformation used to generate a burst from single image
        interpolation_type - interpolation operator used when performing affine transformations and downsampling
    """

    if interpolation_type == 'bilinear':
        interpolation = cv2.INTER_LINEAR
    elif interpolation_type == 'lanczos':
        interpolation = cv2.INTER_LANCZOS4
    else:
        raise ValueError

    normalize = False
    if isinstance(image, torch.Tensor):
        if image.max() < 2.0:
            image = image * 255.0
            normalize = True
        image = torch_to_numpy(image).astype(np.uint8)

    burst = []
    sample_pos_inv_all = []

    rvs, cvs = torch.meshgrid([torch.arange(0, image.shape[0]),
                               torch.arange(0, image.shape[1])])

    sample_grid = torch.stack((cvs, rvs, torch.ones_like(cvs)), dim=-1).float()

    for i in range(burst_size):
        translation = translations[i]
        theta = thetas[i]
        shear_factor = shear_factors[i]
        ar_factor = ar_factors[i]
        scale_factor = scale_factors[i]
        output_sz = (image.shape[1], image.shape[0])

        # Generate a affine transformation matrix corresponding to the sampled parameters
        t_mat = get_tmat((image.shape[0], image.shape[1]), translation, theta, shear_factor, scale_factor)
        t_mat_tensor = torch.from_numpy(t_mat)

        # Apply the sampled affine transformation
        image_t = cv2.warpAffine(image, t_mat, output_sz, flags=interpolation,
                                 borderMode=cv2.BORDER_CONSTANT)

        t_mat_tensor_3x3 = torch.cat((t_mat_tensor.float(), torch.tensor([0.0, 0.0, 1.0]).view(1, 3)), dim=0)
        t_mat_tensor_inverse = t_mat_tensor_3x3.inverse()[:2, :].contiguous()

        sample_pos_inv = torch.mm(sample_grid.view(-1, 3), t_mat_tensor_inverse.t().float()).view(
            *sample_grid.shape[:2], -1)

        # Downsample the image
        image_t = cv2.resize(image_t, None, fx=1.0 / downsample_factor, fy=1.0 / downsample_factor,
                             interpolation=interpolation)
        sample_pos_inv = cv2.resize(sample_pos_inv.numpy(), None, fx=1.0 / downsample_factor,
                                    fy=1.0 / downsample_factor,
                                    interpolation=interpolation)

        sample_pos_inv = torch.from_numpy(sample_pos_inv)

        if normalize:
            image_t = numpy_to_torch(image_t).float() / 255.0
        else:
            image_t = numpy_to_torch(image_t).float()
        burst.append(image_t)
        sample_pos_inv_all.append(sample_pos_inv / downsample_factor)

    burst_images = torch.stack(burst)
    sample_pos_inv_all = torch.stack(sample_pos_inv_all)

    # Compute the flow vectors to go from the i'th burst image to the base image
    flow_vectors = sample_pos_inv_all - sample_pos_inv_all[:1, ...]

    return burst_images, flow_vectors
