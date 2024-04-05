""" CS4243 Lab 1: Template Matching
"""

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import math

##### Part 1: Image Preprossessing #####

def rgb2gray(img):
    """
    5 points
    Convert a colour image greyscale
    Use (R,G,B)=(0.299, 0.587, 0.114) as the weights for red, green and blue channels respectively
    :param img: numpy.ndarray (dtype: np.uint8)
    :return img_gray: numpy.ndarray (dtype:np.uint8)
    """
    if len(img.shape) != 3:
        print('RGB Image should have 3 channels')
        return
    
    """ Your code starts here """
    weights = np.array([0.299, 0.587, 0.114])
    img_gray = np.dot(img, weights)
    img_gray = img_gray.astype(np.uint8)
    """ Your code ends here """
    return img_gray

def gray2grad(img):
    """
    5 points
    Estimate the gradient map from the grayscale images by cross-correlating with Sobel filters (horizontal and vertical gradients) and Sobel-like filters (gradients oriented at 45 and 135 degrees)
    The coefficients of Sobel filters are provided in the code below.
    :param img: numpy.ndarray
    :return img_grad_h: horizontal gradient map. numpy.ndarray
    :return img_grad_v: vertical gradient map. numpy.ndarray
    :return img_grad_d1: diagonal gradient map 1. numpy.ndarray
    :return img_grad_d2: diagonal gradient map 2. numpy.ndarray
    """
    sobelh = np.array([[-1, 0, 1], 
                       [-2, 0, 2], 
                       [-1, 0, 1]], dtype = float)
    sobelv = np.array([[-1, -2, -1], 
                       [0, 0, 0], 
                       [1, 2, 1]], dtype = float)
    sobeld1 = np.array([[-2, -1, 0],
                        [-1, 0, 1],
                        [0,  1, 2]], dtype = float)
    sobeld2 = np.array([[0, -1, -2],
                        [1, 0, -1],
                        [2, 1, 0]], dtype = float)
    

    """ Your code starts here """
    rows = img.shape[0]
    cols = img.shape[1]
    img_grad_h, img_grad_v, img_grad_d1, img_grad_d2 = np.zeros((rows, cols)), np.zeros((rows, cols)), np.zeros((rows, cols)), np.zeros((rows, cols))
    img = pad_zeros(img, 1, 1, 1, 1)
    for x in range(rows):
        for y in range(cols):
            img_grad_h[x, y] = (sobelh * img[x:x+3, y:y+3]).sum()
            img_grad_v[x, y] = (sobelv * img[x:x+3, y:y+3]).sum()
            img_grad_d1[x, y] = (sobeld1 * img[x:x+3, y:y+3]).sum()
            img_grad_d2[x, y] = (sobeld2 * img[x:x+3, y:y+3]).sum()
    """ Your code ends here """
    return img_grad_h, img_grad_v, img_grad_d1, img_grad_d2

def pad_zeros(img, pad_height_bef, pad_height_aft, pad_width_bef, pad_width_aft):
    """
    5 points
    Add a border of zeros around the input images so that the output size will match the input size after a convolution or cross-correlation operation.
    e.g., given matrix [[1]] with pad_height_bef=1, pad_height_aft=2, pad_width_bef=3 and pad_width_aft=4, obtains:
    [[0 0 0 0 0 0 0 0]
    [0 0 0 1 0 0 0 0]
    [0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0]]
    :param img: numpy.ndarray
    :param pad_height_bef: int
    :param pad_height_aft: int
    :param pad_width_bef: int
    :param pad_width_aft: int
    :return img_pad: numpy.ndarray. dtype is the same as the input img. 
    """
    height, width = img.shape[:2]
    new_height, new_width = (height + pad_height_bef + pad_height_aft), (width + pad_width_bef + pad_width_aft)
    img_pad = np.zeros((new_height, new_width)) if len(img.shape) == 2 else np.zeros((new_height, new_width, img.shape[2]))

    """ Your code starts here """
    dims = list(img.shape)
    dims[0] += pad_height_bef + pad_height_aft
    dims[1] += pad_width_bef + pad_width_aft

    img_pad = np.zeros(dims, dtype=img.dtype)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            img_pad[x + pad_height_bef, y + pad_width_bef] = img[x, y]
    """ Your code ends here """
    return img_pad

##### Part 2: Normalized Cross Correlation #####
def normalized_cross_correlation(img, template):
    """
    10 points.
    Implement the cross-correlation operation in a naive 4, 5 or 6 nested for-loops. 
    The loops should at least include the height and width of the output and height and width of the template.
    When it is 5 or 6 loops, the channel of the output and template may be included.
    Output size: (Ho, Wo)
    :param img: numpy.ndarray.
    :param template: numpy.ndarray.
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    """ Your code starts here """      
    img = img.astype(np.float32)
    template = template.astype(np.float32)

    C = 1
    if img.ndim == 3:
        C = img.shape[2]

    sum_square_template = 0
    for i in range(Hk):
        for j in range(Wk):
            for k in range(3):
                sum_square_template += template[i, j, k] ** 2

    response = np.zeros((Ho, Wo), dtype=np.float32)
    for x in range(Ho):
        for y in range(Wo):
            sum_product = 0.0
            sum_square_img = 0.0
            for i in range(Hk):
                for j in range(Wk):
                    for k in range(C):
                        img_val = img[x+i, y+j, k] if C > 1 else img[x+i, y+j]
                        template_val = template[i, j, k] if C > 1 else template[i, j]
                        sum_product += img_val * template_val
                        sum_square_img += img_val ** 2
            
            denom = np.sqrt(sum_square_img * sum_square_template)
            response[x, y] = sum_product / denom if denom != 0 else 0
    """ Your code ends here """
    return response

def normalized_cross_correlation_fast(img, template):
    """
    10 points.
    Implement the cross correlation with 3 nested for-loops. 
    The for-loop over the template is replaced with the element-wise multiplication between the kernel and the image regions.
    Output size: (Ho, Wo)
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    """ Your code starts here """
    img = img.astype(np.float32)
    template = template.astype(np.float32)

    C = 1
    if img.ndim == 3:
        C = img.shape[2]

    response = np.zeros((Ho, Wo), dtype=np.float32)
    template_denom = np.sqrt(np.sum(template ** 2))
    for x in range(Ho):
        for y in range(Wo):
            img_region = img[x:x+Hk, y:y+Wk]
            img_denom = np.sqrt(np.sum(img_region ** 2))
            numer = np.sum(img_region * template)
            response[x, y] = numer / (template_denom * img_denom)
    """ Your code ends here """
    return response

def normalized_cross_correlation_matrix(img, template):
    """
    10 points.
    Converts cross-correlation into a matrix multiplication operation to leverage optimized matrix operations.
    Please check the detailed instructions in the pdf file.
    Output size: (Ho, Wo)
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    """ Your code starts here """
    img = img.astype(np.float32)
    template = template.astype(np.float32)

    C = 1
    if img.ndim == 3:
        C = img.shape[2]

    reshaped_img = np.zeros((Ho * Wo, C * Hk * Wk), dtype=np.float32)
    for x in range(Ho):
        for y in range(Wo):
            for z in range(C):
                row = x * Wo + y
                col = z * Hk * Wk
                if C > 1:
                    reshaped_window = np.reshape(img[x:x+Hk, y:y+Wk, z], (1, Hk * Wk))
                else: 
                    reshaped_window = np.reshape(img[x:x+Hk, y:y+Wk], (1, Hk * Wk))
                reshaped_img[row, col:col+(Hk*Wk)] = reshaped_window
    
    reshaped_template = np.zeros((C * Hk * Wk, 1), dtype=np.float32)
    ones_template = np.ones((C * Hk * Wk, 1), dtype=np.float32)
    for x in range(Hk):
        for y in range(Wk):
            for z in range(C):
                row = z * Hk * Wk + x * Wk + y
                reshaped_template[row, 0] = template[x, y, z] if C > 1 else template[x, y]

    template_denom = np.sqrt(np.sum(template ** 2))
    img_denoms = np.sqrt(np.matmul(reshaped_img ** 2, ones_template))
    response_map = np.matmul(reshaped_img, reshaped_template) / img_denoms / template_denom
    response = np.reshape(response_map, (Ho, Wo))
    """ Your code ends here """
    return response


##### Part 3: Non-maximum Suppression #####

def non_max_suppression(response, suppress_range, threshold=None):
    """
    10 points
    Implement the non-maximum suppression for translation symmetry detection
    The general approach for non-maximum suppression is as follows:
	1. Set a threshold τ; values in X<τ will not be considered.  Set X<τ to 0.  
    2. While there are non-zero values in X
        a. Find the global maximum in X and record the coordinates as a local maximum.
        b. Set a small window of size h×w points centered on the found maximum to 0.
	3. Return all recorded coordinates as the local maximum.
    :param response: numpy.ndarray, output from the normalized cross correlation
    :param suppress_range: a tuple of two ints (H_range, W_range). 
                           the points around the local maximum point within this range are set as 0. In this case, there are 2*H_range*2*W_range points including the local maxima are set to 0
    :param threshold: int, points with value less than the threshold are set to 0
    :return res: a sparse response map which has the same shape as response
    """
    
    """ Your code starts here """
    H_range, W_range = suppress_range
    if threshold is not None:
        response[response < threshold] = 0 
    res = np.zeros_like(response) 
    temp_response = response.copy()

    while np.any(temp_response > 0): 
        max_index = np.argmax(temp_response)
        max_index_x = max_index // temp_response.shape[1]
        max_index_y = max_index - max_index_x * temp_response.shape[1]
        max_idx = (max_index_x, max_index_y)
        
        res[max_idx] = temp_response[max_idx]
        
        x_start = max(max_idx[0] - H_range, 0)
        x_end = min(max_idx[0] + H_range + 1, temp_response.shape[0])
        y_start = max(max_idx[1] - W_range, 0)
        y_end = min(max_idx[1] + W_range + 1, temp_response.shape[1])
        temp_response[x_start:x_end, y_start:y_end] = 0
    """ Your code ends here """
    return res

##### Part 4: Question And Answer #####
    
def normalized_cross_correlation_ms(img, template):
    """
    10 points
    Please implement mean-subtracted cross correlation which corresponds to OpenCV TM_CCOEFF_NORMED.
    For simplicty, use the "fast" version.
    Output size: (Ho, Wo)
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    """ Your code starts here """
    C = 1
    if img.ndim == 3:
        C = img.shape[2]

    img = img.astype(np.float32)
    template = template.astype(np.float32)

    f_mean_subtracted = template - np.mean(template, axis=(0, 1), keepdims=True)
    f_norm = np.sqrt(np.sum(f_mean_subtracted ** 2))

    response = np.zeros((Ho, Wo), dtype=np.float32)

    for i in range(Ho):
        for j in range(Wo):
            window = img[i:i + Hk, j:j + Wk]

            if C > 1:
                w_mean_subtracted = window - np.mean(window, axis=(0, 1), keepdims=True)
                w_norm = np.sqrt(np.sum(w_mean_subtracted ** 2))
                sum_product = np.sum(w_mean_subtracted * f_mean_subtracted)
            else:
                w_mean_subtracted = window - np.mean(window)
                w_norm = np.sqrt(np.sum(w_mean_subtracted ** 2))
                sum_product = np.sum(w_mean_subtracted * f_mean_subtracted)

            norm = w_norm * f_norm
            response[i, j] = sum_product / norm if norm != 0 else 0
    """ Your code ends here """
    return response

"""Helper functions: You should not have to touch the following functions.
"""
def read_img(filename):
    '''
    Read HxWxC image from the given filename
    :return img: numpy.ndarray, size (H, W, C) for RGB. The value is between [0, 255].
    '''
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def show_imgs(imgs, titles=None):
    '''
    Display a list of images in the notebook cell.
    :param imgs: a list of images or a single image
    '''
    if isinstance(imgs, list) and len(imgs) != 1:
        n = len(imgs)
        fig, axs = plt.subplots(1, n, figsize=(15,15))
        for i in range(n):
            axs[i].imshow(imgs[i], cmap='gray' if len(imgs[i].shape) == 2 else None)
            if titles is not None:
                axs[i].set_title(titles[i])
    else:
        img = imgs[0] if (isinstance(imgs, list) and len(imgs) == 1) else imgs
        plt.figure()
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)

def show_img_with_points(response, img_ori=None):
    '''
    Draw small red rectangles of size defined by rec_shape around the non-zero points in the image.
    Display the rectangles and the image with rectangles in the notebook cell.
    :param response: numpy.ndarray. The input response should be a very sparse image with most of points as 0.
                     The response map is from the non-maximum suppression.
    :param img_ori: numpy.ndarray. The original image where response is computed from
    :param rec_shape: a tuple of 2 ints. The size of the red rectangles.
    '''
    response = response.copy()
    if img_ori is not None:
        img_ori = img_ori.copy()

    xs, ys = response.nonzero()
    for x, y in zip(xs, ys):
        response = cv2.circle(response, (y, x), radius=0, color=(255, 0, 0), thickness=5)
        if img_ori is not None:
            img_ori = cv2.circle(img_ori, (y, x), radius=0, color=(255, 0, 0), thickness=5)
        
    if img_ori is not None:
        show_imgs([response, img_ori])
    else:
        show_imgs(response)


