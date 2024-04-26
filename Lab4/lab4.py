# -*- coding: utf-8 -*-
"""NUS CS4243 Lab4.
Name: 
Student Number: 
"""

import numpy as np
import cv2
from skimage import filters
from scipy.ndimage import gaussian_filter

# ================ TASK 1.1 ================ #

def calcOpticalFlowHS(prevImg: np.array, nextImg: np.array, param_lambda: float, param_delta: float) -> np.array:
    """Computes a dense optical flow using the Horn–Schunck algorithm.
    
    The function finds an optical flow for each prevImg pixel using the Horn and Schunck algorithm [Horn81] so that: 
    
        prevImg(y,x) ~ nextImg(y + flow(y,x,2), x + flow(y,x,1)).


    Args:
        prevImg (np.array): First 8-bit single-channel input image.
        nextImg (np.array): Second input image of the same size and the same type as prevImg.
        param_lambda (float): Smoothness weight. The larger it is, the smoother optical flow map you get.
        param_delta (float): pre-set threshold for determing convergence between iterations.

    Returns:
        flow (np.array): Computed flow image that has the same size as prevImg and single 
            type (2-channels). Flow for (x,y) is stored in the third dimension.
        
    """
    # Your code starts here #
    # Calculate image gradients
    # Ix = np.gradient(prevImg, axis=1)  # Gradient along X
    # Iy = np.gradient(prevImg, axis=0)  # Gradient along Y
    # It = nextImg - prevImg               # Temporal gradient


    # Apply Gaussian blur for smoothing
    sigma = 1  # Standard deviation for Gaussian kernel
    prevImg_smooth = gaussian_filter(prevImg, sigma=sigma)
    nextImg_smooth = gaussian_filter(nextImg, sigma=sigma)

    # Compute gradients
    Ix = filters.sobel_h(prevImg_smooth)  # Gradient along X
    Iy = filters.sobel_v(prevImg_smooth)  # Gradient along Y
    It = nextImg_smooth - prevImg_smooth             # Temporal gradient

    # Ix = np.gradient(prevImg_smooth, axis=1)  # Gradient along X
    # Iy = np.gradient(prevImg_smooth, axis=0)  # Gradient along Y
    # It = nextImg_smooth - prevImg_smooth               # Temporal gradient

    # Initialize flow fields
    u = np.zeros(prevImg.shape)
    v = np.zeros(prevImg.shape)

    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4

    # Iterate until convergence
    i = 0
    for _ in range(1000):
        u_avg = cv2.filter2D(u, -1, kernel)
        v_avg = cv2.filter2D(v, -1, kernel)

        # Compute flow updates
        p = Ix * u_avg + Iy * v_avg + It
        u_numerator = p * Ix
        v_numerator = p * Iy
        denominator = param_lambda**-1 + Ix**2 + Iy**2

        u_delta = u_numerator / denominator
        v_delta = v_numerator / denominator

        u_prev = u
        v_prev = v

        # Update flow fields
        u = u_avg - u_delta
        v = v_avg - v_delta
        i += 1

        # Check for convergence
        if np.linalg.norm(u - u_prev, 2) < param_delta and np.linalg.norm(v - v_prev, 2) < param_delta:
            break
    print(i)

    flow = np.stack((u, v), axis=-1)

    # Your code ends here #

    return flow


### Task 1 Auxiliary function

def combine_and_normalize_features(feat1: np.array, feat2: np.array, gamma: float) -> np.array:
    """Combine two features together with proper normalization.

    Args:
        feat1 (np.array): of size (..., N1).
        feat2 (np.array): of size (..., N2).

    Returns:
        feats (np.array): combined features of size of size (..., N1+N2), with feat2 weighted by gamma.
        
    """
    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    feats = np.concatenate((normalize(feat1), normalize(feat2)*gamma), axis=-1)
    
    return feats


def build_gaussian_kernel(sigma: int) -> np.array:

    def gaussianKernel(sigma):
        halfSize = int(np.ceil(3.0*sigma))
        kernel = np.zeros((2*halfSize+1, 1))
        s2 = sigma * sigma
        f = 1.0 / np.sqrt(2.0 * np.pi * s2)
        w2 = 1.0 / (2.0 * s2)
        for i in range(2*halfSize+1):
            p = i - halfSize
            kernel[i] = f * np.exp(-(p * p) * w2)
        return kernel

    g = gaussianKernel(sigma)

    kernel = g @ g.transpose()

    return kernel

def build_gaussian_derivative_kernel(sigma: int) -> np.array:
    
    def gaussianKernel(sigma):
        halfSize = int(np.ceil(3.0*sigma))
        kernel = np.zeros((2*halfSize+1, 1))
        s2 = sigma * sigma
        f = 1.0 / np.sqrt(2.0 * np.pi * s2)
        w2 = 1.0 / (2.0 * s2)
        for i in range(2*halfSize+1):
            p = i - halfSize
            kernel[i] = f * np.exp(-(p * p) * w2)
        return kernel
    
    def gaussianDerivativeKernel(sigma):
        halfSize = int(np.ceil(3.0*sigma))
        kernel = np.zeros((2*halfSize+1, 1))
        s2 = sigma * sigma
        f = 1.0 / np.sqrt(2.0 * np.pi * s2)
        w = 1.0 / (s2)
        w2 = 1.0 / (2.0 * s2)
        for i in range(2*halfSize+1):
            p = i - halfSize
            kernel[i] = - p * w * f * np.exp(-(p * p) * w2)
        return kernel

    dg = gaussianDerivativeKernel(sigma)
    g = gaussianKernel(sigma)


    kernel_y = dg @ g.transpose()
    kernel_x = g @ dg.transpose()
    
    return kernel_y, kernel_x


def build_LoG_kernel(sigma: int) -> np.array:
    
    def gaussianKernel(sigma):
        halfSize = int(np.ceil(3.0*sigma))
        kernel = np.zeros((2*halfSize+1, 1))
        s2 = sigma * sigma
        f = 1.0 / np.sqrt(2.0 * np.pi * s2)
        w2 = 1.0 / (2.0 * s2)
        for i in range(2*halfSize+1):
            p = i - halfSize
            kernel[i] = f * np.exp(-(p * p) * w2)
        return kernel

    g1 = gaussianKernel(sigma)

    kg1 = g1 @ g1.transpose()

    kernel = cv2.Laplacian(kg1, -1)

    
    return kernel

# ================ TASK 2.1 ================ #

from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KDTree

class Textonization:
    def __init__(self, kernels, n_clusters=200):
        self.n_clusters = n_clusters
        self.kernels = kernels

    def training(self, training_imgs):
        """Takes all training images as input and stores the clustering centers for testing.

        Args:
            training_imgs (list): list of training images.
            
        """
        # Your code starts here #
        responses = []
        for img in training_imgs:
            response = features_from_filter_bank(img, self.kernels)
            responses.append(response.reshape(-1, 17))
        responses = np.vstack(responses)
        # print(responses.shape)
        
        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=88)
        kmeans.fit(responses)
        self.cluster_centers = kmeans.cluster_centers_
        # print(self.cluster_centers.shape)
        # Your code ends here #
        
        pass

    def testing(self, img):
        """Predict the texture label for each pixel of the input testing image. For each pixel in the test image, an ID from a learned texton dictionary can represent it. 

        Args:
            img (np.array): of size (..., 3).
            
        Returns:
            textons (np.array): of size (..., 1).
        
        """
        # Your code starts here #
        responses = features_from_filter_bank(img, self.kernels)
        responses = responses.reshape(-1, 17)
        tree = KDTree(self.cluster_centers)
        _, indices = tree.query(responses)
        textons = indices.reshape(img.shape[0], img.shape[1], 1)
        # Your code ends here #
        
        return textons

### Task 2 Auxiliary function

def features_from_filter_bank(image, kernels):
    """Returns 17-dimensional feature vectors for the input image.

    Args:
        img (np.array): of size (..., 3).
        kernels (dict): dictionary storing gaussian, gaussian_derivative, and LoG kernels.

    Returns:
        feats (np.array): of size (..., 17).
        
    """
    from scipy import ndimage
    import skimage
    from scipy.cluster.vq import whiten

    ddepth = -1
    c = []
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    w, h, _ = lab_image.shape
    for i in kernels['gaussian']:
        c.append(cv2.filter2D(lab_image[:,:,0], ddepth, i))
        c.append(cv2.filter2D(lab_image[:,:,1], ddepth, i))
        c.append(cv2.filter2D(lab_image[:,:,2], ddepth, i))

    for i in kernels['gaussian_derivative']:
        c.append(cv2.filter2D(lab_image[:,:,0], ddepth, i))
        
    for i in kernels['LoG']:
        c.append(cv2.filter2D(lab_image[:,:,0], ddepth, i))
  
    
    c = np.stack(c, axis = -1)
    c = c.reshape(w*h, 17)
    feats = whiten(c).reshape(w,h,17)
    return feats


def histogram_per_pixel(textons, window_size):
    """ Compute texton histogram by computing the distribution of texton indices within the window.

    Args:
        textons (np.array): of size (..., 1).
        
    Returns:
        hists (np.array): of size (..., 200).
    
    """
    h, w, d = img.shape
    img = img.reshape(h,w,1).astype(np.float32)
    hists = np.zeros((h,w,200))
    for i in range(h):
        for j in range(w):
            top = np.clip(i - window_size//2, 0, h-1)
            bot = np.clip(i + window_size//2 + 1, 0, h-1)
            left = np.clip(j - window_size//2, 0, w-1)
            right = np.clip(j + window_size//2+1, 0, w-1)
            #hist, edge = np.histogram(img[top:bot,left:right], bins=np.arange(201), density=False)
            #hist,_ = np.histogram(img, np.arange(0, 256), normed=True)
            hist = cv2.calcHist([img[top:bot,left:right]],[0],None,[200],[0,200])
            hists[i,j]=hist.reshape((200,))
    
    return hists


